"""
eval.py — Evaluation harness for the Multi-Tenant RAG FAQ chatbot.

Runs a fixed set of test questions through the SAME pipeline the chatbot uses
(LLM router -> per-company retrieval -> grounded generation) and reports, per
question:
  * whether the router picked the right company, and
  * whether the bot answered from the FAQ, or correctly refused ("I don't have
    that information") on the trick questions that have no answer in the data.

Because both the router and the generator run at temperature=0.1, results are
stable from run to run.

Usage:
    python eval.py              # run the full suite
    python eval.py -v           # also print each bot answer in full
    python eval.py -n 3         # smoke-test: only the first 3 cases
"""

import argparse

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Reuse the real pipeline so the eval exercises the actual system, not a copy.
from app import (
    config,
    DB_PATH,
    detect_company,
    get_context_for_company,
    format_context,
    ANSWER_TEMPLATE,
    FALLBACK_MESSAGE,
)

# ---------------------------------------------------------------------------
# TEST SET
#   expect_answer=True  -> the answer IS in that company's FAQ; the bot should
#                          route correctly and answer from it.
#   expect_answer=False -> "trick" question: clearly in the company's domain but
#                          NOT in its FAQ, so the bot should refuse with the
#                          fallback instead of inventing an answer.
# Questions are paraphrased (not copied from the CSVs) to test that retrieval
# works semantically rather than on exact wording.
# ---------------------------------------------------------------------------
TEST_CASES = [
    # --- Answerable: one per industry -------------------------------------
    {"question": "My tracking says delivered but the package never showed up. What do I do?",
     "expected_company": "Ecommerce", "expect_answer": True},
    {"question": "How much do I need to keep in my savings account to avoid the monthly fee?",
     "expected_company": "Banking", "expect_answer": True},
    {"question": "How can I change the credit card you bill for my subscription?",
     "expected_company": "SaaS", "expect_answer": True},
    {"question": "How heavy can my checked bag be if I'm flying economy?",
     "expected_company": "Airline", "expect_answer": True},
    {"question": "My delivery arrived cold — can I get my money back?",
     "expected_company": "Food Delivery", "expect_answer": True},
    {"question": "How do I book a new appointment with a doctor?",
     "expected_company": "Healthcare Provider", "expect_answer": True},
    {"question": "What internet speeds and prices do you offer?",
     "expected_company": "Telecom", "expect_answer": True},
    {"question": "Can you explain what a deductible is?",
     "expected_company": "Insurance", "expect_answer": True},
    {"question": "How do I save listings I like so I can find them again later?",
     "expected_company": "Real Estate", "expect_answer": True},
    {"question": "Is there a free trial for the courses?",
     "expected_company": "Online Education Platform", "expect_answer": True},

    # --- Trick questions: in-domain, but no answer exists in the FAQ ------
    {"question": "What was your bank's net profit last quarter?",
     "expected_company": "Banking", "expect_answer": False},
    {"question": "Which specific pilot will be flying my plane tomorrow?",
     "expected_company": "Airline", "expect_answer": False},
    {"question": "Can you write me a Python script to scrape my account dashboard?",
     "expected_company": "SaaS", "expect_answer": False},
]


def generate_answer(company_name, question, llm, embeddings):
    """Retrieve top-k for the company and run the grounded prompt.

    Returns the model's response text, or None if that company has no
    ingested collection.
    """
    if not get_context_for_company(company_name):
        return None
    sanitized = company_name.replace(" ", "_")
    db = Chroma(
        collection_name=sanitized,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    context_text = format_context(docs)

    chain = ChatPromptTemplate.from_template(ANSWER_TEMPLATE) | llm
    response = chain.invoke(
        {"context": context_text, "question": question, "company": company_name}
    )
    return response.content.strip()


def run_case(case, llm, embeddings):
    """Route -> answer -> grade a single test case."""
    question = case["question"]
    expected = case["expected_company"]
    expect_answer = case["expect_answer"]

    detection = detect_company(question)
    routed = detection.get("company")
    confidence = detection.get("confidence", 0.0)

    # Mirror the app: if the router is unsure it asks to clarify rather than
    # answering, which we treat as a refusal (no fabricated answer).
    answer = generate_answer(routed, question, llm, embeddings) if routed else None
    refused = (answer is None) or (FALLBACK_MESSAGE in answer)

    routed_ok = (routed == expected)
    if expect_answer:
        # Must reach the right company AND actually answer from the FAQ.
        passed = routed_ok and not refused
    else:
        # Trick question: success = the bot did not invent an answer.
        passed = refused

    return {
        "question": question,
        "expected": expected,
        "expect_answer": expect_answer,
        "routed": routed,
        "confidence": confidence,
        "routed_ok": routed_ok,
        "answer": answer,
        "refused": refused,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG FAQ chatbot.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print each bot answer in full")
    parser.add_argument("-n", "--limit", type=int, default=None,
                        help="only run the first N cases (smoke test)")
    args = parser.parse_args()

    cases = TEST_CASES[: args.limit] if args.limit else TEST_CASES

    llm = ChatOllama(
        model=config["default_model"], temperature=0.1, num_ctx=4096, reasoning=False
    )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print(f"Running {len(cases)} test cases against model "
          f"'{config['default_model']}'...\n")

    results = []
    for i, case in enumerate(cases, 1):
        r = run_case(case, llm, embeddings)
        results.append(r)

        kind = "answer" if r["expect_answer"] else "trick "
        verdict = "PASS" if r["passed"] else "FAIL"
        outcome = "answered" if not r["refused"] else "refused"
        routed = r["routed"] or "—"
        print(f"[{i:2}/{len(cases)}] {verdict}  ({kind})  "
              f"expected={r['expected']:<26} routed={routed:<26} "
              f"conf={r['confidence']:<4} -> {outcome}")
        if args.verbose and r["answer"]:
            print("-" * 70)
            print(r["answer"])
            print("-" * 70 + "\n")

    # --- Summary ----------------------------------------------------------
    total = len(results)
    passed = sum(r["passed"] for r in results)
    routing_ok = sum(r["routed_ok"] for r in results)
    print(f"\nSummary: {passed}/{total} passed  |  "
          f"routing {routing_ok}/{total} correct")

    # --- Markdown table (paste-ready for the write-up) --------------------
    print("\n--- Markdown results table ---\n")
    print("| # | Question | Expected | Routed | Outcome | Pass |")
    print("|---|----------|----------|--------|---------|------|")
    for i, r in enumerate(results, 1):
        want = "answer" if r["expect_answer"] else "refuse (trick)"
        outcome = "answered" if not r["refused"] else "refused"
        mark = "✅" if r["passed"] else "❌"
        q = r["question"].replace("|", "\\|")
        print(f"| {i} | {q} | {r['expected']} ({want}) | "
              f"{r['routed'] or '—'} | {outcome} | {mark} |")


if __name__ == "__main__":
    main()
