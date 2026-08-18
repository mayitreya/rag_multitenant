import sys
import os
import json
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import chromadb

CONFIG_PATH = "/home/may/d/chatbot/config.json"

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

config = load_config()
COMPANIES = config['companies']

COMPANY_LOOKUP = {name.lower().strip(): name for name in COMPANIES}

def resolve_company_name(raw_name):
    if not raw_name:
        return None
    return COMPANY_LOOKUP.get(raw_name.lower().strip())

DB_PATH = "/home/may/d/chatbot/chroma_db"
CSV_DIR = "/home/may/d/chatbot/data"

company_cache = {}
chat_history = [] 

def get_context_for_company(company_name):
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collections = client.list_collections()
        sanitized = company_name.replace(" ", "_")
        return sanitized in [c.name for c in collections]
    except Exception as e:
        print(f"Error checking collection existence: {e}")
        return False

def detect_company(query):
    router_llm = ChatOllama(model=config['default_model'], temperature=0.1, reasoning=False)
    
    prompt = f"""Analyze the following customer query and identify which company it belongs to.
    The available companies in our system are: {COMPANIES}

    User Query: "{query}"

    Output ONLY a valid JSON object in this format:
    {{ "company": "Company Name", "confidence": 0.0-1.0, "reasoning": "brief explanation" }}
    
    Rules:
    1. If the query is ambiguous (e.g., "order"), choose the most likely one but set confidence low if unsure.
    2. If you are not sure which company it belongs to, return confidence < 0.7 so I can ask for clarification.
    """

    response = router_llm.invoke(prompt)
    content = response.content.strip()
    
    if "```" in content:
        content = content.replace("```json", "").replace("```", "").strip()
        
    try:
        parsed = json.loads(content)
        parsed["company"] = resolve_company_name(parsed.get("company"))
        return parsed
    except json.JSONDecodeError:
        print(f"Router parsing error for query: '{query}'. Response was: {content}")
        return {"company": None, "confidence": 0.1}

def cleanup_cache(detected_name):
    if detected_name in company_cache:
        del company_cache[detected_name]
        print(f"Removed {detected_name} from cache.")

FALLBACK_MESSAGE = "I don't have that specific information in my current FAQ database."

ANSWER_TEMPLATE = """You are a professional customer support assistant for {company}.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Provide a helpful, paraphrased answer based on the provided CONTEXT. Combine relevant information smoothly.
2. Do not use outside knowledge. If the context is missing the info, say EXACTLY: "I don't have that specific information in my current FAQ database."

AFTER your answer, strictly list the sources using this format (one per line) ONLY if you have sources. If you couldn't find specific information in the current FAQ database, your response can stop there. Otherwise:

<top_sources>
{company} FAQ ID <ID>: "<Verbatim Text>"
</top_sources>

- Do not include any introductory text like "Sources:" or "References:".
- If multiple sources apply, list all of them.
- Quote the text verbatim inside the quotes."""


def format_context(docs):
    if not docs:
        return "No relevant information found in the FAQ database."
    context_lines = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        doc_id = meta.get("doc_id", f"Unknown-{i}")
        answer = meta.get("answer_text", "No answer found")
        question_display = meta.get("question_text", "")
        context_lines.append(f"[FAQ ID: {doc_id}] Question: {question_display}\nAnswer: {answer}")
    return "\n\n".join(context_lines)

def run_chatbot():
    global chat_history
    
    llm = ChatOllama(
        model=config['default_model'], 
        temperature=0.1,       
        num_ctx=4096,          
        reasoning=False
    )

    print(f"Loaded {len(COMPANIES)} companies from config.json")
    
    current_context = None
    context_locked = False

    while True:
        question = input("\nYou (type '/quit' to exit, '/list' for commands): ")
        
        if question.lower() in ['/quit', 'exit']:
            print("Cleaning up memory...")
            for company in list(company_cache.keys()):
                cleanup_cache(company)
            break
        
        if question.startswith('/context'):
            parts = question.split()
            if len(parts) > 1:
                arg = " ".join(parts[1:]).strip()

                if arg.lower() in ('off', 'clear', 'auto', 'none'):
                    current_context = None
                    context_locked = False
                    print("Context unpinned. Back to automatic routing.")
                    continue

                target = resolve_company_name(arg)
                if target:
                    current_context = target
                    context_locked = True
                    print(f"Context pinned to {target}. (Use '/context off' to return to auto-routing.)")
                    cleanup_cache(current_context)
                else:
                    print(f"Invalid company. Available: {', '.join(COMPANIES)}")
            else:
                if context_locked:
                    print(f"Currently pinned to {current_context}. Use '/context <company>' to switch, or '/context off' to unpin.")
                else:
                    print("No context pinned (automatic routing). Use '/context <company>' to pin one.")
            continue
        
        if question.startswith('/list'):
            print("Available Contexts:")
            for c in COMPANIES:
                status = "Loaded" if c in company_cache else "Unloaded"
                print(f"- {c} ({status})")
            continue

        if not question.strip():
            continue

        if context_locked:
            detected_name = current_context
        else:
            detection = detect_company(question)

            if current_context and detection["confidence"] < 0.6:
                 detected_name = current_context
            else:
                detected_name = detection.get("company")
                confidence = detection.get("confidence", 0.0)

                if confidence < 0.7 and not current_context:
                    print(f"\nI'm a bit confused about the context. Did you mean one of these?")
                    for c in COMPANIES[:5]: print(f"- {c}")
                    print("Please rephrase, or set it directly with /context <company>.")
                    continue

        if not detected_name:
            print(f"Please type a company name manually (e.g., '{COMPANIES[0]}') or use /list.")
            manual_input = input("> ")
            detected_name = resolve_company_name(manual_input)
            if detected_name:
                current_context = detected_name 
                cleanup_cache(detected_name)  
            continue

        if not get_context_for_company(detected_name):
             print(f"Warning: {detected_name} database has not been ingested yet. Please add files to /data.")
             continue

        if detected_name not in company_cache:
            print(f"Loading {detected_name} Database...")
            sanitized_name = detected_name.replace(" ", "_")
            
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            
            db = Chroma(
                collection_name=sanitized_name, 
                embedding_function=embeddings,
                persist_directory=DB_PATH
            )
            
            company_cache[detected_name] = {
                "db": db,
                "retriever": db.as_retriever(search_kwargs={"k": 5}),
                "company_name": detected_name
            }

        bot_config = company_cache[detected_name]
        
        docs = bot_config["retriever"].invoke(question)
        
        context_text = format_context(docs)

        chat_history.append(f"System: Active context is now {detected_name}")

        prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
        qa_chain = prompt | llm
        
        print(f"\nBot ({detected_name}): ", end="", flush=True)
        
        try:
            response = ""
            for chunk in qa_chain.stream({
                "context": context_text, 
                "question": question, 
                "company": bot_config["company_name"],
                "history": "\n".join(chat_history[-8:]) if chat_history else ""
            }):
                print(chunk.content, end="", flush=True)
                response += chunk.content
            
            chat_history.append(f"User: {question}")
            chat_history.append(f"Bot: {response.strip()}")
                
        except Exception as e:
            print(f"\nError occurred: {e}")
            continue
            
        print("\n")

if __name__ == "__main__":
    run_chatbot()
