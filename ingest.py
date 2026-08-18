import os
import json
import shutil
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

CONFIG_PATH = "/home/may/d/chatbot/config.json"

def load_company_lookup():
    with open(CONFIG_PATH, 'r') as f:
        companies = json.load(f)['companies']
    return {name.lower().strip(): name for name in companies}

def ingest_faqs(csv_dir, db_path):
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    company_lookup = load_company_lookup()

    for filename in os.listdir(csv_dir):
        if not filename.endswith(".csv"):
            continue

        base_name = filename.replace(".csv", "")
        derived_name = base_name.replace("faq_", "").replace("_faqs", "").replace("_", " ").title()

        company_name = company_lookup.get(derived_name.lower().strip())
        if company_name is None:
            print(f"'{filename}' → '{derived_name}' has no matching entry in config.json. "
                  f"Add it to 'companies' so the chatbot can find it. Skipping for now.")
            continue

        print(f"Processing {company_name}...")
        sanitized_collection_name = company_name.replace(" ", "_")
        
        db = Chroma(
            collection_name=sanitized_collection_name, 
            embedding_function=embeddings,
            persist_directory=db_path
        )

        df = pd.read_csv(os.path.join(csv_dir, filename))
        
        headers = df.columns.tolist()
        
        id_col = headers[0] 
        
        q_col = [c for c in headers if 'question' in str(c).lower()][0]
        a_col = [c for c in headers if 'answer' in str(c).lower()][0]

        documents = []
        metadatas = []
        ids = []

        for index, row in df.iterrows():
            raw_id = str(row[id_col]) if id_col in row.index else f"{sanitized_collection_name}_{index}"
            q_text = str(row[q_col])
            a_text = str(row[a_col])
            
            doc_content = f"Q: {q_text}\nA: {a_text}"
            
            documents.append(doc_content)
            
            metadatas.append({
                "source_file": filename, 
                "company": company_name,
                "doc_id": raw_id,           
                "question_text": q_text,   
                "answer_text": a_text
            })
            
            ids.append(f"{sanitized_collection_name}_{index}") 
            
        db.add_texts(texts=documents, metadatas=metadatas, ids=ids)
        print(f"Done ingesting {company_name}.")

if __name__ == "__main__":
    CSV_DIR = "/home/may/d/chatbot/data"
    DB_PATH = "/home/may/d/chatbot/chroma_db"
    
    ingest_faqs(CSV_DIR, DB_PATH)
