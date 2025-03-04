import os 
import tqdm 
import json
from typing import Optional, List, Tuple
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from openai import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer


load_dotenv()

def convert_context_to_langchain_docs(df):

    langchain_docs = []

    for i, row in df.drop_duplicates(subset="context").iterrows():
        context = row.context
        title = row.title
        
        document = LangchainDocument(
            page_content=context,
            metadata={"title": title})

        langchain_docs.append(document)

    return langchain_docs


def split_documents(
    chunk_size: int,
    chunk_overlap: int, 
    knowledge_base: List[LangchainDocument],
    #tokenizer_name: str,
    ) -> List[LangchainDocument]:
    """Chunks Langchain documents. """
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / chunk_overlap),
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    
    docs_processed = []
    
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    """
    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)"""

    return docs_processed 


def create_vector_store(docs,
                        embeddings,
                        store_name, 
                        db_dir, 
                        chunk_size: int = 200,
                        chunk_overlap: int = 15):
    """Vector embeddings and storage in vector database. """

    docs_processed = split_documents(chunk_size, chunk_overlap, docs)

    persistent_directory = os.path.join(db_dir, store_name)
    
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(
            docs_processed, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")
        


def query_vector_store(store_name, query, embedding_function, db_dir, k=3, score_threshold=0.1):
    persistent_directory = os.path.join(db_dir, store_name)

    relevant_docs = []
    
    if os.path.exists(persistent_directory):
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )
        relevant_docs = retriever.invoke(query)
        
        # Display the relevant results with metadata
        #print(f"\n--- Relevant Documents for {store_name} ---")
        #for i, doc in enumerate(relevant_docs, 1):
         #   print(f"Document {i}:\n{doc.page_content}\n")

    else:
        print(f"Vector store {store_name} does not exist.")

    return relevant_docs


def save_llm_answers(df, 
                    docs, 
                    embeddings,
                    embeddings_name="text-embedding-ada-002", 
                    model_name="gpt-3.5-turbo",
                    chunk_size=200, 
                    num_context_documents=3,
                    filename="pred_500.json"):

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    df_to_test = df.copy()
    
    cwd = os.getcwd()
    db_dir = os.path.join(cwd, "vector_databases", f"db_{chunk_size}_{embeddings_name}")

    print(db_dir)

    create_vector_store(docs, embeddings, "chroma_db_openai", db_dir, chunk_size=chunk_size)

    answers = []
    contexts = []

    for question in tqdm.tqdm(df_to_test.question):

        relevant_docs = query_vector_store("chroma_db_openai", question, embeddings, db_dir, k=num_context_documents)

        # Concatenate all relevant documents with numbering
        context = "\n\n".join([f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(relevant_docs)])

        # Create improved structured prompt
        prompt = f"""
        You are a highly accurate and reliable assistant. Answer the user's question using **only** the provided context. 
        If the answer is not in the context, return an empty response (**""**) without making up information.

        Context:
        {context}

        Instructions:
        - Answer concisely and precisely.
        - If the answer is explicitly stated in the context, extract it as-is.
        - If the answer is not in the context, return **""** (empty string).
        - Do **not** infer, assume, or add external information.

        Example:
            **Question:** What is the capital of Italy?
            **Answer:** Rome

        Question: {question}
        Answer (just the answer, no extra words, or "" if unknown):
        """

        # Get response from OpenAI
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ]
        )
        
        answer = response.choices[0].message.content
        
    # reference = context[0]  # Using first context as reference
    # the reference is the GT?

        answers.append(answer)
        contexts.append(context)
  
    preds = dict(zip(df_to_test["id"], answers))

    results_folder = "eval_results"
    filepath = os.path.join(results_folder, filename)

    with open(filepath, "w") as f:
        json.dump(preds, f, indent=4, sort_keys=True)


def fine_tune_rag(df, langchain_docs):
    # helpful function to merge all the scores to the questions df, for debugging purposes 

    # TODO: try different text splitters
    # TODO: improve num of documents retrieved and the retrieval threshold 

    #chunk_sizes = [100, 200, 300, 400, 500, 800, 1200, 1600]
    chunk_sizes =  [100] #[600, 800]

    models = ["gpt-3.5-turbo"]

    embed_options = {
        "text-embedding-3-small": "OpenAI", 
        "text-embedding-3-large": "OpenAI", 
        #"text-embedding-ada-002": "OpenAI" # poor performance 
        #"voyageai/voyage-3-m-exp": "custom" # best retrieval model mar 2025 based on HuggingFace MTEB leaderboard, but proprietary model, paid 
        # "Snowflake/snowflake-arctic-embed-l-v2.0": "HuggingFace_SentenceTransformers"# ranked 6th, 568M params
    }
                  
   
    for embeddings_name, embeddings_platform in embed_options.items(): 
        cache_dir = './model_cache'

        if embeddings_platform == "OpenAI": 
            embeddings = OpenAIEmbeddings(model=embeddings_name)
        elif embeddings_platform == "HuggingFace":
            embeddings = HuggingFaceEmbeddings(model_name=embeddings_name, show_progress=True)
        elif embeddings_platform == "HuggingFace_SentenceTransformers":
            embeddings = SentenceTransformer(embeddings_name, cache_folder=cache_dir) 
       # elif embeddings_platform == "custom":
        #    cache_dir = './model_cache'
            #embeddings = VoyageEmbeddings(voyage_api_key="", model="voyage-3") # proprietary, paid 
        else:
            print("Embeddings error")
        
        for model_name in models: 
            for chunk_size in chunk_sizes:
                save_llm_answers(
                    df,
                    langchain_docs,
                    embeddings, 
                    embeddings_name=embeddings_name,
                    model_name=model_name, 
                    chunk_size=chunk_size, 
                    num_context_documents=3,
                    filename=f"pred_500_{chunk_size}_{embeddings_name}_{model_name}.json"
                ) 