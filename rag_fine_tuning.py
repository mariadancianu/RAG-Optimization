import os 
import tqdm 
import json
from typing import List
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from openai import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import pprint
import chromadb
from mistralai import Mistral

load_dotenv()


def convert_context_to_langchain_docs(df):
    
    langchain_docs = []

    for _, row in df.drop_duplicates(subset="context").iterrows():
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


def create_vector_store_new(docs, 
                            embeddings_function, 
                            store_name, 
                            db_dir, 
                            chunk_size: int = 200,
                            chunk_overlap: int = 15):

    docs_processed = split_documents(chunk_size=chunk_size, chunk_overlap=chunk_overlap, knowledge_base=docs)

    docs = [doc.page_content for doc in docs_processed]

    document_embeddings = embeddings_function.encode(docs) 

    persistent_directory = os.path.join(db_dir, store_name)

    chroma_client = chromadb.PersistentClient(path=persistent_directory)
    collection = chroma_client.get_or_create_collection(name=store_name)

    ids = [f"id{i}" for i in list(range(len(docs)))]

    # Ensure the number of IDs matches the number of documents
    # note: use upsert instead of add to avoid adding existing documents 
    collection.upsert(
        ids=ids,  
        documents=docs, 
        embeddings=document_embeddings  
    )


def query_vector_store_new(store_name, 
                           query, 
                           embeddings_function,
                           db_dir, 
                           n_results=3, 
                           score_threshold=0.1):

    persistent_directory = os.path.join(db_dir, store_name)

    relevant_docs = []

    if os.path.exists(persistent_directory):
        query_embeddings = embeddings_function.encode(query)

        chroma_client = chromadb.PersistentClient(path=persistent_directory)
        collection = chroma_client.get_or_create_collection(name=store_name)

        relevant_docs = collection.query(query_embeddings=query_embeddings, n_results=n_results)
    else:
        print(f"Vector store {store_name} does not exist.")

    return relevant_docs


def query_vector_store(store_name, query, embedding_function, db_dir, n_results=3, score_threshold=0.1):
    persistent_directory = os.path.join(db_dir, store_name)

    relevant_docs = []
    
    if os.path.exists(persistent_directory):
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": n_results, "score_threshold": score_threshold},
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
                     embeddings_function,
                     embeddings_name="text-embedding-ada-002", 
                     model_name="gpt-3.5-turbo",
                     chunk_size=200, 
                     num_context_documents=3,
                     filename="result.json", 
                     results_folder="eval_results",
                     client="openai",
                     save_context=False):

    if client == "openai": 
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    elif client == "mistral": 
        client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    else:
        print("Invalid client!")

    df_to_test = df.copy()
    
    cwd = os.getcwd()
    db_dir = os.path.join(cwd, "vector_databases", f"db_{chunk_size}_{embeddings_name}")

    print(db_dir)

    # TODO: fix collection names - it's not always openai
    create_vector_store_new(docs, embeddings_function, "chroma_db_openai", db_dir, chunk_size=chunk_size)

    answers = []
    contexts = []

    for question in tqdm.tqdm(df_to_test.question):

        relevant_docs = query_vector_store_new("chroma_db_openai", question, embeddings_function, db_dir, n_results=num_context_documents)

        context = relevant_docs["documents"]

        # Concatenate all relevant documents with numbering
        context = "\n\n".join([f"Source {i+1}: {doc}" for i, doc in enumerate(context[0])])

        #print(context)

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
        answers.append(answer)

        if save_context:
            contexts.append(context)
  
    preds_dict = dict(zip(df_to_test["id"], answers))

    filepath_pred = os.path.join(results_folder, f"pred_{filename}")

    with open(filepath_pred, "w") as f:
        json.dump(preds_dict, f, indent=4, sort_keys=True)
    
    if save_context:
        contexts_dict = dict(zip(df_to_test["id"], contexts))

        filepath_context = os.path.join(results_folder, f"context_{filename}")

        with open(filepath_context, "w") as f:
            json.dump(contexts_dict, f, indent=4, sort_keys=True)

parameters_dict = {
    "chunk_sizes": [100, 200, 400, 500, 600],
    "embed_options": { 
        "text-embedding-3-small": "OpenAI", 
        "text-embedding-3-large": "OpenAI", 
        "text-embedding-ada-002": "OpenAI",
         #"voyageai/voyage-3-m-exp": "custom" # best retrieval model mar 2025 based on HuggingFace MTEB leaderboard, but proprietary model, paid 
        # "Snowflake/snowflake-arctic-embed-l-v2.0": "HuggingFace_SentenceTransformers"# ranked 6th, 568M params, released in december 2024 
        },
    "models": {"gpt-3.5-turbo": "openai", 
               "mistral-large-latest": "mistral"}
}


def fine_tune_rag(df, 
                  langchain_docs, 
                  parameters_dict=parameters_dict,
                  results_folder="eval_results", 
                  save_context=False):
   
    print("Fine tuning RAG with the following parameters: ")
    pprint.pprint(parameters_dict)

    chunk_sizes = parameters_dict.get("chunk_sizes", [100])
    embed_options = parameters_dict.get("embed_options", {"text-embedding-3-small": "OpenAI"})
    models = parameters_dict.get("models", {"gpt-3.5-turbo": "openai"})
   
    for embeddings_name, embeddings_platform in embed_options.items(): 
        cache_dir = './model_cache'

        if embeddings_platform == "OpenAI": 
            embeddings_function = OpenAIEmbeddings(model=embeddings_name)
        elif embeddings_platform == "HuggingFace":
            embeddings_function = HuggingFaceEmbeddings(model_name=embeddings_name, show_progress=True)
        elif embeddings_platform == "HuggingFace_SentenceTransformers":
            embeddings_function = SentenceTransformer(embeddings_name, cache_folder=cache_dir) 
        else:
            print("Embeddings error")

        """
        elif embeddings_platform == "custom":
            cache_dir = './model_cache'
            #embeddings = VoyageEmbeddings(voyage_api_key="", model="voyage-3") # proprietary, paid 
        """
        
        for model_name, client in models.items(): 
            for chunk_size in chunk_sizes:
                print(f"Running {model_name} - {chunk_size} - {embeddings_name}")

                filename = f"500_{chunk_size}_{embeddings_name}_{model_name}.json"
                filepath = os.path.join(results_folder, f"pred_{filename}")

                if os.path.isfile(filepath):
                    print("Results already exist for these settings: skipping!")
                    continue 

                save_llm_answers(
                    df,
                    langchain_docs,
                    embeddings_function, 
                    embeddings_name=embeddings_name,
                    model_name=model_name, 
                    chunk_size=chunk_size, 
                    num_context_documents=3,
                    filename=filename,
                    results_folder=results_folder,
                    client=client,
                    save_context=save_context
                ) 

