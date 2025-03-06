import json
import os
import pprint
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import pandas as pd
import tqdm
from dotenv import load_dotenv
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from mistralai import Mistral
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

prompt_message = f"""
You are a highly accurate and reliable assistant. Answer the user's question using **only** the provided context.
If the answer is not in the context, return an empty response (**""**) without making up information.

Context:
%s

Instructions:
- Answer concisely and precisely.
- If the answer is explicitly stated in the context, extract it as-is.
- If the answer is not in the context, return **""** (empty string).
- Do **not** infer, assume, or add external information.

Example:
    **Question:** What is the capital of Italy?
    **Answer:** Rome

Question: %s
Answer (just the answer, no extra words, or "" if unknown):
"""


class JSONDefaultConfigFile:
    def __init__(self, file_path: Optional[str] = None, default: Dict[str, Any] = {}):
        """
        Initialize the JSONDefaultConfigFile with a file path and default properties.

        Args:
            file_path (Optional[str]): Path to the configuration file.
            default (Dict[str, Any]): Default properties for the configuration.
        """
        if file_path is None:
            self.file_path = "/Users/mariadancianu/Desktop/Git Projects/SQuAD_RAG_experiments/default_rag_config.json"
        else:
            self.file_path = file_path

        self._default_properties = default

    def get(self) -> Dict[str, Any]:
        """
        Get the configuration from the file, updating with default properties.

        Returns:
            Dict[str, Any]: The configuration dictionary.
        """
        configurations = deepcopy(self._default_properties)

        with open(self.file_path) as file:
            configurations.update(json.load(file))

        return configurations


class CustomRAG:
    def __init__(
        self,
        knowledge_base: List[LangchainDocument],
        prompt_message: str,
        config: Optional[Dict[str, Any]] = None,
        results_folder: Optional[str] = None,
        vector_db_folder: Optional[str] = None,
    ):
        """
        Initialize the CustomRAG class with the given parameters.

        Args:
            knowledge_base (List[LangchainDocument]): The knowledge base documents.
            prompt_message (str): The prompt message for the LLM.
            config (Optional[Dict[str, Any]]): Configuration dictionary.
            results_folder (Optional[str]): Folder to save results.
            vector_db_folder (Optional[str]): Folder to save vector database.
        """
        config_file = JSONDefaultConfigFile()
        self.default_rag_config = config_file.get()

        if config is None:
            print("Warning: the RAG configurations are missing! Using the default ones")
            config = self.default_rag_config

        if results_folder is None:
            results_folder = os.getcwd()

        if vector_db_folder is None:
            vector_db_folder = os.getcwd()

        self.config = config
        self.results_folder = results_folder
        self.vector_db_folder = vector_db_folder
        self.knowledge_base = knowledge_base
        self.prompt_message = prompt_message

        pprint.pprint(f"CustomRAG config: {config}")

        self.set_config_options()
        self.create_required_folders()

    def set_config_options(self) -> None:
        """
        Set the configuration options for the CustomRAG instance.
        """
        self.chunk_size = self.config["chunk_size"]
        self.chunk_overlap = self.config["chunk_overlap"]

        self.embeddings_model_name = self.config["embeddings_function"]["model_name"]
        self.embeddings_platform = self.config["embeddings_function"]["platform"]

        self.llm = self.config["llm"]["model_name"]
        self.llm_client = self.config["llm"]["client"]

        self.vector_database = self.config["vector_database"]
        self.vector_database_name = f"{self.chunk_size}_{self.embeddings_model_name}"

        self.filename = f"{self.chunk_size}_{self.embeddings_model_name}_{self.llm}"

    def create_required_folders(self) -> None:
        """
        Create the required folders for results and vector database if they do not exist.
        """
        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)
        if not os.path.exists(self.vector_db_folder):
            os.mkdir(self.vector_db_folder)

    def initialize_embeddings_function(self) -> None:
        """
        Initialize the embeddings function based on the platform specified in the configuration.
        """
        cache_dir = "./model_cache"

        supported_platforms = ["OpenAI", "SentenceTransformers"]

        if self.embeddings_platform not in supported_platforms:
            print("Warning: {self.embeddings_platform} is not supported yet")
            print("Switching to default platform")

            self.embeddings_platform = "OpenAI"

        self.embeddings_function = None

        if self.embeddings_platform == "OpenAI":
            self.embeddings_function = OpenAIEmbeddings(
                model=self.embeddings_model_name
            )
        elif self.embeddings_platform == "SentenceTransformers":
            self.embeddings_function = SentenceTransformer(
                self.embeddings_model_name, cache_folder=cache_dir
            )

    def initialize_llm_client(self) -> None:
        """
        Initialize the LLM client based on the client specified in the configuration.
        """
        supported_clients = ["OpenAI", "Mistral"]

        if self.llm_client not in supported_clients:
            print("Warning: {self.llm_client} is not supported yet")
            print("Switching to default client")

            self.llm_client = "OpenAI"

        self.llm_initialized_client = None

        if self.llm_client == "OpenAI":
            self.llm_initialized_client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        elif self.llm_client == "Mistral":
            self.llm_initialized_client = Mistral(
                api_key=os.environ.get("MISTRAL_API_KEY")
            )

    def split_documents(
        self, knowledge_base: List[LangchainDocument]
    ) -> List[LangchainDocument]:
        """
        Split the documents in the knowledge base into smaller chunks.

        Args:
            knowledge_base (List[LangchainDocument]): The knowledge base documents.

        Returns:
            List[LangchainDocument]: The processed documents.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=int(self.chunk_size / self.chunk_overlap),
            add_start_index=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        docs_processed = []

        for doc in knowledge_base:
            docs_processed += text_splitter.split_documents([doc])

        return docs_processed

    def create_chroma_vector_store(self) -> None:
        """
        Create a Chroma vector store for the processed documents.
        """
        docs_processed = self.split_documents(self.knowledge_base)

        persistent_directory = os.path.join(
            self.vector_db_folder, self.vector_database_name
        )

        if not os.path.exists(persistent_directory):
            print(f"Creating vector store {self.vector_database_name}")

            Chroma.from_documents(
                docs_processed,
                self.embeddings_function,
                persist_directory=persistent_directory,
            )

            print(f"Finished creating vector store {self.vector_database_name}")
        else:
            print(
                f"Vector store {self.vector_database_name} already exists. No need to initialize."
            )

    def create_vector_database(self) -> None:
        """
        Create the vector database based on the configuration.
        """
        supported_vector_databases = ["chromadb"]

        if self.vector_database not in supported_vector_databases:
            print("Warning: {self.vector_database} is not supported yet")
            print("Switching to default vector database")

            self.vector_database = "chromadb"

        if self.vector_database == "chromadb":
            self.create_chroma_vector_store()

    def query_chroma_vector_store(
        self, query: str, n_results: int = 3, score_threshold: float = 0.1
    ) -> List[LangchainDocument]:
        """
        Query the Chroma vector store for relevant documents.

        Args:
            query (str): The query string.
            n_results (int): The number of results to return.
            score_threshold (float): The score threshold for filtering results.

        Returns:
            List[LangchainDocument]: The relevant documents.
        """
        # print("Querying chroma vector store")
        # print(query)

        persistent_directory = os.path.join(
            self.vector_db_folder, self.vector_database_name
        )

        relevant_docs = []

        if os.path.exists(persistent_directory):
            db = Chroma(
                persist_directory=persistent_directory,
                embedding_function=self.embeddings_function,
            )
            retriever = db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": n_results, "score_threshold": score_threshold},
            )
            relevant_docs = retriever.invoke(query)
        else:
            print(f"Vector store {self.vector_db_folder} does not exist.")

        return relevant_docs

    def query_vector_store(
        self, query: str, n_results: int = 3, score_threshold: float = 0.1
    ) -> List[LangchainDocument]:
        """
        Query the vector store for relevant documents.

        Args:
            query (str): The query string.
            n_results (int): The number of results to return.
            score_threshold (float): The score threshold for filtering results.

        Returns:
            List[LangchainDocument]: The relevant documents.
        """
        relevant_docs = []

        if self.vector_database == "chromadb":
            relevant_docs = self.query_chroma_vector_store(
                query, n_results, score_threshold
            )

        return relevant_docs

    def get_llm_single_question_answer(self, query: str) -> Tuple[str, str]:
        """
        Get the answer to a single question using the LLM.

        Args:
            query (str): The query string.

        Returns:
            Tuple[str, str]: The answer and the context.
        """
        relevant_docs = self.query_vector_store(query)

        context = "\n\n".join(
            [f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(relevant_docs)]
        )

        # print(f"** Context: {context}")

        prompt = self.prompt_message % (context, query)

        response = self.llm_initialized_client.chat.completions.create(
            model=self.llm,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
        )

        answer = response.choices[0].message.content

        return answer, context

    def save_llm_results(
        self, questions_ids: List[str], answers: List[str], contexts: List[str]
    ) -> None:
        """
        Save the LLM results to files.

        Args:
            questions_ids (List[str]): The list of question IDs.
            answers (List[str]): The list of answers.
            contexts (List[str]): The list of contexts.
        """
        preds_dict = dict(zip(questions_ids, answers))
        contexts_dict = dict(zip(questions_ids, contexts))

        filepath_pred = os.path.join(self.results_folder, f"pred_{self.filename}")

        with open(filepath_pred, "w") as f:
            json.dump(preds_dict, f, indent=4, sort_keys=True)

        filepath_context = os.path.join(self.results_folder, f"context_{self.filename}")

        with open(filepath_context, "w") as f:
            json.dump(contexts_dict, f, indent=4, sort_keys=True)

    def get_llm_multiple_questions_answers(self, questions_df: pd.DataFrame) -> None:
        """
        Get answers to multiple questions using the LLM.

        Args:
            questions_df (pd.DataFrame): The DataFrame containing questions and their IDs.
        """
        self.initialize_llm_client()
        self.initialize_embeddings_function()
        self.create_vector_database()

        answers = []
        contexts = []
        questions = questions_df["question"]
        questions_ids = questions_df["id"]

        for query in tqdm.tqdm(questions):
            answer, context = self.get_llm_single_question_answer(query)

            answers.append(answer)
            contexts.append(context)

        self.save_llm_results(questions_ids, answers, contexts)


def convert_knowledge_base_to_langchain_docs(
    df: pd.DataFrame,
) -> List[LangchainDocument]:
    """
    Convert the context knowledge base in the DataFrame to Langchain documents.

    Args:
        df (pd.DataFrame): The DataFrame containing context and titles.

    Returns:
        List[LangchainDocument]: The list of Langchain documents.
    """
    langchain_docs = []

    for _, row in df.drop_duplicates(subset="context").iterrows():
        context = row.context
        title = row.title

        document = LangchainDocument(page_content=context, metadata={"title": title})

        langchain_docs.append(document)

    return langchain_docs


"""
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

"""


parameters_dict = {
    "chunk_sizes": [100, 200, 400, 500, 600],
    "embed_options": {
        "text-embedding-3-small": "OpenAI",
        "text-embedding-3-large": "OpenAI",
        "text-embedding-ada-002": "OpenAI",
    },
    "models": {"gpt-3.5-turbo": "OpenAI", "mistral-large-latest": "Mistral"},
}


def fine_tune_rag(
    df: pd.DataFrame,
    knowledge_base_docs: List[LangchainDocument],
    parameters_dict: Dict[str, Any] = parameters_dict,
    results_folder: str = "eval_results",
    vector_db_folder: str = "vector_databases",
) -> None:
    """
    Fine-tune the RAG model with the given parameters.

    Args:
        df (pd.DataFrame): The DataFrame containing questions and their IDs.
        knowledge_base_docs (List[LangchainDocument]): The list of Langchain documents.
        parameters_dict (Dict[str, Any]): The dictionary of parameters for fine-tuning.
        results_folder (str): The folder to save results.
        vector_db_folder (str): The folder to save vector database.
    """
    print("Fine tuning RAG with the following parameters: ")
    pprint.pprint(parameters_dict)

    chunk_sizes = parameters_dict.get("chunk_sizes", [100])
    embed_options = parameters_dict.get(
        "embed_options", {"text-embedding-3-small": "OpenAI"}
    )
    models = parameters_dict.get("models", {"gpt-3.5-turbo": "openai"})

    for embeddings_name, embeddings_platform in embed_options.items():
        for model_name, client in models.items():
            for chunk_size in chunk_sizes:
                print(f"Running {model_name} - {chunk_size} - {embeddings_name}")

                filename = f"{chunk_size}_{embeddings_name}_{model_name}.json"
                filepath = os.path.join(results_folder, f"pred_{filename}")

                if os.path.isfile(filepath):
                    print("Results already exist for these settings: skipping!")
                    continue

                parameters_dict = {
                    "chunk_size": chunk_size,
                    "chunk_overlap": 15,
                    "vector_database": "chromadb",
                    "embeddings_function": {
                        "model_name": embeddings_name,
                        "platform": embeddings_platform,
                    },
                    "llm": {"model_name": model_name, "client": client},
                }

                rag = CustomRAG(
                    knowledge_base=knowledge_base_docs,
                    prompt_message=prompt_message,
                    config=parameters_dict,
                    results_folder=results_folder,
                    vector_db_folder=vector_db_folder,
                )

                rag.get_llm_multiple_questions_answers(df)
