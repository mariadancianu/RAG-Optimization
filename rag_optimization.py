import json
import os
import pprint
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import replicate
import tqdm
from dotenv import load_dotenv
from langchain.docstore.document import Document as LangchainDocument
#from langchain_community.llms import Replicate
from mistralai import Mistral
from openai import OpenAI

from vector_store import ChromaVectorStore, QdrantVectorStore

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
            self.file_path = "./default_rag_config.json"
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
        config_dict: Optional[Dict[str, Any]] = None,
        results_folder: Optional[str] = None,
        vector_db_folder: Optional[str] = None,
        save_results=True,
    ):
        """
        Initialize the CustomRAG class with the given parameters.

        Args:
            knowledge_base (List[LangchainDocument]): The knowledge base documents.
            prompt_message (str): The prompt message for the LLM.
            config_dict (Optional[Dict[str, Any]]): Configuration dictionary.
            results_folder (Optional[str]): Folder to save results.
            vector_db_folder (Optional[str]): Folder to save vector database.
        """

        # TODO: split between llm config and vector store config

        config_file = JSONDefaultConfigFile()
        self.default_rag_config = config_file.get()

        if config_dict is None:
            print("Warning: the RAG configurations are missing! Using the default ones")
            config_dict = self.default_rag_config

        if results_folder is None:
            results_folder = os.path.join(os.getcwd(), "eval_results")

        self.results_folder = results_folder
        self.prompt_message = prompt_message
        self.save_results = save_results

        pprint.pprint(f"CustomRAG config: {config_dict}")

        self.set_config_options(config_dict)
        self.create_required_folders()
        self.initialize_llm_client()

        if config_dict["vector_database"] == "chromadb":
            self.vector_store = ChromaVectorStore(
                knowledge_base=knowledge_base,
                config_dict=config_dict,
                vector_db_folder=vector_db_folder,
            )

        elif config_dict["vector_database"] == "qdrant":
            self.vector_store = QdrantVectorStore(
                knowledge_base=knowledge_base,
                config_dict=config_dict
            )
        else:
            raise ValueError("Invalid vector database!")

        self.vector_store.create_vector_store()

    def set_config_options(self, config_dict: Dict[str, Any]) -> None:
        """
        Set the configuration options for the CustomRAG instance.

        Args:
            config_dict (Dict[str, Any]): The configuration dictionary.
        """

        self.llm = config_dict["llm"]["model_name"]
        self.llm_client = config_dict["llm"]["client"]

        chunk_size = config_dict["chunk_size"]
        embeddings_model_name = config_dict["embeddings_function"]["model_name"]
        self.filename = f"{chunk_size}_{embeddings_model_name}_{self.llm.split("/")[-1]}"

    def create_required_folders(self) -> None:
        """
        Create the required folders for results and vector database if they do not exist.
        """
        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)

    def initialize_llm_client(self) -> None:
        """
        Initialize the LLM client based on the client specified in the configuration.
        """
        supported_clients = ["OpenAI", "Mistral", "Replicate"]

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
       # elif self.llm_client == "Replicate":
           # self.llm_initialized_client = Replicate(
            #    model=self.llm,
                #model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
           # )

    def get_llm_single_question_answer(self, query: str) -> Tuple[str, str]:
        """
        Get the answer to a single question using the LLM.

        Args:
            query (str): The query string.

        Returns:
            Tuple[str, str]: The answer and the context.
        """

        context = self.vector_store.query_vector_store(query)

       # print(f"** Context: {context}")

        prompt = self.prompt_message % (context, query)

        if self.llm_client == "OpenAI":
            response = self.llm_initialized_client.chat.completions.create(
                model=self.llm,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
            )

            answer = response.choices[0].message.content

        elif self.llm_client == "Mistral":
            response = self.llm_initialized_client.chat.complete(
                model=self.llm,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
            )

            answer = response.choices[0].message.content

        elif self.llm_client == "Replicate":
            response = replicate.run(
                self.llm,
                input={
                    "prompt": query,
                    "system_prompt": prompt,
                    "max_tokens": 512,
                    "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".format(system_prompt=prompt, prompt="{prompt}"),})

            answer = "".join(s for s in response if s not in ['\n', '\t', '\r', '""'])

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

        filepath_pred = os.path.join(self.results_folder, f"pred_{self.filename}.json")

        with open(filepath_pred, "w") as f:
            json.dump(preds_dict, f, indent=4, sort_keys=True)

        filepath_context = os.path.join(
            self.results_folder, f"context_{self.filename}.json"
        )

        with open(filepath_context, "w") as f:
            json.dump(contexts_dict, f, indent=4, sort_keys=True)

    def get_llm_multiple_questions_answers(self, questions_df: pd.DataFrame) -> None:
        """
        Get answers to multiple questions using the LLM.

        Args:
            questions_df (pd.DataFrame): The DataFrame containing questions and their IDs.
        """

        answers = []
        contexts = []
        questions = questions_df["question"]
        questions_ids = questions_df["id"]

        for query in tqdm.tqdm(questions):
            answer, context = self.get_llm_single_question_answer(query)

            answers.append(answer)
            contexts.append(context)

        if self.save_results:
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



parameters_dict = {
    "chunk_sizes": [100, 200, 400, 500, 600],
    "embed_options": {
        "text-embedding-3-small": "OpenAI",
        "text-embedding-3-large": "OpenAI",
        "text-embedding-ada-002": "OpenAI",
    },
    "models": {"gpt-3.5-turbo": "OpenAI", "mistral-large-latest": "Mistral"},
}


def optimize_rag_parameters(
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
