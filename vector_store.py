import os
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from fastembed import SparseTextEmbedding
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    Fusion,
    FusionQuery,
    PointStruct,
    Prefetch,
    SearchParams,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    UpdateStatus,
    VectorParams,
)
from sentence_transformers import SentenceTransformer


class VectorStore(ABC):
    @abstractmethod
    def set_config_options(self):
        pass

    @abstractmethod
    def create_vector_store(self) -> None:
        pass

    @abstractmethod
    def query_vector_store(self):
        pass


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        knowledge_base: List[LangchainDocument],
        config_dict: Dict[str, Any],
    ):
        """Initialize the QdrantVectorStore class with the given parameters.

        Args:
            knowledge_base (List[LangchainDocument]): The knowledge base documents.
            config_dict (Dict[str, Any]): The configuration dictionary.
        """

        # TODO: add reranking
        # TODO: experiment with filtering methods
        # TODO: add function to update the vector db with additional documents

        self.knowledge_base = knowledge_base
        self.set_config_options(config_dict)

    def set_config_options(self, config_dict: Dict[str, Any]) -> None:
        """
        Set the configuration options for the Vector Store instance.

        Args:
            config_dict (Dict[str, Any]): The configuration dictionary.
        """

        self.client = QdrantClient(
            url="localhost",
            port=6333,
        )

        self.chunk_size = config_dict["chunk_size"]
        self.chunk_overlap = config_dict["chunk_overlap"]

        self.embeddings_model_name = config_dict["embeddings_function"]["model_name"]
        self.embeddings_platform = config_dict["embeddings_function"]["platform"]

        self.vector_database = config_dict["vector_database"]
        self.vector_database_name = f"{self.chunk_size}_{self.embeddings_model_name}"

        self.sparse_text_model = config_dict["sparse_text_model"]

        if self.embeddings_platform == "OpenAI":
            self.embeddings_client = OpenAI()

        test_vector = (
            self.embeddings_client.embeddings.create(
                input=["test embeddings"], model=self.embeddings_model_name
            )
            .data[0]
            .embedding
        )

        self.vector_size = len(test_vector)
        self.vector_distance = Distance.COSINE
        self.sparse_model = SparseTextEmbedding(model_name=self.sparse_text_model)

    def create_vector_store(self):
        """
        Create the vector database based on the configuration.
        """

        try:
            collection_info = self.client.get_collection(
                collection_name=self.vector_database_name
            )
            print("Collection exists! Skipping creation!")
        except Exception as e:
            print("Collection does not exist, creating collection now")

            """
            self.client.recreate_collection(
                collection_name=self.vector_database_name,
                # dense vector index
                vectors_config=VectorParams(
                    size=self.vector_size, distance=self.vector_distance
                ),
            )"""

            # TODO: add support for hybrid search
            self.client.recreate_collection(
                collection_name=self.vector_database_name,
                # dense vector index
                vectors_config={
                    "text-dense": VectorParams(
                        size=self.vector_size, distance=self.vector_distance
                    )
                },
                # sparse vector index
                sparse_vectors_config={
                    "text-sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False,
                        )
                    )
                },
            )

            collection_info = self.client.get_collection(
                collection_name=self.vector_database_name
            )
            self.upsert_data()

        print(collection_info)

    def upsert_data(self):
        """
        Saves vectors in the vector database.
        """

        print("Adding data in the database")

        points = self.compute_vectors()

        operation_info = self.client.upsert(
            collection_name=self.vector_database_name, wait=True, points=points
        )

        if operation_info.status == UpdateStatus.COMPLETED:
            print("Data inserted successfully!")
        else:
            print("Failed to insert data")

    def compute_vectors(self):
        """
        Computes the vectors.
        """

        points = []
        docs = []
        metadata = []

        for doc in self.knowledge_base:
            docs.append(doc.page_content)
            metadata.append(doc.metadata)

        for doc in docs:
            text_vector_dense = (
                self.embeddings_client.embeddings.create(
                    input=[doc], model=self.embeddings_model_name
                )
                .data[0]
                .embedding
            )
            text_id = str(uuid.uuid4())
            context = {"context": doc}

            text_vector_sparse = list(self.sparse_model.embed(doc))

            # point = PointStruct(id=text_id, vector=text_vector_dense, payload=context)

            # TODO: add support for dense and sparse vectors

            point = PointStruct(
                id=text_id,
                vector={
                    "text-sparse": SparseVector(
                        indices=text_vector_sparse[0].indices.tolist(),
                        values=text_vector_sparse[0].values.tolist(),
                    ),
                    "text-dense": text_vector_dense,
                },
                payload=context,
            )

            points.append(point)

        return points

    def delete_vector_store(self) -> None:
        """
        Deletes the collection.
        """
        print("Deleting the collection")

        collections = self.client.get_collections()

        print("Collections before deletion: ", collections)

        self.client.delete_collection(collection_name=self.vector_database_name)

        collections = self.client.get_collections()

        print("Collections after deletion: ", collections)

    def get_query_vector(self, query: str) -> Dict:
        """
        Get the sparse and dense vectors for the query.

        Args:
            query (str): The query string.

        Returns:
            query_vector_store: dictionary with both dense and sparse
            vectors for the query.
        """

        dense_vector = (
            self.embeddings_client.embeddings.create(
                input=[query], model=self.embeddings_model_name
            )
            .data[0]
            .embedding
        )

        sparse_vector = list(self.sparse_model.embed(query))
        sparse_indices = sparse_vector[0].indices.tolist()
        sparse_scores = sparse_vector[0].values.tolist()

        query_vector_store = {
            "sparse_vector": {"indices": sparse_indices, "values": sparse_scores},
            "dense_vector": dense_vector,
        }

        return query_vector_store

    def query_vector_store(
        self,
        query: str,
        n_results: int = 3,
        join_context: bool = True,
    ) -> str:
        """
        Query the vector store for relevant documents.

        Args:
            query (str): The query string.
            n_results (int): The number of results to return.

        Returns:
            context: retrieved context.
        """

        query_vector = self.get_query_vector(query)

        # search_result = self.client.search(
        #    collection_name=self.vector_database_name,
        #   query_vector=input_vector,
        #  limit=n_results,
        # )

        prefetch_n_sparse = n_results * 5
        prefetch_n_dense = n_results * 5
        filters = None

        prefetch = [
            Prefetch(
                query=SparseVector(**query_vector.get("sparse_vector")),
                using="text-sparse",
                limit=prefetch_n_sparse,
                filter=filters,
            ),
            Prefetch(
                query=query_vector.get("dense_vector"),
                using="text-dense",
                limit=prefetch_n_dense,
                filter=filters,
            ),
        ]

        # TODO: deep-dive into this part
        search_result = self.client.query_points(
            collection_name=self.vector_database_name,
            prefetch=prefetch,
            with_payload=True,
            limit=n_results,
            query=FusionQuery(fusion=Fusion.RRF),
            # Qdrant has a built-in exact search mode, which can be used
            # to measure the quality of the search results.
            # not suitable for production use with high load
            search_params=SearchParams(exact=True),  # Turns on the exact search mode
        )

        if join_context:
            context = "\n\n".join(
                item.payload["context"] for item in search_result.points
            )
        else:
            context = [item.payload["context"] for item in search_result.points]

        return context


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        knowledge_base: List[LangchainDocument],
        config_dict: Dict[str, Any],
        vector_db_folder: Optional[str] = None,
    ):
        """Initialize the ChromaVectorStore class with the given parameters.

        Args:
            knowledge_base (List[LangchainDocument]): The knowledge base documents.
            config_dict (Dict[str, Any]): The configuration dictionary.
            vector_db_folder (Optional[str]): Folder to save vector database.
        """

        if vector_db_folder is None:
            vector_db_folder = os.path.join(os.getcwd(), "vector_databases")

        self.vector_db_folder = vector_db_folder
        self.knowledge_base = knowledge_base
        self.set_config_options(config_dict)
        self.create_required_folders()
        self.initialize_embeddings_function()

    def set_config_options(self, config_dict: Dict[str, Any]) -> None:
        """
        Set the configuration options for the Vector Store instance.

        Args:
            config_dict (Dict[str, Any]): The configuration dictionary.
        """

        self.chunk_size = config_dict["chunk_size"]
        self.chunk_overlap = config_dict["chunk_overlap"]

        self.embeddings_model_name = config_dict["embeddings_function"]["model_name"]
        self.embeddings_platform = config_dict["embeddings_function"]["platform"]

        self.vector_database = config_dict["vector_database"]
        self.vector_database_name = f"{self.chunk_size}_{self.embeddings_model_name}"

    def create_required_folders(self) -> None:
        """
        Create the required folders for results and vector database if they do not exist.
        """
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

    def create_chroma_vector_store_new(self) -> None:
        """
        Create a Chroma vector store for the processed documents.
        """
        docs_processed = self.split_documents(self.knowledge_base)

        docs = [doc.page_content for doc in docs_processed]

        document_embeddings = self.embeddings_function.encode(docs)

        persistent_directory = os.path.join(
            self.vector_db_folder, self.vector_database_name
        )

        chroma_client = chromadb.PersistentClient(path=persistent_directory)
        collection = chroma_client.get_or_create_collection(
            name=self.vector_database_name
        )

        ids = [f"id{i}" for i in list(range(len(docs)))]

        # Ensure the number of IDs matches the number of documents
        # note: use upsert instead of add to avoid adding existing documents
        collection.upsert(ids=ids, documents=docs, embeddings=document_embeddings)

    def create_vector_store(self) -> None:
        """
        Create the vector database based on the configuration.
        """
        supported_vector_databases = ["chromadb"]

        if self.vector_database not in supported_vector_databases:
            print("Warning: {self.vector_database} is not supported yet")
            print("Switching to default vector database")

            self.vector_database = "chromadb"

        if self.vector_database == "chromadb":
            if self.embeddings_platform == "SentenceTransformers":
                self.create_chroma_vector_store_new()
            else:
                self.create_chroma_vector_store()

    def query_chroma_vector_store_new(
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

        persistent_directory = os.path.join(
            self.vector_db_folder, self.vector_database_name
        )

        relevant_docs = []

        if os.path.exists(persistent_directory):
            query_embeddings = self.embeddings_function.encode(query)

            chroma_client = chromadb.PersistentClient(path=persistent_directory)
            collection = chroma_client.get_or_create_collection(
                name=self.vector_database_name
            )

            relevant_docs = collection.query(
                query_embeddings=query_embeddings, n_results=n_results
            )
        else:
            print(f"Vector store {self.vector_database_name} does not exist.")

        return relevant_docs

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
        self,
        query: str,
        n_results: int = 3,
        score_threshold: float = 0.1,
        join_context: bool = True,
    ) -> str:
        """
        Query the vector store for relevant documents.

        Args:
            query (str): The query string.
            n_results (int): The number of results to return.
            score_threshold (float): The score threshold for filtering results.

        Returns:
            context: retrieved context.
        """

        context = ""

        if self.vector_database == "chromadb":
            if self.embeddings_platform == "SentenceTransformers":
                relevant_docs = self.query_chroma_vector_store_new(
                    query, n_results, score_threshold
                )
                relevant_docs = relevant_docs["documents"][0]

                if join_context:
                    context = "\n\n".join(doc for doc in relevant_docs)
                else:
                    context = relevant_docs.copy()
            else:
                relevant_docs = self.query_chroma_vector_store(
                    query, n_results, score_threshold
                )
                if join_context:
                    context = "\n\n".join(
                        [
                            f"Source {i+1}: {doc.page_content}"
                            for i, doc in enumerate(relevant_docs)
                        ]
                    )
                else:
                    context = [doc.page_content for doc in relevant_docs]

        return context
