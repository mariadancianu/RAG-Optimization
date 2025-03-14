import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class RetrievalEvaluator:
    def __init__(
        self,
        df_ground_truth: pd.DataFrame,
        vector_store: Optional[object] = None,
        path_to_results: Optional[str] = None,
    ) -> None:
        """
        Initialize the RetrievalEvaluator with ground truth data and optional vector store.

        Args:
            df_ground_truth (pd.DataFrame): DataFrame containing ground truth data.
            vector_store (Optional[object]): An optional vector store for querying.
            path_to_results (Optional[str]): Path to save evaluation results.
        """

        # TODO: add option to evaluate directly with contexts, instead of fetching from the db

        self.df_ground_truth = df_ground_truth
        self.vector_store = vector_store

        if path_to_results is None:
            path_to_results = os.path.join(os.getcwd(), "eval_results")

        if not os.path.exists(path_to_results):
            os.makedirs(path_to_results)

        self.path_to_results = path_to_results

    def calculate_scores(self) -> Dict[str, float]:
        """
        Calculate evaluation scores including hit rate, mean reciprocal rank, precision, and recall.

        Returns:
            Dict[str, float]: A dictionary containing average scores and total queries.
        """
        results_dict = {
            "avg_hit_rate": None,
            "avg_mean_reciprocal_rank": None,
            "avg_precision": None,
            "avg_recall": None,
        }

        top_k = 3
        hit_count = 0
        recall_scores, precision_scores, mrr_scores = [], [], []
        total_queries = 0

        for _, row in tqdm(
            self.df_ground_truth[self.df_ground_truth.is_impossible == False].iterrows()
        ):
            total_queries += 1
            question = row.question
            ground_truth = row[
                ["answer_0", "answer_1", "answer_2", "answer_3"]
            ].values.tolist()
            ground_truth = [x for x in ground_truth if pd.notna(x)]

            context = self.vector_store.query_vector_store(question, join_context=False)

            relevant_chunks = set()

            for answer in ground_truth:
                for i, chunk in enumerate(context):
                    if answer in chunk:
                        relevant_chunks.add(i)

            retrieved_relevant_chunks = sum(
                1 for i, chunk in enumerate(context) if i in relevant_chunks
            )

            # Compute Recall
            recall = (
                retrieved_relevant_chunks / len(relevant_chunks)
                if relevant_chunks
                else 0
            )
            recall_scores.append(recall)

            # Compute Precision
            precision = retrieved_relevant_chunks / top_k
            precision_scores.append(precision)

            # Compute Hit Rate
            if retrieved_relevant_chunks > 0:
                hit_count += 1

            # Compute MRR
            rank = next(
                (
                    idx + 1
                    for idx, i in enumerate(range(len(context)))
                    if i in relevant_chunks
                ),
                0,
            )
            mrr_scores.append(1 / rank if rank > 0 else 0)

        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        avg_precision = (
            sum(precision_scores) / len(precision_scores) if precision_scores else 0
        )
        hit_rate = hit_count / total_queries
        mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0

        results_dict["avg_hit_rate"] = hit_rate
        results_dict["avg_mean_reciprocal_rank"] = mrr
        results_dict["avg_precision"] = avg_precision
        results_dict["avg_recall"] = avg_recall
        results_dict["total_queries"] = total_queries

        return results_dict

    def save_scores(self, results_dict: Dict[str, float]) -> None:
        """
        Save the evaluation scores to a JSON file.

        Args:
            results_dict (Dict[str, float]): Dictionary containing evaluation results to save.
        """
        database_type = self.vector_store.vector_database
        chunk_size = self.vector_store.chunk_size
        chunk_overlap = self.vector_store.chunk_overlap
        embeddings_model_name = self.vector_store.embeddings_model_name
        results_filename = os.path.join(
            self.path_to_results,
            f"{database_type}_{chunk_size}_{chunk_overlap}_{embeddings_model_name}_evaluation_results.json",
        )

        with open(results_filename, "w") as f:
            json.dump(results_dict, f, indent=4)

    def evaluate(self) -> None:
        """
        Evaluate the retrieval performance and save the results.
        """
        results_dict = self.calculate_scores()
        self.save_scores(results_dict)
