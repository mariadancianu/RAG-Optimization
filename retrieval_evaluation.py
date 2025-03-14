import json
import os

import numpy as np
import pandas as pd


class RetrievalEvaluator:
    def __init__(self, y_pred=None, y_true=None, path_to_results=None):
        if y_pred is None or y_true is None:
            raise ValueError("y_pred and y_true must be provided.")

        self.y_pred = y_pred
        self.y_true = y_true

        if path_to_results is None:
            path_to_results = os.path.join(os.getcwd(), "eval_results")

        if not os.path.exists(path_to_results):
            os.makedirs(path_to_results)

        self.path_to_results = path_to_results
        self.results_dict = {"mean_reciprocal_rank": None, "hit_rate": None}

    def calculate_mean_reciprocal_rank(self):
        # TODO
        return None

    def calculate_hit_rate(self):
        # TODO
        return None

    def calculate_scores(self):
        mean_reciprocal_rank = self.calculate_mean_reciprocal_rank()
        hit_rate = self.calculate_hit_rate()

        self.results_dict["mean_reciprocal_rank"] = mean_reciprocal_rank
        self.results_dict["hit_rate"] = hit_rate

    def save_scores(self):
        results_filename = os.path.join(self.path_to_results, "evaluation_results.json")

        with open(results_filename, "w") as f:
            json.dump(self.results_dict, f)

    def evaluate(self):
        self.calculate_scores()
        self.save_scores()
