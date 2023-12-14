from typing import Dict, Union, List
from transformers import Trainer
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util

import evaluate
import torch
import openai
import os

from dotenv import load_dotenv

from ..utils.logger import dist_logger


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class Metrics:
    def __init__(self, metric_name: str, paraphrase_cosine_model: SentenceTransformer = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dist_logger.info(f"Initializing Metrics class on device: {self.device}")

        try:
            self.metric = evaluate.load(metric_name)
            self.paraphrase_cosine = paraphrase_cosine_model or SentenceTransformer(
                "deutsche-telekom/gbert-large-paraphrase-cosine"
            ).to(self.device)
            dist_logger.info("Metrics class initialized successfully")
        except Exception as e:
            dist_logger.error(f"Error initializing Metrics class: {e}")
            raise

    def calculate_cosine_similarity(self, ground_truth: str, answer: str) -> Dict[str, float]:
        dist_logger.info("Calculating cosine similarity")
        try:
            gt_ada = self._get_openai_embedding(ground_truth)
            answer_ada = self._get_openai_embedding(answer)

            gt_paraphrase_cos = self.paraphrase_cosine.encode(ground_truth, device=self.device)
            answer_paraphrase_cos = self.paraphrase_cosine.encode(answer, device=self.device)

            similarity_ada = util.cos_sim(gt_ada, answer_ada)
            similarity_paraphrase_cos = util.cos_sim(gt_paraphrase_cos, answer_paraphrase_cos)

            dist_logger.info("Cosine similarity calculated successfully")
            return {
                "paraphrase_cosine": similarity_paraphrase_cos.mean().item(),
                "ada": similarity_ada.mean().item(),
            }
        except Exception as e:
            dist_logger.error(f"Error calculating cosine similarity: {e}")
            raise

    def _get_openai_embedding(self, text: str) -> torch.Tensor:
        embedding = openai.Embedding.create(
            input=text, model="text-embedding-ada-002"
        )["data"][0]["embedding"]
        return torch.tensor(embedding).to(self.device)

    def compute_metrics(self, trainer: Trainer, eval_dataset: Union[Dataset, List[dict]]) -> Dict[str, float]:
        dist_logger.info("Computing metrics")
        try:
            if trainer is None or eval_dataset is None:
                raise ValueError("Trainer and evaluation dataset are required.")

            eval_output = trainer.predict(eval_dataset)
            predictions = eval_output.predictions.argmax(-1)
            label_ids = eval_output.label_ids
            results = self.metric.compute(predictions=predictions, references=label_ids)

            if self.cosine_similarity:
                cosine_results = self._calculate_dataset_cosine_similarity(eval_dataset)
                results.update(cosine_results)

            dist_logger.info("Metrics computed successfully")
            return results
        except Exception as e:
            dist_logger.error(f"Error in compute_metrics: {e}")
            raise

    def _calculate_dataset_cosine_similarity(self, eval_dataset: Union[Dataset, List[dict]]) -> Dict[str, float]:
        cosine_results = {}

        if self.dataset_key == 'alpaca':
            cosine_results['alpaca_cosine_similarity'] = self._compute_cosine_for_dataset(eval_dataset, lambda x: x['instruction'] + x['output'], lambda x: x['output'])
        elif self.dataset_key == 'squad':
            cosine_results['squad_cosine_similarity'] = self._compute_cosine_for_dataset(eval_dataset, lambda x: x['question'], lambda x: x['answers'][0]['text'], filter_func=lambda x: not x['is_impossible'])
        elif self.dataset_key == 'context_qa':
            cosine_results['context_qa_cosine_similarity'] = self._compute_cosine_for_dataset(eval_dataset, lambda x: x['article'] + " " + x['question'], lambda x: x['answer'])
        elif self.dataset_key == 'custom_alapca':
            cosine_results['custom_format_cosine_similarity'] = self._compute_cosine_for_dataset(
            eval_dataset, 
            lambda x: x['instruction'], 
            lambda x: x['output']
        )
            
        return cosine_results

    def _compute_cosine_for_dataset(self, dataset, text_extractor1, text_extractor2, filter_func=None):
        filtered_dataset = filter(filter_func, dataset) if filter_func else dataset
        cosine_scores = [
            self.calculate_cosine_similarity(text_extractor1(example), text_extractor2(example))['paraphrase_cosine']
            for example in filtered_dataset
        ]
        return sum(cosine_scores) / len(cosine_scores) if cosine_scores else 0.0