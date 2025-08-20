import pandas as pd
import numpy as np
from src.Rater.rater import call_rater_llm_meta_prompt, call_rater_llm_prompt
from sklearn.metrics import accuracy_score, f1_score

from src.TopK_Heap.top_k import TopKHeap


def calculate_metrics(rater_response: list[dict], file_path: str = "../Dataset/dataset/df_M11_sampled.parquet") -> dict:
  """
    Calculates accuracy and F1-score per metric (fluency, coherence, consistency, relevance).

    :param file_path: File containing sample points with ground-truth scores.
    :param rater_response: RATER response containing predicted scores
    :return: Dict in format:
             {
               "fluency": {"accuracy": ..., "f1": ...},
               "coherence": {"accuracy": ..., "f1": ...},
               ...
             }
  """
  df = pd.read_parquet(file_path)

  metrics = {
    "fluency": {"y_true": [], "y_pred": [], "diffs" : []},
    "coherence": {"y_true": [], "y_pred": [], "diffs" : []},
    "consistency": {"y_true": [], "y_pred": [], "diffs" : []},
    "relevance": {"y_true": [], "y_pred": [], "diffs" : []},
  }

  for i, entry in enumerate(rater_response):
    if not entry or "score" not in entry or not entry["score"]:
      continue

    if i >= len(df): continue

    score = entry["score"]
    sample = df.iloc[i]

    for metric in ["fluency", "consistency", "relevance", "coherence",]: #  "fluency", "consistency", "relevance", "coherence",
      try:
        ground_score = int(sample[f"{metric}"])
        predicted_score = int(score[f"predicted_{metric}"])
      except (KeyError, ValueError, TypeError): continue

      if ground_score == 0 or predicted_score == 0:
        continue

      metrics[metric]["y_true"].append(ground_score)
      metrics[metric]["y_pred"].append(predicted_score)
      # metrics[metric]["diffs"].append(predicted_score - ground_score)

  result = {}
  for metric, data in metrics.items():
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    diffs = data["diffs"]

    if not y_true or not y_pred:
      result[metric] = {"accuracy": 0, "f1": 0,} # "mean_diff": 0}
      continue

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    # mean_diff = round(np.mean(diffs), 3) if diffs else 0

    result[metric] = {
      "accuracy": round(acc, 3),
      "f1": round(f1, 3),
      # "mean_diff": mean_diff
    }

  return result

if __name__ == "__main__":
  rater_llm_name = "meta-llama/llama-3-8b-instruct"
  file_path = "../Dataset/dataset/df_M11_sampled.csv"


  top_k_prompts = TopKHeap(3)
  # print("Created top_k_prompts:")
  # print("Calling Optimizer!")
  meta_prompt = call_rater_llm_meta_prompt(top_k_prompts, rater_llm_name)

  print(meta_prompt)
  print('=' * 100)

  rater_response = call_rater_llm_prompt(meta_prompt, rater_llm_name=rater_llm_name, num_examples=20, max_workers=20)
  print("Metrics ->")

  metrics = calculate_metrics(rater_response)
  print(metrics)