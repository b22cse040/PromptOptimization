import pandas as pd
import numpy as np
import heapq
import torch
import torch.nn as nn
from src.Rater.rater import call_rater_llm_meta_prompt, call_rater_llm_prompt
from sklearn.metrics import accuracy_score, f1_score, log_loss


from src.TopK_Heap.top_k import TopKHeap


def calculate_metrics(rater_response: list[dict], file_path: str = "../Dataset/dataset/df_M11_sampled.parquet") -> dict:
  """
    Calculates accuracy and F1-score per metric (fluency, coherence, consistency, relevance).

    :param file_path: File containing sample points with ground-truth scores.
    :param rater_response: RATER response containing predicted scores
    :return: Dict in format:
             {
               "fluency": {"accuracy": ..., "f1": ..., "CE_fluency" : ...},
               "coherence": {"accuracy": ..., "f1": ..., "CE_coherence" : ...},
               ...
               "CE_Total" : Float
             }
  """
  df = pd.read_parquet(file_path)

  metrics = {
    "fluency": {"y_true": [], "y_pred": []},
    "coherence": {"y_true": [], "y_pred": []},
    "consistency": {"y_true": [], "y_pred": []},
    "relevance": {"y_true": [], "y_pred": []},
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
  ce_values = []
  ce_loss = nn.CrossEntropyLoss()
  for metric, data in metrics.items():
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    if not y_true or not y_pred:
      result[metric] = {"accuracy": 0, "f1": 0, f"CE_{metric}" : 0} # "mean_diff": 0}
      continue

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    try:
      y_true_tensor = torch.tensor(y_true, dtype=torch.long)
      y_pred_tensor = torch.tensor(y_pred, dtype=torch.long)

      num_classes = max(max(y_true), max(y_pred)) + 1
      logits = torch.zeros((len(y_pred), num_classes))
      logits[torch.arange(len(y_pred)), y_pred_tensor] = 1.0  # confident prediction

      ce = ce_loss(logits, y_true_tensor).item()


    except Exception as e:
      print(f"Exception while calculating {metric} : {repr(e)}")
      ce = 0

    ce_values.append(ce)

    result[metric] = {
      "accuracy": round(acc, 3),
      "f1": round(f1, 3),
      f"CE_{metric}": round(ce, 3),
      # "mean_diff": mean_diff
    }

    # result["CE_Total"] = round(np.mean(ce_values), 3) if ce_values else 0

  return result

def find_most_imformative_points(rater_response: list[dict], file_path: str = "../Dataset/dataset/df_M11_sampled.parquet", top_k: int = 5) -> list[dict]:
  '''
  Finding the points with the most LCE / MSE, and based on those errors, to
  determine whether the LLM is detecting a harshly or leniently with the help of
  mean difference, i.e. {predicted_score - ground_score}, a negative score indicating
  model is harsh, while a positive score indicating model is lenient.

  Returns: a list of dict where each dict has the following keys:
    {
      point_idx : int, # indicates the df.iloc[i] of the samples
      LCE: float, # indicates the LCE of the sample,
      mean_diff: int, # indicates the mean_diff of the sample, indicating whether the sample is harsher.
    }
  '''
  df = pd.read_parquet(file_path)
  ce_loss = nn.CrossEntropyLoss()

  heap = []

  for i, entry in enumerate(rater_response):
    if not entry or "score" not in entry or not entry["score"]:
      continue
    if i >= len(df):
      continue

    score = entry["score"]
    sample = df.iloc[i]

    total_ce = 0.0
    diffs = []
    valid_metrics = 0

    for metric in ["fluency", "consistency", "relevance", "coherence",]:
      try:
        ground_score = int(sample[f"{metric}"])
        predicted_score = int(score[f"predicted_{metric}"])
      except (KeyError, ValueError, TypeError): continue

      if ground_score == 0 or predicted_score == 0: continue

      num_classes = max(ground_score, predicted_score) + 1
      logits = torch.zeros((1, num_classes))
      logits[0, predicted_score] = 1
      ce = ce_loss(logits, torch.tensor([ground_score])).item()
      total_ce += ce

      diffs.append(predicted_score - ground_score)
      valid_metrics += 1

    if valid_metrics == 0: continue

    mean_diff = sum(diffs) / valid_metrics

    heapq.heappush(heap, (total_ce, i, {
      "point_idx": i,
      "LCE": round(total_ce, 3) / 4,
      "mean_diff": round(mean_diff, 3),
    }))

    if len(heap) > top_k:
      heapq.heappop(heap)

  top_points = [item for _, _, item in heap]
  top_points.sort(key=lambda x: x["LCE"], reverse=True)
  return top_points

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

  top_points = find_most_imformative_points(rater_response)
  print(top_points)