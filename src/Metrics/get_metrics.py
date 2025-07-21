from src.Dataset.random_subsample import create_sample_points
from src.Optim.optimizer import call_optimizer_llm
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict

from src.TopK_Heap.top_k import TopKHeap


def calculate_metrics(sample_points: list[Dict[str, str]], optim_llm_response: dict) -> dict:
  """
    Calculates accuracy and F1-score per metric (fluency, coherence, consistency, relevance).

    :param sample_points: Original sample points with ground-truth scores.
    :param optim_llm_response: Model predictions with scores per sample.
    :return: Dict in format:
             {
               "fluency": {"accuracy": ..., "f1": ...},
               "coherence": {"accuracy": ..., "f1": ...},
               ...
             }
  """

  metrics = {
    "fluency": {"y_true": [], "y_pred": []},
    "coherence": {"y_true": [], "y_pred": []},
    "consistency": {"y_true": [], "y_pred": []},
    "relevance": {"y_true": [], "y_pred": []}
  }

  for i, sample in enumerate(sample_points, start=1):
    response_key = str(i)
    if response_key not in optim_llm_response["sample_points"]:
      continue

    predicted_scores = optim_llm_response["sample_points"][response_key]["score"]

    for metric in metrics:
      ground_score = int(sample[f"ground_{metric}"] if sample[f"ground_{metric}"] else 0.0)
      predicted_score = int(predicted_scores[f"predicted_{metric}"] if predicted_scores else 0.0)

      if ground_score == 0 or predicted_score == 0: continue

      metrics[metric]["y_true"].append(ground_score)
      metrics[metric]["y_pred"].append(predicted_score)

  # Calculating eval metrics: F1 and accuracy
  result = {}
  for metric, data in metrics.items():
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    if not y_true or not y_pred:
      result[metric] = {"accuracy": 0, "f1": 0}
      continue

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    result[metric] = {
      "accuracy": round(acc, 4),
      "f1": round(f1, 4)
    }

  return result

if __name__ == "__main__":
  optim_llm_name = "deepseek/deepseek-r1-0528-qwen3-8b:free"
  sample_points = create_sample_points("../Dataset/dataset/df_model_M11.csv")
  # print("Created sample_points:")
  top_k_prompts = TopKHeap(3)
  # print("Created top_k_prompts:")
  # print("Calling Optimizer!")
  optim_llm_response = call_optimizer_llm(sample_points, top_k_prompts, optim_llm_name)

  # print(optim_llm_response)
  # print('\n\n')
  # print('=' * 100)
  # print("Metrics ->")
  metrics = calculate_metrics(sample_points, optim_llm_response)
  # print(metrics)

  print(metrics)