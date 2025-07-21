from src.Dataset.random_subsample import create_sample_points
from src.Optim.optimizer import call_optimizer_llm
from sklearn.metrics import classification_report
from typing import Dict

from src.TopK_Heap.top_k import TopKHeap


def calculate_metrics(sample_points: list[Dict[str, str]], optim_llm_response: dict) -> dict:
  """
  Calculates metrics based on sample_points and optim_llm_response
  :param sample_points: Original Sample Points with Ground Truth scores.
  :param optim_llm_response: Predicted LLM response with predicted scores
  :return: A dict with metrics

  { "label" : "classification_report" }
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

    for metric in metrics.keys():
      ground_key = f"ground_{metric}"
      predicted_key = f"predicted_{metric}"

      y_true = int(sample[ground_key])
      y_pred = int(predicted_scores[predicted_key])

      metrics[metric]["y_true"].append(y_true)
      metrics[metric]["y_pred"].append(y_pred)

  result = {}
  for metric, data in metrics.items():
    report = classification_report(
      y_true=data["y_true"],
      y_pred=data["y_pred"],
      labels=[1, 2, 3, 4, 5],
      zero_division=0,
      output_dict=True
    )
    result[metric] = report

  return result

if __name__ == "__main__":
  optim_llm_name = "meta-llama/llama-3.2-3b-instruct:free"
  sample_points = create_sample_points("../Dataset/dataset/df_model_M11.csv")
  print("Created sample_points:")
  top_k_prompts = TopKHeap(3)
  print("Created top_k_prompts:")
  print("Calling Optimizer!")
  optim_llm_response = call_optimizer_llm(sample_points, top_k_prompts, optim_llm_name)

  print(optim_llm_response)
  print('\n\n')
  print('=' * 100)
  print("Metrics ->")
  metrics = calculate_metrics(sample_points, optim_llm_response)
  # print(metrics)

  for metric, report in metrics.items():
    print(metric)
    print(report,)
    print('=' * 100)