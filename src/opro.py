import os
from src.Recommender.recommender import call_recommender_llm, process_reply
from src.Metrics.get_metrics import calculate_metrics
from src.Rater.rater import call_rater_llm_meta_prompt, call_rater_llm_prompt
from src.TopK_Heap.top_k import TopKHeap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_metric_over_epochs(metric_values: dict, save_dir="Plots"):
  """
  Plots f1, accuracy, and mean_diff over epochs for all evaluation metrics
  (fluency, coherence, consistency, relevance) in single plots.

  :param metric_values: Nested dictionary:
    {
      "fluency": {"f1": [...], "accuracy": [...], "mean_diff": [...]},
      ...
    }
  :param save_dir: Directory where the plots will be saved.
  """
  os.makedirs(save_dir, exist_ok=True)
  metric_types = ["f1", "accuracy", "mean_diff"]
  eval_metrics = list(metric_values.keys())  # ['fluency', 'coherence', ...]

  num_epochs = len(next(iter(metric_values.values()))['f1'])
  epochs = list(range(1, num_epochs + 1))

  for metric_type in metric_types:
    plt.figure()
    for eval_metric in eval_metrics:
      values = metric_values[eval_metric][metric_type]
      plt.plot(epochs, values, marker='o', label=eval_metric.capitalize())

    plt.title(f"{metric_type.replace('_', ' ').capitalize()} Scores Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric_type.replace('_', ' ').capitalize())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{metric_type}_scores.png")
    plt.savefig(save_path)
    print(f"Saved plot at '{save_path}'")
    plt.close()


def run_opro(
  file_path: str = "Dataset/dataset/df_M11_sampled.parquet",
  top_k: int = 10,
  num_epochs: int = 30,
  rater_llm_name: str = "meta-llama/llama-3-8b-instruct",
  reco_llm_name: str = "meta-llama/llama-3-8b-instruct",
) -> dict:

  ## Top-K prompts
  top_k_prompts = TopKHeap(top_k)
  print(f"Step 1: Created a heap to store top-{top_k} prompts (Successful)")

  metric_names = ["fluency", "coherence", "consistency", "relevance"]
  metric_history = {
    metric : {"f1" : [], "accuracy" : [], "mean_diff" : []} for metric in metric_names
  }

  for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
    instruction = call_rater_llm_meta_prompt(top_k_prompts=top_k_prompts,
                                             rater_llm_name=rater_llm_name)
    print("Generated instruction")

    evals = call_rater_llm_prompt(instruction=instruction, file_path=file_path, rater_llm_name=rater_llm_name)
    print("Generated evals")

    metrics = calculate_metrics(evals, file_path=file_path)
    print("Calculated metrics")

    for metric in metric_names:
      metric_history[metric]["f1"].append(metrics[metric]["f1"])
      metric_history[metric]["accuracy"].append(metrics[metric]["accuracy"])
      metric_history[metric]["mean_diff"].append(metrics[metric]["mean_diff"])

    recommendation = call_recommender_llm(instruction, metrics, reco_llm_name=reco_llm_name)
    print("Generated recommendation")

    processed_reply = process_reply(instruction=instruction, recommendation=recommendation, heap=top_k_prompts, metrics=metrics)
    print("Processed Reply")

  for metric in metric_names:
    plot_metric_over_epochs(metric_values=metric_history)

  return metric_history

if __name__ == "__main__":
  rater_llm_name = "meta-llama/llama-3-8b-instruct"
  reco_llm_name = "meta-llama/llama-3-8b-instruct"
  filepath = "Dataset/dataset/df_M11_sampled.parquet"
  opro_results = run_opro(file_path=filepath, top_k=10, num_epochs=30
                          , rater_llm_name=rater_llm_name, reco_llm_name=reco_llm_name)