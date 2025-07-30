import os
from src.Recommender.recommender import call_recommender_llm, process_reply
from src.Metrics.get_metrics import calculate_metrics
from src.Rater.rater import call_rater_llm_meta_prompt, call_rater_llm_prompt
from src.TopK_Heap.top_k import TopKHeap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_metric_history(metric_history: dict, filepath="metric_history_opro.txt"):
  with open(filepath, "w", encoding="utf-8") as f:
    for metric, scores in metric_history.items():
      f.write(f"=== {metric.upper()} ===\n")
      for score_type, values in scores.items():
        f.write(f"{score_type}: {values}\n")
      f.write("\n")
  print(f"Saved metric history to {filepath}")

def save_top_k_prompts(top_k_prompts: TopKHeap, filepath="top_k_prompts_opro.txt"):
  top_k_prompts = top_k_prompts.get_topK()

  with open(filepath, "w", encoding="utf-8") as f:
    for i, item in enumerate(top_k_prompts, 1):
      f.write(f"\n\n\n--- Top {i} Prompt ---\n")
      f.write(f"Instruction:\n{item.get('instruction', '')}\n\n")

      metrics = item.get("metrics", {})
      f.write("Metrics:\n")
      for metric_name, metric_values in metrics.items():
        f.write(f"   {metric_name}:\n")
        for k, v in metric_values.items():
          f.write(f"       {k}: {v}\n")
      reco = item.get("recommendation", "")
      f.write(f"\n{reco}\n")

  print(f"Saved top-k prompts to {filepath}")

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

  def ema(series, alpha=0.3):
    smoothed = [series[0]]
    for i in range(1, len(series)):
      smoothed.append(alpha * series[i] + (1 - alpha) * smoothed[-1])
    return smoothed

  metric_types = ["f1", "accuracy", "mean_diff"]
  eval_metrics = list(metric_values.keys())  # ['fluency', 'coherence', ...]

  num_epochs = len(next(iter(metric_values.values()))['f1'])
  epochs = list(range(1, num_epochs + 1))

  for metric_type in metric_types:
    plt.figure()
    for eval_metric in eval_metrics:
      values = metric_values[eval_metric][metric_type]
      smoothed_values = ema(values)
      plt.plot(epochs, smoothed_values, marker='o', label=eval_metric.capitalize())

    plt.title(f"EMA df {metric_type.replace('_', ' ').capitalize()} Over Epochs")
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
  calls_per_minute: int = 120,
  max_workers: int = 10,
  num_examples: int = 100,
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

    evals = call_rater_llm_prompt(
      instruction=instruction, file_path=file_path,
      rater_llm_name=rater_llm_name, max_workers=max_workers,
      calls_per_minute=calls_per_minute, num_examples=num_examples
    )
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

  return {
    "metric_histories": metric_history,
    "top_k_prompts": top_k_prompts,
  }

if __name__ == "__main__":
  rater_llm_name = "meta-llama/llama-3-8b-instruct"
  reco_llm_name = "meta-llama/llama-3-8b-instruct"
  filepath = "Dataset/dataset/df_M11_sampled.parquet"
  opro_results = run_opro(file_path=filepath, top_k=10, num_epochs=30,
                          rater_llm_name=rater_llm_name, reco_llm_name=reco_llm_name, calls_per_minute=60, max_workers=10, num_examples=100)

  # for i, item in enumerate(opro_results["top_k_prompts"], 1):
  #   print(f"\n--- Top {i} Prompt ---")
  #   print(f"Instruction: {item['instruction']}")
  #   print("Metrics:")
  #   for metric, value in item['metrics'].items():
  #     print(f"  {metric}: {value}")
  #   print(f"Recommendation: {item['recommendation']}")
  save_metric_history(opro_results["metric_histories"], "metric_history_opro.txt")
  save_top_k_prompts(opro_results["top_k_prompts"], "top_k_prompts_opro.txt")