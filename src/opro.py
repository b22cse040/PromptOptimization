import os
from src.Recommender.recommender import call_recommender_llm, process_reply
from src.Metrics.get_metrics import calculate_metrics, find_most_imformative_points
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
        if isinstance(metric_values, dict):
          for k, v in metric_values.items():
            f.write(f"       {k}: {v}\n")
        else:
          # Handles CE_Total which is a float
          f.write(f"       {metric_name}: {metric_values}\n")
      reco = item.get("recommendation", "")
      f.write(f"\n{reco}\n")

  print(f"Saved top-k prompts to {filepath}")

def plot_metric_over_epochs(metric_values: dict, save_dir="Plots", model: str = "8b"):
  """
  Plots f1, accuracy, and CE losses (per-metric) and CE_Total over epochs for all evaluation metrics
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

  eval_metrics = [m for m in metric_values.keys() if m != "CE_Total"] # Split per-metric and CE_Total
  metric_types = ["f1", "accuracy"]

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

    save_path = os.path.join(save_dir, f"{metric_type}_scores_{model}.png")
    plt.savefig(save_path)
    print(f"Saved plot at '{save_path}'")
    plt.close()

    # Plot all CE metrics + CE_Total together
    plt.figure()

    # Plot per-metric CE
    for eval_metric in eval_metrics:
      values = metric_values[eval_metric][f"CE_{eval_metric}"]
      smoothed_values = ema(values)
      plt.plot(
        epochs,
        smoothed_values,
        marker='o',
        label=f"CE_{eval_metric}"
      )

    # Plot CE_Total in bold + black
    if "CE_Total" in metric_values:
      values = metric_values["CE_Total"]["CE_Total"]
      smoothed_values = ema(values)
      plt.plot(
        epochs,
        smoothed_values,
        marker='o',
        color='black',
        linewidth=3,
        label="CE_Total"
      )

    plt.title("EMA of Cross-Entropy Loss (Per-Metric + Total)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"CE_all_scores_{model}.png")
    plt.savefig(save_path)
    print(f"Saved combined plot at '{save_path}'")
    plt.close()

def write_replies(processed_replies: list, model: str = "8b") -> None:
  """
  Write all processed replies into a readable text format.

  Args:
      processed_replies (list): List of processed replies, each from one epoch.
      file_path (str): File path to write the replies.
  """
  file_path: str = f"processed_replies_{model}.txt"
  with open(file_path, "w", encoding="utf-8") as f:
    for epoch, reply in enumerate(processed_replies, start=1):
      f.write(f"Epoch {epoch}\n")
      f.write(f"Instruction: {reply.get('instruction', '').strip()}\n")
      f.write("Metrics:\n")
      for metric, value in reply.get("metrics", {}).items():
        f.write(f"  {metric}: {value}\n")
      f.write(f"Recommendation: {reply.get('recommendation', '').strip()}\n")
      f.write("\n\n\n")  # blank lines between epochs
  print(f"Processed replies successfully written to {file_path}")


def run_opro(
  file_path: str = "Dataset/dataset/df_M11_sampled.parquet",
  top_k: int = 10,
  num_epochs: int = 30,
  rater_llm_name: str = "meta-llama/llama-3-8b-instruct",
  reco_llm_name: str = "meta-llama/llama-3-8b-instruct",
  calls_per_minute: int = 120,
  max_workers: int = 10,
  num_examples: int = 100,
  rater_temp: float = 0.1,
  reco_temp: float = 1.0,
  rater_top_p: float = 0.95,
  reco_top_p: float = 0.95,
  top_k_most_imp_points: int = 10,
  model: str = "8b",
) -> dict:

  ## Top-K prompts
  top_k_prompts = TopKHeap(top_k)
  print(f"Step 1: Created a heap to store top-{top_k} prompts (Successful)")

  metric_names = ["coherence"] # "fluency", "consistency", "relevance","coherence"
  metric_history = {
    metric : {"f1" : [], "accuracy" : [], f"CE_{metric}" : []} for metric in metric_names
  }
  # metric_history["CE_Total"] = {"CE_Total" : []}
  processed_replies = [] # Keeps track of all the instruction, their metrics and their recommendations for each epoch

  for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
    instruction = call_rater_llm_meta_prompt(
      top_k_prompts=top_k_prompts,
      rater_llm_name=rater_llm_name,
      rater_temp=rater_temp, rater_top_p=rater_top_p,
    )
    print("Generated instruction")

    evals = call_rater_llm_prompt(
      instruction=instruction, file_path=file_path,
      rater_llm_name=rater_llm_name, max_workers=max_workers,
      calls_per_minute=calls_per_minute, num_examples=num_examples,
      rater_temp=rater_temp, rater_top_p=rater_top_p,
    )
    print("Generated evals")

    metrics = calculate_metrics(evals, file_path=file_path)
    print("Calculated metrics")

    top_points = find_most_imformative_points(evals, file_path=file_path, top_k=top_k_most_imp_points)
    print("Found most informative points")

    for metric in metric_names:
      metric_history[metric]["f1"].append(metrics[metric]["f1"])
      metric_history[metric]["accuracy"].append(metrics[metric]["accuracy"])
      metric_history[metric][f"CE_{metric}"].append(metrics[metric][f"CE_{metric}"])
      # metric_history[metric]["mean_diff"].append(metrics[metric]["mean_diff"])
    # metric_history["CE_Total"]["CE_Total"].append(metrics["CE_Total"])

    recommendation = call_recommender_llm(instruction, metrics, file_path=file_path, reco_llm_name=reco_llm_name, top_points=top_points, reco_temp=reco_temp, reco_top_p=reco_top_p)
    print("Generated recommendation")

    processed_reply = process_reply(instruction=instruction, recommendation=recommendation, heap=top_k_prompts, metrics=metrics)
    processed_replies.append(processed_reply)
    print("Processed Reply")

    if (epoch + 1) % 10 == 0:
      metric_history_file_path = f"metric_history_opro_{model}.txt"
      top_k_prompts_file_path = f"top_k_prompts_opro_{model}.txt"
      save_metric_history(metric_history, metric_history_file_path)
      save_top_k_prompts(top_k_prompts, top_k_prompts_file_path)
      write_replies(processed_replies)
      print(f"Checkpoint saved at epoch {epoch + 1}")

  for metric in metric_names:
    plot_metric_over_epochs(metric_values=metric_history, model=model)

  write_replies(processed_replies, model)
  return {
    "metric_histories": metric_history,
    "top_k_prompts": top_k_prompts,
    "processed_replies": processed_replies,
  }

def testing_results(
  opro_top_k_prompts: list[dict],
  rater_llm_name: str, rater_temp: float, rater_top_p: float,
  max_workers: int = 25, calls_per_minute: int = 120, num_examples: int = 480,
  test_file_path: str = "src/Dataset/dataset/cleaned_test_df.csv", model: str = "8b"
):
  if not opro_top_k_prompts:
    raise ValueError("opro_top_k_prompts is empty")

  best_prompt = opro_top_k_prompts[0]
  best_instruction = best_prompt["instruction"]
  # best_loss = best_prompt["loss"]

  print("Using the instruction to test: ", best_instruction)

  evals = call_rater_llm_prompt(
    instruction=best_instruction, file_path=test_file_path,
    rater_llm_name=rater_llm_name, rater_top_p=rater_top_p,
    rater_temp=rater_temp,max_workers=max_workers, calls_per_minute=calls_per_minute,
    num_examples=num_examples,
  )

  test_metrics = calculate_metrics(evals, file_path=test_file_path)
  print("Calculated metrics")

  output_file_path = f"test_performance_{model}.txt"
  with open(output_file_path, "w") as f:
    f.write(f"Best instruction:\n{best_instruction}\n\n")
    f.write(f"Test metrics:\n")
    for metric, values in test_metrics.items():
      f.write(f"{metric}: {values}\n")

  print(f"Output saved at {output_file_path}")
  return {
    "best_instruction": best_instruction,
    "test_metrics": test_metrics,
  }

def main(
    train_file_path: str,
    test_file_path: str,
    rater_llm_name: str,
    reco_llm_name: str,
    top_k: int = 10,
    num_epochs: int = 40,
    rater_temp: float = 0.1,
    reco_temp: float = 1.0,
    rater_top_p: float = 0.95,
    reco_top_p: float = 0.95,
    calls_per_minute: int = 60,
    max_workers: int = 20,
    train_num_examples: int = 160,
    test_num_examples: int = 480,
    model: str = "8b",
) -> None:
  opro_results = run_opro(
    file_path=train_file_path,
    rater_llm_name=rater_llm_name,
    reco_llm_name=reco_llm_name,
    model=model,
    top_k=top_k,
    num_epochs=num_epochs,
    rater_temp=rater_temp,
    reco_temp=reco_temp,
    rater_top_p=rater_top_p,
    reco_top_p=reco_top_p,
    calls_per_minute=calls_per_minute,
    max_workers=max_workers,
    num_examples=train_num_examples,
  )

  metric_history_file_path = f"metric_history_opro_{model}.txt"
  top_k_prompts_file_path = f"top_k_prompts_opro_{model}.txt"

  save_metric_history(opro_results["metric_histories"], metric_history_file_path)
  print(f"Metrics saved")
  save_top_k_prompts(opro_results["top_k_prompts"], top_k_prompts_file_path)
  print(f"Top K Prompts saved")

  test_results = testing_results(
    opro_top_k_prompts=opro_results["top_k_prompts"],
    rater_llm_name=rater_llm_name, test_file_path=test_file_path,
    rater_top_p=rater_top_p, rater_temp=rater_temp, model=model, num_examples=test_num_examples,
  )

  print("Test evaluation done")
  print(f"Best Instruction: {test_results['best_instruction']}")
  print(f"Test Metrics: {test_results['test_metrics']}")
  return

if __name__ == "__main__":
  rater_llm_name_8b = "meta-llama/llama-3.1-8b-instruct"
  reco_llm_name_8b = "meta-llama/llama-3.1-8b-instruct"
  test_filepath = "Dataset/dataset/cleaned_test_df.parquet"
  train_filepath = "Dataset/dataset/cleaned_train_df.parquet"

  rater_llm_name_70b = "meta-llama/llama-3.1-70b-instruct"
  reco_llm_name_70b = "meta-llama/llama-3.1-70b-instruct"

  main(
    train_file_path=train_filepath, test_file_path=test_filepath,
    rater_llm_name=rater_llm_name_8b,
    reco_llm_name=reco_llm_name_8b, top_k=10, num_epochs=50,
    rater_temp=0.0, reco_temp=0.0, rater_top_p=1.0, reco_top_p=1.0,
    calls_per_minute=75, max_workers=25, train_num_examples=160, model="8b",
    test_num_examples=480,
  )

  main(
    train_file_path=train_filepath, test_file_path=test_filepath,
    rater_llm_name=rater_llm_name_70b, reco_llm_name=reco_llm_name_70b,
    top_k=10, num_epochs=50,
    rater_temp=0.0, reco_temp=0.0, rater_top_p=1.0, reco_top_p=1.0,
    calls_per_minute=75, max_workers=25, train_num_examples=160, model="70b",
    test_num_examples=480,
  )