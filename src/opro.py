from src.Dataset.random_subsample import create_sample_points
from src.Recommender.recommender import call_evaluator_llm, process_reply
from src.Metrics.get_metrics import calculate_metrics
from src.Rater.rater import call_optimizer_llm
from src.TopK_Heap.top_k import TopKHeap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_epoch_scores(f1_scores: dict, acc_scores: dict, save_path="Plots/epoch_scores.png"):
  epochs = range(1, len(f1_scores["fluency"]) + 1)

  plt.figure(figsize=(12, 5))

  # Plot 1: F1 Scores
  plt.subplot(1, 2, 1)
  for metric, values in f1_scores.items():
    plt.plot(epochs, values, marker='o', label=metric)
  plt.title("F1 Scores Over Epochs")
  plt.xlabel("Epoch")
  plt.ylabel("F1 Score")
  plt.ylim(0, 1.05)
  plt.grid(True)
  plt.legend()

  # Plot 2: Accuracy Scores
  plt.subplot(1, 2, 2)
  for metric, values in acc_scores.items():
    plt.plot(epochs, values, marker='s', label=metric)
  plt.title("Accuracy Scores Over Epochs")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.ylim(0, 1.05)
  plt.grid(True)
  plt.legend()

  plt.tight_layout()
  path = f"{save_path}.png"
  plt.savefig(path)
  print(f"Saved F1/Accuracy plot at '{path}'")
  plt.close()

def run_opro(filepath: str, optim_llm_name: str, eval_llm_name: str, k: int  = 5, num_epochs: int = 3, run_id: int = 0) -> dict:
  ## Sampling sample points
  sample_points = create_sample_points(filepath)
  print(f"Run {run_id + 1} ==> Step 1: Creating {len(sample_points)} Sample points (Successful)")

  ## Top-K prompts
  top_k_prompts = TopKHeap(k)
  print(f"Run {run_id + 1} ==> Step 2: Created a heap to store top-{k} prompts (Successful)")

  optim_summaries = None

  # scores = {metric: [] for metric in ["fluency", "coherence", "consistency", "relevance"]}
  label_metrics = ["fluency", "coherence", "consistency", "relevance"]
  f1_scores = {metric: [] for metric in label_metrics}
  acc_scores = {metric: [] for metric in label_metrics}

  for epoch in range(num_epochs):
    print(f"Run {run_id + 1} ==> Epoch {epoch + 1}/{num_epochs}")

    optim_summaries = call_optimizer_llm(sample_points, top_k_prompts, optim_llm_name)
    print(f"Run {run_id + 1} ==> Epoch: {epoch} at Step 3: Generated initial summaries from Optim LLM (Successful)")

    print(f"Run {run_id + 1} ==>")
    print(optim_summaries)
    print("\n")

    # Retry until correct response format!
    while optim_summaries == {}:
      print(f"Run {run_id + 1} ==> Invalid Output from optim LLM, retrying this step!")
      optim_summaries = call_optimizer_llm(sample_points, top_k_prompts, optim_llm_name)

    ## Process the predicted summaries for F1 and Accuracy
    metrics = calculate_metrics(sample_points, optim_summaries)
    for metric in f1_scores.keys():
      f1_scores[metric].append(metrics[metric]["f1"])
      acc_scores[metric].append(metrics[metric]["accuracy"])

    eval_judgements = call_evaluator_llm(sample_points, optim_summaries, eval_llm_name)
    print(f"Run {run_id + 1} ==> Epoch: {epoch} at Step 4: Generating judgements (Successful)")

    eval_result = process_reply(eval_judgements, top_k_prompts, metrics)
    print(f"Run {run_id + 1} ==> Step 5: Processed evaluated judgements (Successful)")
    print('=' * 100)

    # for metric, score in eval_result["scores"].items():
    #   scores[metric].append(score)

  plot_path = f"Plots/epoch_scores_run_{run_id}.png"
  plot_epoch_scores(f1_scores, acc_scores, save_path=plot_path)

  return {
    "optim_summaries" : optim_summaries,
    "prev_top_k_prompts" : top_k_prompts,
  }

if __name__ == "__main__":
  eval_llm_name = "openai/gpt-4.1-nano"
  # optim_llm_name = "meta-llama/llama-3.2-3b-instruct:free"
  optim_llm_name = "deepseek/deepseek-r1-0528-qwen3-8b:free"
  filepath = "Dataset/dataset/df_model_M11.csv"
  opro_results = run_opro(filepath, optim_llm_name, eval_llm_name, num_epochs=3, run_id=0)