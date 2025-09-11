import pandas as pd
from src.Rater.rater import call_rater_llm_prompt, call_rater_llm_prompt, \
  call_rater_llm_meta_prompt
from src.Metrics.get_metrics import calculate_metrics, find_most_informative_points
from src.TopK_Heap.top_k import TopKHeap
from typing import Dict

# - The total average cross entropy loss of the said instruction.

def create_task_desc_recommender(metric_names: list[str], optimizer: str) -> str:
  _METRIC_DEFINITIONS = {
    "relevance": """The rating measures how well the summary captures the key points of the article. Consider whether all and only the important aspects are contained in the summary.""",
    "consistency": """The rating measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.""",
    "fluency": """The rating measures the quality of individual sentences, are they well-written and grammatically correct. Consider the quality of individual sentences.""",
    "coherence": """The rating measures the quality of all sentences collectively, to fit together and sound naturally. Consider the quality of the summary as a whole."""
  }

  _OPTIMIZER_STATEMENT = {
    "min-all" : '''Find a particular recommendation with the objective to minimize all the losses simultaneously.''',
    "min-max" : '''Find a particular recommendation with the objective to minimize the loss with the maximum values while also minimize others.'''
  }

  if optimizer not in _OPTIMIZER_STATEMENT:
    raise ValueError(f"Optimizer {optimizer} is not supported. Choose 'min-all' or 'min-max'.")

  metrics_text = ""
  for metric_name in metric_names:
    if metric_name not in _METRIC_DEFINITIONS:
      print(f"Metric {metric_name} not found. Try again.")
      continue

    metrics_text += (
      f"- {metric_name} : {_METRIC_DEFINITIONS[metric_name]}\n"
    )

  _TASK_DESCRIPTION_RECOMMENDER = f"""
You will be given:
- A string that was the instruction received to the rater LLM to judge summaries
  on {len(metric_names)} metrics: {", ".join(metric_names)}.
- The performance of the said instruction on different samples.
- The performance contains Cross-Entropy Loss for each metric.

Each sample is evaluated along the following four metrics, with score values ranging from 1 to 5:
{metrics_text}


You may:
- Recommend adjustments to metric definitions if misalignment is observed.
- Propose revised instructions or guiding principles that would help the model better align with expert annotators.
- {_OPTIMIZER_STATEMENT[optimizer]}
 -> Bad example - "Improve relevance definition"
 -> Good example - "Revise the coherence instruction to emphasize capturing key points of the article, and whether all the important aspects are covered."
- DO NOT INPUT ANY DATA FROM THE SAMPLES INTO THE RECOMMENDATION AS THOSE ARE SOLELY FOR YOU.

Be specific, grounded in the provided evidence, and focus on actionable improvements.
  """
  return _TASK_DESCRIPTION_RECOMMENDER


def create_recommender_prompt(
    instruction: str,
    evals: dict,
    metric_names: list[str],
    optimizer: str,
    top_points: list[dict] | None = None, # Optional Param
    file_path: str = "../Dataset/dataset/df_M11_sampled.parquet",
) -> str:

  ## Read file for importing max error points
  df = pd.read_parquet(file_path)

  ## Create recommender task description
  task_description = create_task_desc_recommender(metric_names, optimizer=optimizer)

  evals_text = ""
  for key, value in evals.items():
    if key == "CE_Total":
      # metrics_text += f"Average Cross Entropy Loss: {value}\n\n"
      continue

    evals_text += f"Metric: {key}\n"

    for eval_metric, score in value.items():
      if eval_metric == "accuracy" or eval_metric == "f1": continue
      evals_text += f"Cross-Entropy Loss of {key} : {score}\n"
    evals_text += "\n"

  top_points_text = ""
  if top_points is not None and len(top_points) > 0:
    top_points_text += f"Most important points:\n"
    i = 0
    for p in top_points:
      point_idx = p.get("point_idx")
      lce = p.get("LCE", "N/A")
      mean_diff = p.get("mean_diff", "N/A")
      i += 1
      ## Pulling sample from dataframe
      try:
        sample = df.iloc[point_idx]
        text_val = sample.get("text", "N/A")
        machine_summary_val = sample.get("machine_summary", "N/A")
      except Exception as e:
        text_val, machine_summary_val = f"Error: {e}", "N/A"

      top_points_text += (
        f"## Example: {i}\n"
        f"- text: {text_val}\n"
        f"- machine_summary: {machine_summary_val}\n"
        # f"- lce: {lce}\n"
        f"- mean_diff: {mean_diff}\n"
      )


  _RECOMMENDER_PROMPT = f"""
You are an expert at analyzing performance of instruction. Here is your task:
{task_description}

The instruction received from the rater LLM is:
{instruction}.

The performance of the said instruction is:
{evals_text}


Generate only what is asked. Add no other commentary or grammar than what is needed essentially.
A format for you:

Recommendations:
  1. [First Recommendation, focusing on what did the instruction got wrong.]
  2. [Second recommendation, focusing on metric definitions or instruction clarity.]
  3. [Third recommendation, addressing observed biases or errors.]
  ...

  Do NOT include:
  - Any introductory/closing sentences.
  - Sections like "Instruction Analysis" or "Overall Performance."
  """

  return _RECOMMENDER_PROMPT

if __name__ == "__main__":
  rater_llm_name = "meta-llama/llama-3-8b-instruct"
  file_path = "../Dataset/dataset/cleaned_test_df.parquet"
  metric_names = ["relevance", "consistency"]
  _task_recommender = create_task_desc_recommender(metric_names=metric_names, optimizer='min-all')
  # print(_task_recommender)

  top_k_prompts = TopKHeap(3)

  instruction = call_rater_llm_meta_prompt(top_k_prompts, metric_names=metric_names, rater_llm_name=rater_llm_name, rater_temp=0.0, rater_top_p=0.95)
  print(instruction)
  #
  evals = call_rater_llm_prompt(instruction, metric_names=metric_names, file_path=file_path, rater_llm_name=rater_llm_name, num_examples=20, max_workers=20, rater_top_p=0.95, rater_temp=0.0)
  # print(evals)

  metrics = calculate_metrics(evals, metric_names=metric_names)
  print(metrics)

  top_points = find_most_informative_points(evals, metric_names=metric_names)
  print(top_points)

  recommender_prompt = create_recommender_prompt(instruction, metrics, metric_names=metric_names, optimizer='min-all', top_points=top_points)
  print(recommender_prompt)
  print('=' * 100)
  print(len(recommender_prompt))