import pandas as pd
from src.Rater.rater import call_rater_llm_prompt, call_rater_llm_prompt, \
  call_rater_llm_meta_prompt
from src.Metrics.get_metrics import calculate_metrics, find_most_informative_points
from src.TopK_Heap.top_k import TopKHeap
from typing import Dict

# - The total average cross entropy loss of the said instruction.

def create_task_desc_recommender(metric_names: list[str], optimizer: str, is_top_points_given: bool) -> str:
  _METRIC_DEFINITIONS = {
    "relevance": """The rating measures how well the summary captures the key points of the article. Consider whether all and only the important aspects are contained in the summary.""",
    "consistency": """The rating measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.""",
    "fluency": """The rating measures the quality of individual sentences, are they well-written and grammatically correct. Consider the quality of individual sentences.""",
    "coherence": """The rating measures the quality of all sentences collectively, to fit together and sound naturally. Consider the quality of the summary as a whole."""
  }

  _OPTIMIZER_STATEMENT = {
    "min-all" : '''Select a recommendation that minimizes all loss functions simultaneously, aiming for the best overall balance across each metric.''',
    "min-max" : '''Select a recommendation that minimizes the highest (worst) loss value, while also reducing the remaining losses as much as possible.''',
    "pareto-optimal": '''Select a recommendation that lies on the Pareto frontier, meaning that no metric can be improved without worsening at least one other metric. Prioritize solutions that achieve strong trade-offs across all losses.'''
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

  is_top_points_given_text = ""
  if is_top_points_given:
    is_top_points_given_text = """
- Some points with the most cross-entropy loss will also be given, along with their mean-diff scores.
  mean_diff = predicted_score - ground_truth_score, 
  A positive mean_diff indicates the model is being lenient in their judgement, while a negative mean_diff indicates the model is being harsh in their judgement.
      """

  _TASK_DESCRIPTION_RECOMMENDER = f"""
You will be given:
- A string that was the instruction received to the rater LLM to judge summaries
  on {len(metric_names)} metrics: {", ".join(metric_names)}.
- The performance of the said instruction on different samples.
- The performance contains Cross-Entropy Loss for each metric.
{is_top_points_given_text}

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

def top_points_text_formatter(top_points: list[dict], top_points_format: str, df: pd.DataFrame) -> str:
  """
  Format the top_points into text according to the given top_points_format.
  :param top_points: list[dict]
  :param top_points_format: str ["calibrated", "raw", "calibrated_with_loss"]
  :param df: pd.DataFrame -> the dataframe to format
  :return: str -> Formatted string representation of top_points.
  """
  if not top_points:
    return "No informative points available."

  output = "Most important points:\n"

  for idx, p in enumerate(top_points, start=1):
    point_idx = p.get("point_idx")
    diffs = p.get("diffs", {})
    predicted_scores = p.get("predicted_scores") or {}
    loss = p.get("loss", 0.0)

    try:
      sample = df.iloc[point_idx]
      text_val = sample.get("text", "N/A")
      machine_summary_val = sample.get("machine_summary", "N/A")
    except Exception as e:
      text_val, machine_summary_val = f"Error: {e}", "N/A"
      sample = {}

    if top_points_format == "calibrated":
      output += (
        f"## Example {idx}:\n"
        f"- Text: {text_val}\n"
        f"- Machine Summary: {machine_summary_val}\n"
        f"- Task-wise Differences:\n"
      )

      for metric, delta in diffs.items():
        output += f"  - {metric.capitalize()}: {delta}\n"

    elif top_points_format == "raw":
      output += (
        f"## Example {idx}:\n"
        f"- Text: {text_val}\n"
        f"- Machine Summary: {machine_summary_val}\n"
        f"- Ground Truths:\n"
      )
      for metric in diffs.keys():
        ground_val = sample.get(metric, "N/A")
        output += f"  - {metric.capitalize()}: {ground_val}\n"

      output += ("\n- Predicted:\n")
      for pred_metric, pred_val in predicted_scores.items():
        metric = pred_metric.replace("predicted_", "")
        output += f"  - {metric.capitalize()}: {pred_val}\n"
      output +=('\n\n')

    elif top_points_format == "calibrated_with_loss":
      output += (
        f"## Example {idx}:\n"
        f"- Text: {text_val}\n"
        f"- Machine Summary: {machine_summary_val}\n"
        f"- Task-wise Differences:\n"
      )
      for metric, delta in diffs.items():
        output += f"  - {metric.capitalize()}: {delta}\n"
      output += (f"Loss: {loss}\n")

    else:
      raise ValueError(f"Unknown top_points_format: {top_points_format}")

  return output

def create_recommender_prompt(
    instruction: str,
    evals: dict,
    metric_names: list[str],
    optimizer: str,
    reco_format: str,
    top_points_format: str | None = None,
    top_points: list[dict] | None = None, # Optional Param
    file_path: str = "../Dataset/dataset/df_M11_sampled.parquet",
) -> str:

  ## Read file for importing max error points
  df = pd.read_parquet(file_path)

  ## Create recommender task description
  if reco_format == "ours-unnamed": task_description = create_task_desc_recommender(metric_names, optimizer=optimizer, is_top_points_given=True)
  else: task_description = create_task_desc_recommender(metric_names, optimizer=optimizer, is_top_points_given=False)

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

  top_points_text = top_points_text_formatter(top_points, top_points_format, df=df)

  _RECOMMENDER_PROMPT_RECOMMENDATION_FORMAT = {
    "OPRO": """
  1. [First Recommendation, focusing on what did the instruction got wrong.]
  2. [Second recommendation, focusing on metric definitions or instruction clarity.]
  3. [Third recommendation, addressing observed biases or errors.]""",

    "ours-unnamed": """
  1. [First recommendation, evaluate if the LLM’s scoring is overly strict or too lenient, and suggest how to adjust ratings (increase, decrease, or keep consistent).]
  2. [Second recommendation, check whether the metric definitions or task instructions are clear enough, and propose refinements if they may cause confusion.]
  3. [Third recommendation, identify any systematic biases, recurring mistakes, or misalignments in scoring, and suggest corrective adjustments.]
  """
  }

  if reco_format not in _RECOMMENDER_PROMPT_RECOMMENDATION_FORMAT:
    raise ValueError(f"Recommender Format {reco_format} is not supported.")

  recommendation_text = _RECOMMENDER_PROMPT_RECOMMENDATION_FORMAT[reco_format]


  _RECOMMENDER_PROMPT_TEMPLATE = {
    "OPRO": f"""
You are an expert at analyzing performance of instruction. Here is your task:
{task_description}

The instruction received from the rater LLM is:
{instruction}.

The performance of the said instruction is:
{evals_text}

Generate only what is asked. Add no other commentary or grammar than what is needed essentially.
A format for you:

Recommendations: {recommendation_text}
  ...

  Do NOT include:
  - Any introductory/closing sentences.
  - Sections like "Instruction Analysis" or "Overall Performance."
  """,

    "ours-unnamed": f"""
You are an expert at analyzing performance of instruction. Here is your task:
{task_description}

The instruction received from the rater LLM is:
{instruction}.

The performance of the said instruction is:
{evals_text}

The points with the most errors are: 
{top_points_text}

Generate only what is asked. Add no other commentary or grammar than what is needed essentially.
A format for you:

Recommendations: {recommendation_text}
  ...

  Do NOT include:
  - Any introductory/closing sentences.
  - Sections like "Instruction Analysis" or "Overall Performance."
  """,
  }

  _RECOMMENDER_PROMPT = _RECOMMENDER_PROMPT_TEMPLATE[reco_format]

  return _RECOMMENDER_PROMPT

if __name__ == "__main__":
  rater_llm_name = "meta-llama/llama-3-8b-instruct"
  file_path = "../Dataset/dataset/cleaned_test_df.parquet"
  metric_names = ["relevance", "consistency", "fluency", "coherence"]
  # _task_recommender = create_task_desc_recommender(metric_names=metric_names, optimizer='min-all', is_top_points_given=False)
  # print(_task_recommender)
  # print('=' * 100)

  top_k_prompts = TopKHeap(3)

  instruction = call_rater_llm_meta_prompt(top_k_prompts, metric_names=metric_names, rater_llm_name=rater_llm_name, rater_temp=0.0, rater_top_p=0.95)
  print(instruction)
  print('=' * 100)
  #
  evals = call_rater_llm_prompt(instruction, metric_names=metric_names, file_path=file_path, rater_llm_name=rater_llm_name, num_examples=20, max_workers=20, rater_top_p=0.95, rater_temp=0.0)
  # print(evals)

  metrics = calculate_metrics(evals, metric_names=metric_names)
  print(metrics)
  print('=' * 100)

  top_points = find_most_informative_points(evals, metric_names=metric_names)
  print(top_points)
  print('=' * 100)

  # df = pd.read_parquet(file_path)
  # top_points_text = top_points_text_formatter(top_points, "calibrated", df)
  # print(top_points_text)
  # print('=' * 100)

  recommender_prompt = create_recommender_prompt(instruction, metrics, metric_names=metric_names, optimizer='min-all', reco_format="ours-unnamed", top_points=top_points, top_points_format="calibrated")
  print(recommender_prompt)
  print('=' * 100)
  print(len(recommender_prompt))

  print('=' * 100)

  recommender_prompt = create_recommender_prompt(
    instruction, metrics, metric_names=metric_names,
    optimizer='min-all', reco_format="ours-unnamed", top_points=top_points,
    top_points_format="calibrated"
  )

  print(recommender_prompt)
  print('=' * 100)
  print(len(recommender_prompt))