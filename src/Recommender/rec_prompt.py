import pandas as pd
from src.Rater.rater import call_rater_llm_prompt, call_rater_llm_prompt, \
  call_rater_llm_meta_prompt
from src.Metrics.get_metrics import calculate_metrics, find_most_imformative_points
from src.TopK_Heap.top_k import TopKHeap
from typing import Dict

#   - The total average cross entropy loss of the said instruction.
_TASK_DESCRIPTION_RECOMMENDER = """

  You will be given:
  - A string that was the instruction received to the rater LLM to judge summaries
    on 4 metrics: fluency, coherence, consistency, relevance.
  - The performance of the said instruction on different samples.
  - The performance contains f1-score, accuracy, Cross-Entropy Loss for each metric.
  - A list of sample points that are the worst classified points or the ones with the
    highest Cross-Entropy Loss. You are also given their mean differences where:
      mean_diff = predicted_score - ground_truth_score,
      A positive mean_diff score indicates leniency in the Judgement of the LLM, 
      a negative mean_diff score indicates harshness in the Judgement of the LLM.

  Each sample is evaluated along the following four metrics, with score values ranging from 1 to 5:

  - Relevance: The rating measures how well the summary captures the key points of
  the article. Consider whether all and only the important aspects are contained in
  the summary.
  - Consistency: The rating measures whether the facts in the summary are consistent
  with the facts in the original article. Consider whether the summary does
  reproduce all facts accurately and does not make up untrue information.
  - Fluency: The rating measures the quality of individual sentences, are they
  well-written and grammatically correct. Consider the quality of individual sentences.
  - Coherence: The rating measures the quality of all sentences collectively, to
  fit together and sound naturally. Consider the quality of the summary as a whole.

  You may:
  - Recommend adjustments to metric definitions if misalignment is observed.
  - Highlight whether the current prompt is being lenient or harsh based on the mean_diff scores.
  - Propose revised instructions or guiding principles that would help the model better align with expert annotators.
  - Find a particular recommendation with the objective to minimize the loss with the maximum values.
   -> Bad example - "Improve coherence definition"
   -> Good example - "Revise the coherence instruction to emphasize 'logical transitions
      between sentences'.
  - DO NOT IN ANY DATA FROM THE SAMPLES INTO THE RECOMMENDATION AS THOSE ARE SOLELY FOR YOU.

  Be specific, grounded in the provided evidence, and focus on actionable improvements.
"""

#   - Find a particular recommendation with the objective to minimize all the losses simultaneously.
# _TASK_DESCRIPTION_RECOMMENDER = """
#
# You will be given:
# - A string that was the instruction received to the rater LLM to judge summaries
#   on coherence.
# - The performance of the said instruction on different samples.
# - The performance contains f1-score and accuracy.
#
# Each sample is evaluated along the following metric, with score values ranging from 1 to 5:
#
# - Coherence: The rating measures the quality of all sentences collectively, to
#   fit together and sound naturally. Consider the quality of the summary as a whole.
#
# You may:
# - Recommend adjustments to the fluency definition if misalignment is observed.
# - Highlight patterns in errors across multiple samples (e.g., facts don't match with the original article yet it receives a higher consistency rating.).
# - You may recommend harsher or more lenient scoring but you CANNOT tell the exact f1 and accuracy scores in your recommendation.
# - Propose revised instructions or guiding principles that would help the model better align with expert annotators.
#  -> Bad example - "Improve coherence definition"
#  -> Good example - "Revise the coherence instruction to emphasize whether all sentences collectively, to
#   fit together and sound naturally, and considering the quality of summary as a whole.
#
# Be specific, grounded in the provided evidence, and focus on actionable improvements.
# """


def create_recommender_prompt(
    instruction: str,
    metrics: dict,
    task_description: str = _TASK_DESCRIPTION_RECOMMENDER,
    top_points: list[dict] | None = None, # Optional Param
    file_path: str = "../Dataset/dataset/df_M11_sampled.parquet",
) -> str:

  df = pd.read_parquet(file_path)
  metrics_text = ""
  for key, value in metrics.items():
    if key == "CE_Total":
      # metrics_text += f"Average Cross Entropy Loss: {value}\n\n"
      continue

    metrics_text += f"Metric: {key}\n"

    for eval_metric, score in value.items():
      metrics_text += f"{eval_metric} of {key} : {score}\n"
    metrics_text += "\n"

  top_points_text = ""
  if top_points is not None and len(top_points) > 0:
    top_points_text += f"Most important points:\n"
    for p in top_points:
      point_idx = p.get("point_idx")
      lce = p.get("LCE", "N/A")
      mean_diff = p.get("mean_diff", "N/A")

      ## Pulling sample from dataframe
      try:
        sample = df.iloc[point_idx]
        text_val = sample.get("text", "N/A")
        machine_summary_val = sample.get("machine_summary", "N/A")
      except Exception as e:
        text_val, machine_summary_val = f"Error: {e}", "N/A"

      top_points_text += (
        f"- text: {text_val}\n"
        f"- machine_summary: {machine_summary_val}\n"
        f"- lce: {lce}\n"
        f"- mean_diff: {mean_diff}\n"
      )


  _RECOMMENDER_PROMPT = f"""
You are an expert at analyzing performance of instruction. Here is your task:
{task_description}

The instruction received from the rater LLM is: 
{instruction}.

The performance of the said instruction is:
{metrics_text}

The points with the most errors are:
{top_points_text}

Generate only what is asked. Add no other commentary or grammar than what is needed essentially.
A format for you: 
 
Recommendations:
  1. [First Recommendation, suggesting current scores are lenient or harsh and modify ratings accordingly. e.g., If the rater is lenient in fluency, suggest harsher rating for fluency metric]
  2. [Second recommendation, focusing on metric definitions or instruction clarity.]
  3. [Third recommendation, addressing observed biases or errors.]
  ...

  Do NOT include:
  - Any introductory/closing sentences.
  - Sections like "Instruction Analysis" or "Overall Performance."
  """

  return _RECOMMENDER_PROMPT

if __name__ == "__main__":
  rater_llm_name = "deepseek/deepseek-r1-0528-qwen3-8b:free"
  file_path = "../Dataset/dataset/df_M11_sampled.parquet"

  top_k_prompts = TopKHeap(3)

  instruction = call_rater_llm_meta_prompt(top_k_prompts, rater_llm_name)
  print(instruction)
  #
  evals = call_rater_llm_prompt(instruction, file_path=file_path, rater_llm_name=rater_llm_name, num_examples=20, max_workers=20)
  # print(evals)

  metrics = calculate_metrics(evals)
  print(metrics)

  top_points = find_most_imformative_points(evals)
  print(top_points)

  recommender_prompt = create_recommender_prompt(instruction, metrics, top_points=top_points)
  print(recommender_prompt)
  print('=' * 100)
  print(len(recommender_prompt))