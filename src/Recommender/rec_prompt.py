from src.Rater.rater import call_rater_llm_prompt, call_rater_llm_prompt, \
  call_rater_llm_meta_prompt
from src.Metrics.get_metrics import calculate_metrics
from src.TopK_Heap.top_k import TopKHeap
from typing import Dict

_TASK_DESCRIPTION_RECOMMENDER = """

  You will be given:
  - A string that was the instruction received to the rater LLM to judge summaries
    on 4 metrics: fluency, coherence, consistency, relevance.
  - The performance of the said instruction on different samples.
  - The performance contains f1-score, accuracy and mean-difference for each metric.
    -> mean-difference is the average of (predicted_score - ground_truth_score) for all the samples.
    A positive mean-difference indicates the rater is more lenient than it should be, while a negative
    mean-difference indicates the rater is less lenient than it should be.

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
  - Highlight patterns in errors across multiple samples (e.g., consistently underrating coherence).
  - Propose revised instructions or guiding principles that would help the model better align with expert annotators.
   -> Bad example - "Improve coherence definition"
   -> Good example - "Revise the coherence instruction to emphasize 'logical transitions
      between sentences'.

  Be specific, grounded in the provided evidence, and focus on actionable improvements.
"""

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
) -> str:

  metrics_text = ""
  for key, value in metrics.items():
    metrics_text += f"Metric: {key}\n"

    for eval_metric, score in value.items():
      metrics_text += f"{eval_metric} of {key} : {score}\n"
    metrics_text += "\n"

  _RECOMMENDER_PROMPT = f"""
You are an expert at analyzing performance of instruction. Here is your task:
{task_description}

The instruction received from the rater LLM is: 
{instruction}.

The performance of the said instruction is:
{metrics_text}

Generate only what is asked. Add no other commentary or grammar than what is needed essentially.
A format for you: 
 
Recommendations:
  1. [First recommendation, focusing on metric definitions or instruction clarity.]
  2. [Second recommendation, addressing observed biases or errors.]
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
  evals = call_rater_llm_prompt(instruction, file_path=file_path, rater_llm_name=rater_llm_name, num_examples=5, max_workers=2)
  # print(evals)

  metrics = calculate_metrics(evals)
  print(metrics)

  recommender_prompt = create_recommender_prompt(instruction, metrics)
  print(recommender_prompt)
  print('=' * 100)
  print(len(recommender_prompt))