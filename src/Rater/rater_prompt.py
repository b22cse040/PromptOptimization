import pandas as pd
from src.TopK_Heap.top_k import TopKHeap



_TASK_DESCRIPTION = """
  In this task you will evaluate the quality of summaries written for news article.
  To correctly solve this task, follow these steps:
  
  1. Carefully read the news article, be aware of the information it contains.
  2. Read the proposed summary.
  3. Rate each summary on a scale from 1 (worst) to 5 (best) by its fluency, coherence, consistency and relevance.
  
  Definitions:
   - Relevance: The rating measures how well the summary captures the key points of
     the article. Consider whether all and only the important aspects are contained in
     the summary.
   - Consistency: The rating measures whether the facts in the summary are consistent
     with the facts in the original article. Consider whether the summary does
     reproduce all facts accurately and does not make up untrue information.
   - Coherence: The rating measures the quality of all sentences collectively, to
     fit together and sound naturally. Consider the quality of the summary as a whole.
   - Fluency: The rating measures the quality of individual sentences, are they
     well-written and grammatically correct. Consider the quality of individual sentences.
"""


## sample_points: list of dicts
## prev_top_k_prompts is a list of dicts, each mapping an instruction string to
## a dict of score metrics.
def create_rater_meta_prompt(
    prev_top_k_prompts : TopKHeap = None,
    task_desc=_TASK_DESCRIPTION
):
  # sample_points_pairs_text = ""
  # for idx, pair in enumerate(sample_points, 1):
  #   sample_points_pairs_text += (
  #     f"Pair: {idx}\n"
  #     f"Text: {pair['text']}\n"
  #     # f"Human Summary: {pair['human_summary']}\n\n"
  #     f"Machine Summary: {pair['machine_summary']}\n"
  #     # f"Fluency: {pair['fluency']}\n"
  #     # f"Coherence: {pair['coherence']}\n"
  #     # f"Consistency: {pair['consistency']}\n"
  #     # f"Relevance: {pair['relevance']}\n"
  #   )

  prev_top_k_prompts_text = ""
  if prev_top_k_prompts:
    top_k_prompts = prev_top_k_prompts.get_topK()
    for idx, prompt_data in enumerate(top_k_prompts, 1):
      instruction = prompt_data["instruction"]
      recommendation = prompt_data["recommendation"]
      metrics = prompt_data["metrics"] if prompt_data["metrics"] else {}

      metric_lines = []
      for label in ["coherence"]: # "fluency", "consistency", "relevance", "coherence"
        label_metrics = metrics.get(label, {})
        acc = label_metrics.get("accuracy", 0.0)
        f1  = label_metrics.get("f1", 0.0)
        metric_lines.append(f"{label.title()} - Accuracy: {acc:.3f} - F1: {f1:.3f}")

      prev_top_k_prompts_text += (
        f"Instruction: {instruction}\n"
        + "\n".join(metric_lines) + "\n"
        f"Recommendation: {recommendation}\n\n"
      )

  _RATER_META_PROMPT = f"""
    You are an expert prompt optimizer working on improving summarization quality across 
    multiple evaluation aspects: fluency, coherence, consistency and relevance.
    
    This is the task description: {task_desc}
    
    Below is a list of previous top K prompts. Each prompt includes: 
    - The instruction given by the prompt.
    - Performance of the instruction in terms of accuracy and weighted f1 score.
    - Recommendations on the basis of previous prompts that will help you find the 
      new instruction.
    
    Previous top {len(prev_top_k_prompts)} prompts: 
    {prev_top_k_prompts_text}
    
    Based on the above previous top-{len(prev_top_k_prompts)} prompt's recommendations, 
    generate a new improved instruction that can be used to guide to judge the summarizations 
    (e.g., "Rate the summary of the article from 1 to 5 based on its coherence, relevance, fluency and consistency of sentences.").
    
    Do not add any commentary, markdown, or explanation. If you include anything else, the system will raise an error.
    Please adhere to the said output.
  """

  return _RATER_META_PROMPT

## The biggest difference between Meta Prompt and Prompt is that meta-prompt generates
## an instruction while Prompt uses that instruction to generate scores
def create_rater_prompt(instruction: str, run_id: int  = 0, file_path = "../Dataset/dataset/df_M11_sampled.parquet") -> str:
  df = pd.read_parquet(file_path)

  sample_point = df.iloc[run_id]
  sample_point_text = ""

  cols_to_be_ignored = [
    "human_summaries",
    "fluency",
    "coherence",
    "consistency",
    "relevance"
  ]

  for col in df.columns:
    if col in cols_to_be_ignored: continue
    sample_point_text += f"{col}: {sample_point[col]}\n"

  sample_point_text += '\n'
  sample_point_text = sample_point_text.strip()

  _RATER_PROMPT = f"""
Your task is written below, kindly complete this and return the output in the 
correct format.

Instruction : {instruction}

Your sample point that is to be rated is: 
{sample_point_text}

Output the summaries in a JSON Format of the form:
- Do not add any commentary, markdown, or explanation. As this will raise an error.
- Return strictly in a JSON format.

Format: 
{{
  "score" : {{
    "predicted_fluency" : 1|2|3|4|5,
    "predicted_coherence" : 1|2|3|4|5,
    "predicted_consistency" : 1|2|3|4|5,
    "predicted_relevance" : 1|2|3|4|5,
  }}
}}

Do not add any commentary, markdown, or explanation. As this will raise an error.
  """

  return _RATER_PROMPT

if __name__ == "__main__":
  prev_top_k = TopKHeap(3)

  instruction = "Filler"
  prompt = create_rater_prompt(instruction, run_id=25)
  print(prompt)
  print('=' * 100)
  print(f"Length of prompt: {len(prompt)}")
  # meta_prompt = create_optim_meta_prompt(prev_top_k)
  # print(meta_prompt)