from src.Dataset.random_subsample import create_sample_points
from src.TopK_Heap.top_k import TopKHeap

_TASK_DESCRIPTION = """
  In this task you will evaluate the quality of summaries written for news article.
  To correctly solve this task, follow these steps:
  
  1. Carefully read the news article, be aware of the information it contains.
  2. Read the proposed summary.
  3. Rate each summary on a scale from 1 (worst) to 5 (best) by its relevance, 
  consistency, fluency and coherence.
  
  Definitions:
  
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
"""


## sample_points: list of dicts
## prev_top_k_prompts is a list of dicts, each mapping an instruction string to
## a dict of score metrics.
def create_optim_meta_prompt(prev_top_k_prompts : TopKHeap =None, task_desc=_TASK_DESCRIPTION):
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
      for label in ["fluency", "coherence", "consistency", "relevance"]:
        label_metrics = metrics.get(label, {})
        acc = label_metrics.get("accuracy", 0.0)
        f1  = label_metrics.get("f1", 0.0)
        metric_lines.append(f"{label.title()} - Accuracy: {acc:.3f} - F1: {f1:.3f}")

      prev_top_k_prompts_text += (
        f"Instruction: {instruction}\n"
        + "\n".join(metric_lines) + "\n"
        f"Recommendation: {recommendation}\n\n"
      )

  _OPTIM_META_PROMPT = f"""
    You are an expert prompt optimizer working on improving summarization quality across 
    multiple evaluation aspects: fluency, coherence, consistency, relevance.
    
    This is the task description: {task_desc}
    
    Below is a list of previous top K prompts. Each prompt includes: 
    - The instruction given by the prompt.
    - Scores' difference between the judges and LLM scores of fluency, coherence, 
      consistency, relevance between 0 to 5.
    - Recommendations on the basis of previous prompts that will help you find the 
      new instruction.
    
    Previous top {len(prev_top_k_prompts)} prompts: 
    {prev_top_k_prompts_text}
    
    Based on the above, generate a new improved instruction that can be used to guide 
    to judge the summarizations (e.g., "Rate the summary of the article from 1 to 
    5 based on its relevance, consistency with facts, fluency of sentences, and 
    coherence as a whole.").
    
    Do not add any commentary, markdown, or explanation. If you include anything else, the system will raise an error.
    Please adhere to the said output.
  """

  return _OPTIM_META_PROMPT

if __name__ == "__main__":
  prev_top_k = TopKHeap(3)
  sample_points = create_sample_points(
    r"../Dataset/dataset/df_model_M11.csv")

  meta_prompt = create_optim_meta_prompt(prev_top_k)
  print(meta_prompt)