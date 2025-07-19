from src.Optim.o_prompt import _TASK_DESCRIPTION, create_optim_meta_prompt
from src.Optim.optimizer import call_optimizer_llm
from src.Dataset.random_subsample import create_sample_points

def create_evaluator_prompt(optim_llm_response : dict, task_desc=_TASK_DESCRIPTION) -> str:
  # instruction: str = optim_llm_response["instruction"]
  sample_points_response = optim_llm_response["sample_points"]
  optim_llm_response_text = ""
  for idx, sample_point in sample_points_response.items():
    # idx = int(idx_str)
    optim_llm_response_text += (
      f"Point: {idx}\n"
      f"text: {sample_point['text']}\n"
      f"human_summary: {sample_point['human_summary']}\n"
      f"machine_summary: {sample_point['machine_summary']}\n\n"
    )

  _EVALUATOR_PROMPT = f"""
    You are an expert summarization evaluator. Your task is to act as a strict, fair judge.
    
    Task description: 
    {_TASK_DESCRIPTION}
    
    For each aspect, assign a score from 1 to 5 based on
    how aligned they are with human summaries of the same text. (1 = very poorly 
    aligned, 5 = excellent alignment).
    
    Below are the sample points. Each point includes the original text, the human written summary (if given)
    and the machine-generated summary:
    
    {optim_llm_response_text}
    
    Return your evaluation strictly as a valid JSON object in the following format:
    
    {{
      "instruction": "<Copy the summarization instruction provided in the input>",
      "scores": {{
        "1": {{"fluency": int, "coherence": int, "consistency": int, "relevance": int}},
        "2": {{"fluency": int, "coherence": int, "consistency": int, "relevance": int}},
        ...
      }},
      "recommendation": "Provide specific, actionable suggestions to improve the overall scores (e.g., improve factual coverage, fix grammar, simplify phrasing, make summaries more concise, etc.)
      This should be done in under 100 tokens."
    }}
    
    Respond only with this JSON object. Do not include any additional explanation or commentary.
  """

  return _EVALUATOR_PROMPT

if __name__ == "__main__":
  optim_llm_name = "deepseek/deepseek-r1-0528-qwen3-8b:free"
  filepath = "../Dataset/dataset/summary_pairs.csv"

  sample_points = create_sample_points(filepath)
  # print(sample_points)

  optim_llm_response = call_optimizer_llm(sample_points, optim_llm_name)
  print(optim_llm_response)

  evaluator_prompt = create_evaluator_prompt(optim_llm_response)
  print(evaluator_prompt)