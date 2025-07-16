from src.Dataset.random_subsample import create_sample_points

_TASK_DESCRIPTION = """ 
Generate improved summaries optimizing for fluency, coherence, consistency, and relevance.

- Fluency: The summary should be smooth, natural, and grammatically correct.
- Coherence: Ideas should flow logically and be easy to follow.
- Consistency: The summary must accurately reflect factual details from the original text.
- Relevance: The summary should include all important and informative content, avoiding unnecessary details.

Your goal is to produce high-quality summaries that excel in all four aspects.
"""


## sample_points: list of dicts
## prev_top_k_prompts is a list of dicts, each mapping an instruction string to
## a dict of score metrics.
def create_optim_meta_prompt(sample_points, prev_top_k_prompts=None, task_desc=_TASK_DESCRIPTION):
  sample_points_pairs_text = ""
  for idx, pair in enumerate(sample_points, 1):
    sample_points_pairs_text += (
      f"Pair: {idx}\n"
      f"Text: {pair['text']}\n"
      f"Human Summary: {pair['human_summary']}\n\n"
      # f"Machine Summary: {pair['machine_summary']}\n"
      # f"Fluency: {pair['fluency']}\n"
      # f"Coherence: {pair['coherence']}\n"
      # f"Consistency: {pair['consistency']}\n"
      # f"Relevance: {pair['relevance']}\n"
    )

  prev_top_k_prompts_text = ""
  for idx, prompt in enumerate(prev_top_k_prompts, 1):
    for instruction, score in prompt.items():
      prev_top_k_prompts_text += (
        f"{idx}, Instruction: {instruction} | ",
        f"Fluency: {score['Fluency']} | ",
        f"Coherence: {score['Coherence']} | ",
        f"Consistency: {score['Consistency']} | ",
        f"Relevance: {score['Relevance']}\n",
      )

  _OPTIM_META_PROMPT = f"""
    You are an expert prompt optimizer working on improving summarization quality across 
    multiple evaluation aspects: fluency, coherence, consistency, relevance.
    
    This is the task description: {task_desc}
    
    Below is a list of sample point pairs. Each pair includes: 
     - The original text to summarize
     - The human-written summary
     
    Sample point pairs: 
    {sample_points_pairs_text}
    
    Below is a list of previous top K prompts. Each prompt includes: 
    - The instruction given by the prompt.
    - Scores of fluency, coherence, consistency, relevance between 1 to 5.
    
    Previous top {len(prev_top_k_prompts)} prompts: 
    {prev_top_k_prompts_text}
    
    Based on the above, generate a new improved instruction that can be used to guide 
    summarization (e.g., "Generate concise summaries emphasizing consistency and coherence.").
    
    Then, generate new improved machine summaries for each text using this new instruction.
    
    Important: You must respond with STRICT VALID JSON only. 
    - Use double quotes (") for all keys and string values.
    - Do NOT include any explanations, comments or extra text before or after the JSON.
    - Do not include markdown code blocks (like ```json or ```).
    - Escape all internal newlines and quotes in string values if needed.
    
    Here is the required JSON structure example (strict format):
    
    {{
        "instruction": "Your new summarization instruction here.",
        "sample_points": {{
            "1": {{"text": "<string>", "human_summary": "<string>", "machine_summary": "<string>"}},
            "2": {{"text": "<string>", "human_summary": "<string>", "machine_summary": "<string>"}},
            ...
        }},
    }}
    
    Do not add any commentary, markdown, or explanation. If you include anything else, the system will raise an error.
  """

  return _OPTIM_META_PROMPT

if __name__ == "__main__":
  prev_top_k = []
  sample_points = create_sample_points(r"D:\PromptOptim\src\Dataset\summary_pairs.csv")
  meta_prompt = create_optim_meta_prompt(sample_points, prev_top_k)
  print(meta_prompt)