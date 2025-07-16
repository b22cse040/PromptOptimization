_TASK_DESCRIPTION = """ 
  Generate improved summaries optimizing for fluency, coherence, consistency, 
  and relevance.
"""

def create_optim_meta_prompt(sample_points, prev_top_k_prompts, task_desc=_TASK_DESCRIPTION):
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
    
    Based on the above, generate a new improved instruction that can be used to guide 
    summarization (e.g., "Generate concise summaries emphasizing consistency and coherence.").
    
    Then, generate new improved machine summaries for each text using this new instruction.
    
    Respond ONLY with a valid JSON object of this form:

    {
        "instruction": "Your new summarization instruction here.",
        "sample_points": {
            1: {"text": "...", "human_summary": "...", "machine_summary": "..."},
            2: ...
        }
    }
  """

  return _OPTIM_META_PROMPT