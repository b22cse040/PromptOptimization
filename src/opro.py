import json
from src.Dataset.random_subsample import create_sample_points
from src.Eval.e_prompt import create_evaluator_prompt
from src.Eval.evaluator import call_evaluator_llm, process_reply
from src.Optim.optimizer import call_optimizer_llm
from src.TopK_Heap.top_k import TopKHeap
from tqdm import tqdm

def run_opro(filepath: str, optim_llm_name: str, eval_llm_name: str, k: int  = 5, num_epochs: int = 3) -> dict:
  ## Sampling sample points
  sample_points = create_sample_points(filepath)
  print(f"Step 1: Creating {len(sample_points)} Sample points (Successful)")

  ## Top-K prompts
  top_k_prompts = TopKHeap(k)
  print(f"Step 2: Created a heap to store top-{k} prompts (Successful)")

  optim_summaries = None

  for epoch in tqdm(range(num_epochs), desc="OPRO iterations"):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    optim_summaries = call_optimizer_llm(sample_points, top_k_prompts, optim_llm_name)
    print(f"Epoch: {epoch} at Step 3: Generated initial summaries from Optim LLM (Successful)")

    print(optim_summaries)
    print("\n")

    eval_judgements = call_evaluator_llm(optim_summaries, eval_llm_name)
    print(f"Epoch: {epoch} at Step 4: Generating judgements (Successful)")

    eval_result = process_reply(eval_judgements, top_k_prompts, optim_summaries["instruction"])
    print(f"Step 5: Processed evaluated judgements (Successful)")
    print('=' * 100)

    # top_k_prompts.push({
    #   "instruction": eval_result.get("instruction", ""),
    #   "scores": eval_result.get("scores", {}),
    #   "recommendation": eval_result.get("recommendation", ""),
    # })

  return {
    "optim_summaries" : optim_summaries,
    "prev_top_k_prompts" : top_k_prompts,
  }

if __name__ == "__main__":
  eval_llm_name = "moonshotai/kimi-k2:free"
  optim_llm_name = "mistralai/mistral-small-3.2-24b-instruct:free"
  filepath = "Dataset/summary_pairs.csv"
  opro_results = run_opro(filepath, optim_llm_name, eval_llm_name)
  top_k_prompts = TopKHeap(k=5)