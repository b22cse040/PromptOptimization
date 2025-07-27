import re
from tqdm import tqdm
import json5
from dotenv import load_dotenv
from src.OpenRouter.openrouter import call_openrouter
from src.Rater.rater_prompt import create_rater_meta_prompt, create_rater_prompt
from src.asyncio_executor import RestrictedConcurrencyThreadPoolExecutor
from src.TopK_Heap.top_k import TopKHeap

load_dotenv()

def clean_response(reply: str) -> dict:
  try:
    cleaned_reply = re.sub(r'^```json|```$', '', reply,
                           flags=re.MULTILINE).strip()

    data = json5.loads(cleaned_reply)
    return data
  except Exception as e:
    print(f"Error decoding JSON: {e}")
    return {}

def call_rater_llm_meta_prompt(top_k_prompts: TopKHeap, rater_llm_name: str) -> str:
  """
  rater_llm_name: Name of the optimizer llm to use.
  """
  rater_meta_prompt = create_rater_meta_prompt(prev_top_k_prompts=top_k_prompts)
  # print(optim_meta_prompt)

  # No need to clean the response as it returns a string.
  reply = call_openrouter(rater_meta_prompt, rater_llm_name)
  return reply

def call_rater_llm_prompt_utils(
    instruction: str,
    run_id: int = 0,
    file_path : str = "../Dataset/dataset/df_M11_sampled.parquet",
    rater_llm_name: str = "deepseek/deepseek-r1-0528-qwen3-8b:free"
) -> dict:
  # Forming Prompt : returns a str containing the prompt for rater LLM
  # print(f"Run {run_id}: LLM Prompt formation")
  rater_prompt = create_rater_prompt(instruction, run_id=run_id, file_path=file_path)

  # Generating JSON Response by the Rater LLM
  # print(f"Run {run_id}: LLM Calling")
  reply = call_openrouter(rater_prompt, rater_llm_name)

  # Cleaning JSON response
  # print(f"Run {run_id}: LLM reply received")
  reply = clean_response(reply)
  return reply

def call_rater_llm_prompt(
    instruction: str,
    file_path: str = "../Dataset/dataset/df_M11_sampled.parquet",
    num_examples : int = 100,
    max_workers: int = 10,
    calls_per_minute: int = 120,
    rater_llm_name : str = "meta-llama/llama-3-8b-instruct"
) -> list[dict]:

  calls_per_second = calls_per_minute / 60
  executor = RestrictedConcurrencyThreadPoolExecutor(
    max_workers=max_workers,
    max_calls_per_second=calls_per_second,
  )
  print("Formed Executor!")

  futures = [
    executor.submit(
      call_rater_llm_prompt_utils,
      instruction, run_id, file_path, rater_llm_name
    ) for run_id in range(num_examples)
  ]
  print("Submitted All futures")

  results: list[dict] = [
    fut.result() for fut in tqdm(futures, desc="Processing Prompts")
  ]
  print("Res")
  return results

if __name__ == "__main__":
  rater_llm_name = "meta-llama/llama-3-8b-instruct"
  filepath = "../Dataset/dataset/df_M11_sampled.parquet"
  num_examples = 100
  max_concurrent_calls = 20
  calls_per_minute = 120
  calls_per_second = calls_per_minute / 60

  top_k_prompts = TopKHeap(3)
  print("Calling Optimizer!")
  new_instruction = call_rater_llm_meta_prompt(top_k_prompts, rater_llm_name)
  print(new_instruction)
  print('=' * 100)

  results = call_rater_llm_prompt(new_instruction, rater_llm_name=rater_llm_name)

  print("\nFinal Results:")
  for res in results:
    print(res, '\n\n')