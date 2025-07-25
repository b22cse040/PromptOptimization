import requests, os, re
import json5
from dotenv import load_dotenv
from src.OpenRouter.openrouter import call_openrouter
from src.Rater.rater_prompt import create_rater_meta_prompt, create_rater_prompt
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

def call_rater_llm_meta_prompt(top_k_prompts: TopKHeap, rater_llm_name: str):
  """
  rater_llm_name: Name of the optimizer llm to use.
  """
  rater_meta_prompt = create_rater_meta_prompt(prev_top_k_prompts=top_k_prompts)
  # print(optim_meta_prompt)

  rater_response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
      "Content-Type": "application/json",
      "Authorization": "Bearer " + os.getenv("OPENROUTER_API_KEY")
    },
    data=json5.dumps({
      "model" : rater_llm_name,
      "messages" : [
        {
          "role"  : "user",
          "content" : rater_meta_prompt
        }
      ],
    })
  )

  reply = rater_response.json()["choices"][0]["message"]["content"]
  # print(reply)
  # reply = clean_response(reply)
  return reply

def call_rater_llm_prompt(
    instruction: str,
    run_id: int = 0,
    file_path : str = "../Dataset/dataset/df_M11_sampled.parquet",
    rater_llm_name: str = "deepseek/deepseek-r1-0528-qwen3-8b:free"
):
  # Forming Prompt
  rater_prompt = create_rater_prompt(instruction, run_id=run_id, file_path=file_path)

  # Generating JSON Response
  reply = call_openrouter(rater_prompt, rater_llm_name)

  # Cleaning JSON response
  reply = clean_response(reply)
  return reply

if __name__ == "__main__":
  rater_llm_name = "deepseek/deepseek-r1-0528-qwen3-8b:free"
  filepath = "../Dataset/dataset/df_model_M11.csv"
  top_k_prompts = TopKHeap(3)
  print("Calling Optimizer!")
  new_instruction = call_rater_llm_meta_prompt(top_k_prompts, rater_llm_name)
  print(new_instruction)
  print('=' * 100)
  # print(type(new_instruction))

  ## Rating summary
  summary = call_rater_llm_prompt(new_instruction, 0, rater_llm_name=rater_llm_name)
  print(summary)
  # print(type(summary))