import requests, os, re
import json5
from typing import Dict
from dotenv import load_dotenv
from src.Dataset.random_subsample import create_sample_points
from src.Optim.o_prompt import create_optim_meta_prompt
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

def call_optimizer_llm(sample_points: list[Dict[str, str]], top_k_prompts: TopKHeap, optim_llm_name: str):
  """
  sample_points: list of dict containing samples sampled randomly.
  optim_llm_name: Name of the optimizer llm to use.
  """
  optim_meta_prompt = create_optim_meta_prompt(sample_points, prev_top_k_prompts=top_k_prompts)
  # print(optim_meta_prompt)

  optim_response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
      "Content-Type": "application/json",
      "Authorization": "Bearer " + os.getenv("OPENROUTER_API_KEY")
    },
    data=json5.dumps({
      "model" : optim_llm_name,
      "messages" : [
        {
          "role"  : "user",
          "content" : optim_meta_prompt
        }
      ],
    })
  )

  reply = optim_response.json()["choices"][0]["message"]["content"]
  # print(reply)
  reply = clean_response(reply)
  return reply

if __name__ == "__main__":
  optim_llm_name = "deepseek/deepseek-r1-0528-qwen3-8b:free"
  filepath = "../Dataset/summary_pairs.csv"
  sample_points = create_sample_points(filepath)
  # print(sample_points)
  top_k_prompts = TopKHeap(3)
  reply = call_optimizer_llm(sample_points, top_k_prompts, optim_llm_name)
  print(reply)
  print(type(reply))