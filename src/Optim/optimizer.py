import requests, os
import json
from dotenv import load_dotenv
from src.Dataset.random_subsample import create_sample_points
from src.Optim.o_prompt import create_optim_meta_prompt

load_dotenv()

def call_optimizer_llm(file_path: str, optim_llm_name: str):
  """
  file_path: File to the dataset to randomly subsample for the prompt.
  optim_llm_name: Name of the optimizer llm to use.
  """
  sample_points = create_sample_points(file_path)
  print(sample_points)
  optim_meta_prompt = create_optim_meta_prompt(sample_points, prev_top_k_prompts=[])
  print(optim_meta_prompt)

  optim_response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
      "Content-Type": "application/json",
      "Authorization": "Bearer " + os.getenv("OPENROUTER_API_KEY")
    },
    data=json.dumps({
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

  return reply

if __name__ == "__main__":
  optim_llm_name = "deepseek/deepseek-r1-0528-qwen3-8b:free"
  filepath = "../Dataset/summary_pairs.csv"
  reply = call_optimizer_llm(filepath, optim_llm_name)
  print(reply)