import requests, os, json
from dotenv import load_dotenv

from src.Dataset.random_subsample import create_sample_points
from src.Eval.e_prompt import create_evaluator_prompt
from src.Optim.optimizer import call_optimizer_llm

load_dotenv()

def call_evaluator_llm(optim_llm_response: dict ,eval_llm_name: str) -> str:
  """
  optim_llm_response: the response generated from OPTIM_LLM containing text, human_summary
  and machine_summary
  eval_llm_name: name of the evaluator llm to use
  """
  evaluator_prompt = create_evaluator_prompt(optim_llm_response)

  evaluator_response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
      "Content-Type": "application/json",
      "Authorization": "Bearer " + os.getenv("OPENROUTER_API_KEY")
    },
    data = json.dumps({
      "model" : eval_llm_name,
      "messages" : [
        {
          "role": "user",
          "content" : evaluator_prompt
        }
      ]
    })
  )

  eval_reply = evaluator_response.json()["choices"][0]["message"]["content"]
  # print(eval_reply)

  return eval_reply

if __name__ == '__main__':
  eval_llm_name = "google/gemini-2.0-flash-exp:free"
  optim_llm_name = "deepseek/deepseek-r1-0528-qwen3-8b:free"
  filepath = "../Dataset/summary_pairs.csv"

  sample_points = create_sample_points(filepath)

  optim_summaries = call_optimizer_llm(sample_points, optim_llm_name=optim_llm_name)

  eval_judgements = call_evaluator_llm(optim_summaries, eval_llm_name)
  print(eval_judgements)