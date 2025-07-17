import requests, os, json, re
import numpy as np
from dotenv import load_dotenv

from src.Dataset.random_subsample import create_sample_points
from src.Eval.e_prompt import create_evaluator_prompt
from src.Optim.optimizer import call_optimizer_llm

load_dotenv()

def clean_response(reply: str) -> dict:
  try:
    cleaned_reply = re.sub(r'^```json|```$', '', reply,
                           flags=re.MULTILINE).strip()

    data = json.loads(cleaned_reply)
    return data
  except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    return {}
  except Exception as e:
    print(f"Error decoding JSON: {e}")
    return {}

def process_reply(eval_reply: dict) -> dict:
  """
  Process the reply for meta-prompt. Averaging the scores for all samples.
  """
  scores = eval_reply.get("scores", {})
  metrics = ["fluency", "coherence", "consistency", "relevance"]

  metric_values = {metric: [] for metric in metrics}

  for idx_scores in scores.values():
    for metric in metrics:
      if metric in idx_scores:
        metric_values[metric].append(idx_scores[metric])

  averaged_scores = {
    metric: round(float(np.mean(values)), 2) if values else 0.0
    for metric, values in metric_values.items()
  }

  return {
    "instruction": eval_reply.get("instruction", ""),
    "scores" : averaged_scores,
    "recommendations": eval_reply.get("recommendation", ""),
  }

def call_evaluator_llm(optim_llm_response: dict, eval_llm_name: str) -> dict:
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
  eval_reply = clean_response(eval_reply)
  return eval_reply

if __name__ == '__main__':
  # optim_llm_name = "google/gemini-2.0-flash-exp:free"
  optim_llm_name = "deepseek/deepseek-r1-0528-qwen3-8b:free"
  eval_llm_name = "deepseek/deepseek-r1-0528-qwen3-8b:free"
  filepath = "../Dataset/summary_pairs.csv"

  sample_points = create_sample_points(filepath)
  # print(f"Sample points created: {len(sample_points)}")

  optim_summaries = call_optimizer_llm(sample_points, optim_llm_name=optim_llm_name)
  # print(optim_summaries)
  print(f"OPTIM_LLM: Generated machine summaries")

  eval_judgements = call_evaluator_llm(optim_summaries, eval_llm_name)
  print(f"EVAL_LLM: Judging generated machine summaries")
  print(eval_judgements)
  print('=' * 70)

  processed_eval_judgements = process_reply(eval_judgements)
  print(f"Processing Judgements: {len(processed_eval_judgements)}")
  print(processed_eval_judgements)