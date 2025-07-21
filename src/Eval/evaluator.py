import requests, os, json5, re
# import numpy as np
from dotenv import load_dotenv

from src.Dataset.random_subsample import create_sample_points
from src.Eval.e_prompt import create_evaluator_prompt
from src.Metrics.get_metrics import calculate_metrics
from src.Optim.optimizer import call_optimizer_llm
from src.TopK_Heap.top_k import TopKHeap
from typing import Dict

load_dotenv()

def clean_response(reply: str) -> dict:
  try:
    cleaned_reply = re.sub(r'^```json|```$', '', reply,
                           flags=re.MULTILINE).strip()

    # Fix unescaped backslashes before parsing
    cleaned_reply = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned_reply)

    data = json5.loads(cleaned_reply)
    return data
  # except json.JSONDecodeError as e:
  #   print(f"Error decoding JSON: {e}")
  #   return {}
  except Exception as e:
    print(f"Error decoding JSON: {e}")
    return {}

def process_reply(eval_reply: dict, heap: TopKHeap, metrics: dict) -> dict:
  """
  Processes the evaluator reply. Pushes the instruction, metrics, and recommendation
  into the TopKHeap, and returns the processed dict.
  :param eval_reply: Output from evaluator LLM containing 'instruction' and 'recommendation'
  :param heap: TopKHeap object to maintain top-k prompts
  :param metrics: Dictionary of classification reports per metric (fluency, coherence, etc.)
  :return: Processed dict pushed to heap
  """

  processed = {
    "instruction": eval_reply["instruction"],
    "metrics" : metrics,
    "recommendation": eval_reply["recommendation"],
  }

  heap.push(processed)
  # print(processed)
  return processed

def call_evaluator_llm(sample_points: list[Dict[str, str]], optim_llm_response: dict, eval_llm_name: str) -> dict:
  """
  optim_llm_response: the response generated from OPTIM_LLM containing text, human_summary
  and machine_summary
  eval_llm_name: name of the evaluator llm to use
  """
  evaluator_prompt = create_evaluator_prompt(sample_points, optim_llm_response)

  evaluator_response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
      "Content-Type": "application/json",
      "Authorization": "Bearer " + os.getenv("OPENROUTER_API_KEY")
    },
    data = json5.dumps({
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
  filepath = "../Dataset/dataset/df_model_M11.csv"

  sample_points = create_sample_points(filepath)
  print(f"Sample points created: {len(sample_points)}")

  top_k_prompts = TopKHeap(3)
  optim_summaries = call_optimizer_llm(sample_points, top_k_prompts=top_k_prompts,optim_llm_name=optim_llm_name)
  print(optim_summaries)
  print(f"OPTIM_LLM: Generated summary scores")

  metrics = calculate_metrics(sample_points, optim_summaries)
  print(metrics)
  print(f"Calculated metrics")

  eval_judgements = call_evaluator_llm(sample_points, optim_summaries, eval_llm_name)
  print(f"EVAL_LLM: Generating recommendations")
  print(eval_judgements)
  print('=' * 70)

  processed_eval_judgements = process_reply(eval_judgements, top_k_prompts, metrics)
  print(f"Processing Judgements: {len(processed_eval_judgements)}")
  print(processed_eval_judgements)