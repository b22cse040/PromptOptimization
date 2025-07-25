import requests
import time
import random
import os
from dotenv import load_dotenv

load_dotenv()

def call_openrouter(
    prompt: str,
    llm_name: str = 'meta-llama/llama-3-8b-instruct',
    retries: int = 5,
    backoff: float = 1.0,
    jitter: float = 0.25,
) -> str:
  """
  Calls the OpenRouter API with a given prompt and handles retries with
  exponential backoff and random jitter.

  This fn is designed to be resilient to transient network issues and temporary
  API unavailability.

  Args:
     prompt (str): The user's prompt to send to the LLM.
     llm_name (str): The model identifier on OpenROuter (e.g., 'meta-llama/llama-3-8b-instruct').
                        Defaults to a fast and capable model.
     retries (int): The maximum number of times to retry the request if it fails.
     backoff (float): The base time in seconds for the backoff delay. The delay
                              increases exponentially with each retry.

  Returns:
    str: The response from the OpenRouter API. (Note: Although some prompts would ask for
    JSON output, use clean_response to achieve JSON output.)

  Raises:
    ValueError: If the OPENROUTER_API_KEY environment variable is not set.
    Exception: If the request fails after all retry attempts.
  """
  # API Key and config
  api_key = os.getenv('OPENROUTER_API_KEY')
  if not api_key:
    raise ValueError('OPENROUTER_API_KEY environment variable is not set.')

  # Req Details
  url = "https://openrouter.ai/api/v1/chat/completions"
  headers = {
    "Authorization" : f"Bearer {api_key}",
    "Content-Type" : "application/json",
  }
  data = {
    "model" : llm_name,
    "messages" : [{"role" : "user", "content" : prompt}]
  }

  # Retry loop with exponential backoff and jitter
  for attempt in range(retries):
    print(f"Attempting to call OpenRouter API.. (Attempt {attempt + 1}/{retries})")
    try:
      response = requests.post(url, headers=headers, json=data, timeout=90)

      # Check for HTTP errors (e.g., 4XX client errors, 5XX server errors).
      # If an error status code is returned, this will raise an HTTPError.
      response.raise_for_status()

      # if there was successful, print and return the response.
      print("API call successful.")
      return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
      # This block catches network-related errors like connection timeouts,
      # DNS failures, etc., as well as HTTPError from raise_for_status().
      print(f"Request failed: {e}")

      # If last try, raise the final exception
      if attempt + 1 == retries:
        print("All retries failed.")
        raise Exception(f"API req faile after {retries} retries.") from e

      # Calculate wait time
      backoff_time = backoff * (2 ** attempt)
      # Jitter: adds a small, random delay to prevent thundering herd problem.
      # This is crucial when multiple threads/processes are making requests.
      jitter = random.uniform(0, backoff_time * jitter)  # Add up to 25% of the backoff time as jitter
      sleep_duration = backoff_time + jitter
      print(f"Retrying in {sleep_duration:.2f} seconds...")
      time.sleep(sleep_duration)

  return ""

if __name__ == "__main__":
  prompt = "What is the point of Life?"
  response = call_openrouter(prompt)
  print(response)