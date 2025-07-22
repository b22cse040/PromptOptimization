from src.asyncio_executor import AsyncioExecutor
from src.opro import run_opro

if __name__ == '__main__':
  # Fixed Parameters
  filepath = "Dataset/dataset/df_model_M11.csv"
  eval_llm_name = "deepseek/deepseek-r1-0528:free"
  optim_llm_name = "mistralai/mistral-small-3.2-24b-instruct:free"
  k = 5
  num_epochs = 5
  num_parallel_runs = 5

  executor = AsyncioExecutor()

  futures = [
    executor.submit(run_opro, filepath, eval_llm_name, optim_llm_name, k,
                    num_epochs, i)
    for i in range(num_parallel_runs)
  ]

  results = []
  for i, fut in enumerate(futures):
    try:
      result = fut.result()
      results.append(result)
      print(f"Finished run {i + 1}/{num_parallel_runs}")

    except Exception as e:
      print(f"Run {i + 1}/{num_parallel_runs} failed: {e}")
      results.append(None)

  executor.shutdown()