import polars as pl
from typing import Dict

def random_subsample(file_path: str, n_texts : int = 5, n_summaries: int = 1) -> pl.DataFrame:
  # Note that there are 11 human_summary for each piece of text (100 total)
  ## We will subsample n_texts texts randomly, and each will have n_summaries human_summary
  ## Attached to it.

  df = pl.read_csv(file_path)
  unique_texts = df.select("text").unique()

  sampled_texts = unique_texts.sample(n_texts)

  df_filtered = df.join(sampled_texts, on="text")

  # For each piece of text, pick n_summaries random human_summary
  df_grouped = (
    df_filtered
    .group_by("text")
    .agg(pl.all().sample(n_summaries))
    .explode(pl.exclude("text"))
  )

  df_subsampled = df_grouped.with_row_index(name="NEW_ID", offset=1)

  return df_subsampled

def create_sample_points(file_path: str) -> list[Dict[str, str]]:
  """
  The dataframe is subsampled and processed for _OPTIM_META_PROMPT
  """
  df_subsampled = random_subsample(file_path)

  ## Printing for sanity check.
  # for row in df_subsampled.iter_rows(named=True):
  #   for col in df_subsampled.columns:
  #     print(f"{col}: {row[col]}")
  #   print('\n\n')

  sample_points = []

  for row in df_subsampled.iter_rows(named=True):
    sample_points.append({
      "text": row["text"],
      "human_summaries": row["human_summaries"],
      "machine_summary": row["machine_summary"],
      "ground_fluency": row["fluency"],
      "ground_coherence": row["coherence"],
      "ground_consistency": row["consistency"],
      "ground_relevance": row["relevance"],
    })

  return sample_points


if __name__ == "__main__":
  sample_points = create_sample_points("dataset/df_model_M11.csv")
  print(f"Length of sample_points: {len(sample_points)}\n")
  # print(sample_points)