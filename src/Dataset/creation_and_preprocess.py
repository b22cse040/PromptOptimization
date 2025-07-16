import polars as pl

def load_dataset(
    link: str = 'hf://datasets/mteb/summeval/data/test-00000-of-00001-35901af5f6649399.parquet') -> pl.DataFrame:
  df = pl.read_parquet(link)
  return df


def preprocess_dataset(df: pl.DataFrame) -> pl.DataFrame:
  df_exploded = df.explode("human_summaries")
  df_exploded = df_exploded.rename({"human_summaries": "human_summary"})
  filter_df = df_exploded.with_row_index(name="ID", offset=1)
  filter_df = filter_df.select(["ID", "text", "human_summary"])

  return filter_df

if __name__ == '__main__':
  df = load_dataset()
  filter_df = preprocess_dataset(df)
  # print(filter_df.head())
  print(filter_df.shape)
  print(filter_df.describe())
  filter_df.write_csv("summary_pairs.csv")
