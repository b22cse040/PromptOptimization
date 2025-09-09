import pandas as pd
import ast
from typing import Tuple

# def convert_encoding(input_path: str, output_path: str, from_encoding: str = 'latin-1', to_encoding: str = 'utf-8'):
#   with open(input_path, 'r', encoding=from_encoding) as infile:
#     content = infile.read()
#
#
#   with open(output_path, 'w', encoding=to_encoding) as outfile:
#     outfile.write(content)
#
#   print(f"File saved in UTF-8 encoding at: {output_path}")

def load_dataset(
    file_path: str = "dataset/model_annotations.aligned.paired.jsonl",
    encoding = 'latin-1') -> pd.DataFrame:

  df = pd.read_json(file_path, lines=True, encoding=encoding)
  return df

def fix_encoding(df: pd.DataFrame) -> pd.DataFrame:
  for idx, row in df.iterrows():
    for cols in df.columns:
      text = row[cols]

      if not isinstance(text, str):
        text = str(text)

      text = text.replace('\u00A0', ' ')  # NBSP → space
      text = text.replace('\u200B', '')  # zero-width space → remove
      text = text.replace('\u2013', '-')  # en-dash → dash
      text = text.replace('\u2014', '-')  # em-dash → dash
      text = text.replace('\u2018', "'").replace('\u2019',
                                                 "'")  # curly apostrophes
      text = text.replace('\u201C', '"').replace('\u201D', '"')  # curly quotes

      df.at[idx, cols] = text
  return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
  df = df.explode("expert_annotation")

  df = df.rename(columns={
    "decoded" : "machine_summary",
    "references" : "human_summaries",
    "expert_annotations" : "expert_annotation"
  })

  df = df[[
    "machine_summary",
    "human_summaries",
    "model_id",
    "text",
    "expert_annotation",
  ]]

  df = df.reset_index(drop=True)
  df.insert(0, "ID", df.index.astype(int))

  return df

def subsample_by_model_id(df: pd.DataFrame, model_id: str = "M11"):
  # df = pd.read_json(df, lines=True, encoding="utf-8")
  df = df[df.model_id == model_id].reset_index(drop=True)

  if df["expert_annotation"].dtype == object:
    df["expert_annotation"] = df["expert_annotation"].apply(
      lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith(
      "{") else x
    )

  annotation_keys = ["fluency", "coherence", "consistency", "relevance"]
  for key in annotation_keys:
    df[key] = df["expert_annotation"].apply(lambda x: x.get(key, None) if isinstance(x, dict) else None)

  df = df.groupby(["text", "machine_summary"], group_keys=False).apply(
    lambda x: x.sample(1, random_state=42)).reset_index(drop=True)

  result_df = df[[
    "machine_summary",
    "human_summaries",
    "text",
    "fluency",
    "coherence",
    "consistency",
    "relevance",
    "model_id",
  ]].copy()

  return result_df

def subsample_dataset(df: pd.DataFrame, split: str = "combined", num_points: int = 40) -> pd.DataFrame:
  all_samples = []
  model_ids = df["model_id"].unique()

  for model_id in model_ids:
    df_model = subsample_by_model_id(df, model_id)
    sampled = df_model.sample(n=min(num_points, len(df_model)), random_state=42)
    all_samples.append(sampled)

  final_df = pd.concat(all_samples).reset_index(drop=True)
  return final_df

def split_dataset(df: pd.DataFrame, test_size : float = 0.3, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
  train_parts, test_parts = [], []

  model_ids = df["model_id"].unique()
  print(len(model_ids))
  for model_id in model_ids:
    df_model = df[df["model_id"] == model_id]

    df_model = df_model.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n_test_samples = int(len(df_model) * test_size)
    test_df_model = df_model[:n_test_samples]
    train_df_model = df_model[n_test_samples:]
    train_parts.append(train_df_model)
    test_parts.append(test_df_model)

  train_df = pd.concat(train_parts).reset_index(drop=True)
  test_df = pd.concat(test_parts).reset_index(drop=True)
  return train_df, test_df

if __name__ == '__main__':
  df = pd.read_csv("dataset/cleaned_df.csv")
  # cleaned_df = clean_dataset(df)

  subsampled_df = subsample_dataset(df, num_points=40)
  train_df, test_df = split_dataset(subsampled_df, test_size=0.75)

  train_df.to_parquet("dataset/cleaned_train_df.parquet", index=False)
  test_df.to_parquet("dataset/cleaned_test_df.parquet", index=False)

  train_df.to_csv("dataset/train_df.csv", index=False)
  test_df.to_csv("dataset/test_df.csv", index=False)

  print("Train shape:", train_df.shape)
  print("Test shape:", test_df.shape)
  print("\nTrain head:\n", train_df.head(15))
  print("\nTest head:\n", test_df.head(15))
  print("Saved train_df.csv and test_df.csv")
