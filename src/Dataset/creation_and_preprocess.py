import pandas as pd

def load_dataset(
    file_path: str = "dataset/model_annotations.aligned.paired.jsonl",
    encoding = 'latin-1') -> pd.DataFrame:

  df = pd.read_json(file_path, lines=True, encoding=encoding)
  head = df.head()

  for idx, row in head.iterrows():
    for col in df.columns:
      print(f"{col} : {row[col]}")
    print('\n\n')
  return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
  pass

if __name__ == '__main__':
  df = load_dataset()
