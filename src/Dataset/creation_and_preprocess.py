import pandas as pd

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
  df = df.explode("expert_annotations")

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

if __name__ == '__main__':
  # original_path = "dataset/model_annotations.aligned.paired.jsonl"
  utf8_path = "dataset/model_annotations.utf8.jsonl"
  # convert_encoding(original_path, utf8_path)

  df = load_dataset(file_path=utf8_path, encoding='utf-8')
  cleaned_df = clean_dataset(df)
  head = cleaned_df.head()

  cleaned_df = fix_encoding(cleaned_df)

  for idx, row in head.iterrows():
    for col in cleaned_df.columns:
      print(f"{col} : {row[col]}")
    print('\n\n')

  output_path = "dataset/cleaned_df.csv"
  cleaned_df.to_csv(output_path, encoding='utf-8', index=False)
  print(f"Cleaned dataset saved to {output_path}")
