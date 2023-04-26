import pandas as pd

def read_ui_log_as_dataframe(log_path):
  return pd.read_csv(log_path, sep=",")#, index_col=0)