import numpy as np
import pandas as pd

def hotEncode(data: pd.core.frame.DataFrame, encodeCol: str) -> pd.core.frame.DataFrame :
  return pd.concat(data, pd.get_dummies(data[encodeCol], prefix = encodeCol, prefix_sep = '_',dummy_na=False)).drop([encodeCol], axis=1)

def better_read_csv(data: pd.core.frame.DataFrame, encodeCol: list, dropNa: bool = 0,indexStart: bool = 0) -> pd.core.frame.DataFrame :
  data = data.dropna(inplace=True) if dropNa else data
  

  return data

print(type(pd.read_csv('titanic.csv')))

