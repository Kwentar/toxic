import pandas as pd
import numpy as np

my = pd.read_csv("sumbs/baseline.csv")
blend1 = pd.read_csv('sumbs/blend_sub.csv')
gru_pool = pd.read_csv('sumbs/submission9819.csv')
blend2 = pd.read_csv('sumbs/hight_of_blending.csv')

b1 = my.copy()
col = my.columns

col = col.tolist()
col.remove('id')

for i in col:
    b1[i] = (my[i] + blend1[i] + gru_pool[i] + blend2[i]) / 4

b1.to_csv('my_blend.csv', index=False)
