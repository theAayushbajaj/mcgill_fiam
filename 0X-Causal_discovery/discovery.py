## Requires - pip install avici

import avici
import pandas as pd
import numpy as np

import pgmpy

model = avici.load_pretrained(download="scm-v0")

df = pd.read_pickle('X_dataset.pkl')
df = df.dropna()
x = df.to_numpy()

g_prob = model(x=x)

# perform d seperation