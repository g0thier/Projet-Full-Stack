# https://www.analyticsvidhya.com/blog/2021/05/how-to-use-progress-bars-in-python/

from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import threading
from sentence_transformers import SentenceTransformer, util
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv("export/temps/dataset.csv", delimiter=',', on_bad_lines='skip')
model = SentenceTransformer('all-MiniLM-L6-v2')
print(len(dataset))
dataset = dataset.drop_duplicates()
print(len(dataset))
# Convert string Genre to list
dataset = dataset.rename(columns={"genres": "genre"})
dataset["genre"] = [ str(x).replace(' ', '') for x in dataset["genre"][:] ]
dataset["genres"] = [ x.split(",") for x in dataset["genre"][:] ]
model = SentenceTransformer('all-MiniLM-L6-v2')

dataset['genre'] = ''
dataset['themaScore'] = 0.0

for i in tqdm(range( len(dataset) )):
    bestScore = 0.0
    bestGenre = ''
    for j in range(len(dataset.iloc[i]['genres'])):
        result = util.cos_sim(model.encode(str(dataset.iloc[i]['genres'][j])), model.encode(str(dataset.iloc[i]['resume']))).tolist()[0][0]
        if result > bestScore:
            bestScore = result
            bestGenre = dataset.iloc[i]['genres'][j]
    dataset['themaScore'][i] = bestScore
    dataset['genre'][i] = bestGenre

dataset = dataset.drop(columns=['genres'])

dataset.to_csv(r'temps/datasetWithPonderation.csv', index = False, header=True)

