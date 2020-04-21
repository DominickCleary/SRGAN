import csv
import os.path
import pandas as pd
import time

lines = list()
df = pd.read_csv("/media/docleary/Storage/Documents/Datasets/AVA_dataset/train_ava.csv") 

# print(df.head())
# time.sleep(10)
for index, row in df.iterrows():
    idname = row['id']
    if(not os.path.isfile("/media/docleary/Storage/Documents/Datasets/AVA_dataset/images/" + str(idname) + ".jpg")):
        df = df[df.id != idname]

df.to_csv("/media/docleary/Storage/Documents/Datasets/AVA_dataset/train_ava_new.csv", index=False)
