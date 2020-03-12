import time
import random
import csv
from pandas import DataFrame

percent = 0.05

location = "/media/docleary/Storage/Documents/Datasets/AVA_dataset/"


d = {}
with open(location + "AVA.txt") as f:
    for row, line in enumerate(f):
        name = line.split()[1].split(",")[0]
        d[name] = list(map(str, line.split()))



amount = int(len(d) * percent)

d_new = {}
for i in range(0, amount):
    choice = random.choice(list(d.keys()))
    d_new[choice] = d.pop(choice, None)
    # print(d_new[choice])
    # time.sleep(10)

df = DataFrame(d_new)
df = df.T
# df.drop('c1', 1, inplace=True)

df.to_csv(location + "val_ava.csv", index=False)

d_newnew = {}
for i in range(0, amount):
    choice = random.choice(list(d.keys()))
    d_newnew[choice] = d.pop(choice, None)

dfnew = DataFrame(d_newnew)
dfnew = dfnew.T

dfnew.to_csv(location + "test_ava.csv", index=False)

dfnewnew = DataFrame(d)
dfnewnew = dfnewnew.T

dfnewnew.to_csv(location + "train_ava.csv", index=False)



