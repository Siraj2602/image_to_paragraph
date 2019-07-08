import json


file = open('train_split.json')
k = file.read()
x = json.loads(k)

x = x[5000:8000]

file.close()
file = open('train_split_3000.json','w')

json.dump(x,file)

file.close()

