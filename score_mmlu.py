import sys
import json

path = sys.argv[1]

with open(path) as f:
    out = json.load(f)

accs = []
weights = []

for task in out['results']:
    if "minerva-hendrycksTest" in task:
        this = out['results'][task]['acc']
        accs.append(this)
        weights.append(len(out['cache'][task]))

        print(task, this)

acc = sum(x*y for x,y in zip(accs, weights))/sum(weights)
print('total: ', acc)
