import sys
import json

"""
Tiny helper script for scoring MATH. Aggregates at the document, not subject, level.

Pass in a comma-separated string of paths to files that contain MATH score results.
"""

path = sys.argv[1]

paths = [p for p in path.split(",") if len(p) > 0]
print("Results filepaths received:" paths)

accs = []
weights = []

for path in paths:
    with open(path) as f:
        out = json.load(f)


    for task in out['results']:
        if "minerva_math" in task and "easy" not in task:
            this = out['results'][task]['acc']
            accs.append(this)
            weights.append(len(out['cache'][task]))

            print(task, this)

acc = sum(x*y for x,y in zip(accs, weights))/sum(weights)
print('total: ', acc)