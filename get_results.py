from glob import glob
results = glob("results/*/valid_acc.log")
results_1fold = [pa for pa in results if 'trope_split' not in pa]
from collections import defaultdict
results_5fold = defaultdict(list)
for result in results:
    if 'trope_split' not in result:
        continue
    k = ''.join(result.split("trope_split")[0].split("_")[7:])
    results_5fold[k].append(result)


for result in results_1fold:
    print(result)
    lines = open(result).read().split("\n")
    test_lines = [line for line in lines if "TEST" in line]
    print(test_lines[-1])
    
for k, v in results_5fold.items():
    print(k)
    scores = []
    for result in v:
        lines = open(result).read().split("\n")
        test_lines = [line for line in lines if "TEST" in line]
        scores.append(float((test_lines[-1].split(" ")[-1])))
    print(scores, sum(scores)/5)
        