import json 
import numpy as np


def majority_vote(preds):
    for label in ["same", "minor", "no"]:
        if preds.count(label) >= 2:
            return label 
    return None 

data1 = json.load(open("results_yuning.json"))

data2 = json.load(open("results_yiqing.json"))
data2 = sorted(data2, key=lambda x:x["idx"])

data3 = json.load(open("results_alex.json"))
for x in data3:
    x["answer"] = [x["answer"], ""]

model_pred2human = {k:{} for k in ["yes","no"]}
for k in model_pred2human:
    model_pred2human[k] = {"same":0, "minor":0, "no":0}
for x1, x2, x3 in zip(data1, data2, data3):
    model_pred = "yes" if x1["llm_functionality_check"]["answer"] in ["same", "yes"] else "no"
    # human_pred = x2["answer"][0] if x2["idx"]!=1123 else x1["answer"][0]
    human_pred = majority_vote([x1["answer"][0], x2["answer"][0], x3["answer"][0]])
    assert human_pred is not None 
    
    model_pred2human[model_pred][human_pred] += 1
    
    if not (x1["answer"][0] == x2["answer"][0] and x2["answer"][0] == x3["answer"][0]):
        print(x1["function_name"], x1["answer"][0], x2["answer"][0], x3["answer"][0])

print(model_pred2human)




def fleiss_kappa(matrix):
    N, k = matrix.shape  # N: number of items, k: number of classes
    n = np.sum(matrix[0])  # Total number of ratings per item (assumed constant)

    p = np.sum(matrix, axis=0) / (N * n)
    P = (np.sum(matrix**2, axis=1) - n) / (n * (n - 1))

    P_bar = np.mean(P)
    P_e_bar = np.sum(p**2)

    # Fleiss' kappa
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    return kappa

annotations = [[2, 1]] * 0 + [[1, 2]] * 1 + [[3, 0]] * 14 + [[0, 3]] * 5
annotations = np.array(annotations)

# Calculate Fleiss' kappa
kappa_score = fleiss_kappa(annotations)
print(f"Fleiss' kappa: {kappa_score:.4f}")