import numpy as np
import VaguenessAmbiguity2
import Contradictions
import similarities
#import Intelligibility
#import Overspecification
from flask import Flask, request

reqErrors = {"CONTRADICTIONS": [], "VAGUENESSAMBIGUITY": [], "SIMILARITIES": [], "UNINTELLIGIBILITY": [], "OVERSPECIFICATION": []}

app = Flask(__name__)


def load_glove_embeddings(path):
    embeddings_dict = {}
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

print("Loading GloVe Embeddings\n")
glove_embeddings = load_glove_embeddings("glove.6B.300d.txt")
print("GloVe Embeddings Completed\n")


@app.route('/main', methods=['POST'])
def main():
    reqErrors["CONTRADICTIONS"].clear()
    reqErrors["VAGUENESSAMBIGUITY"].clear()
    reqErrors["SIMILARITIES"].clear()
    reqErrors["OVERSPECIFICATION"].clear()
    reqErrors["UNINTELLIGIBILITY"].clear()
    data = request.json
    textLst = data['req']
    VaguenessAmbiguity(textLst)
    contradiction(textLst)
    similarity(textLst)
    #Unintelligible(textLst)
    #overspecification(textLst)

    return reqErrors

def Unintelligible(requirements):
    if len(requirements) > 0:
        indices_set = set()
        for i in range(len(requirements)):
            #check = Intelligibility.checkReq(requirements[i])
            check = False
            if check:
                indices_set.add(i)
        reqErrors["UNINTELLIGIBILITY"] = list(indices_set)


def VaguenessAmbiguity(requirements):
    if len(requirements) > 0:
        indices_set = set()
        for i in range(len(requirements)):
            check = VaguenessAmbiguity2.checkReq(requirements[i], glove_embeddings)
            if check:
                indices_set.add(i)

        reqErrors["VAGUENESSAMBIGUITY"] = list(indices_set)


def contradiction(requirements):
    if len(requirements) > 1:
        contradictions_set = set()
        for i in range(len(requirements)):
            for j in range(i + 1, len(requirements)):
                check = Contradictions.checkContradiction(requirements[i], requirements[j], glove_embeddings)
                if check:
                    lst_tuple = tuple(sorted([i, j]))
                    if lst_tuple not in contradictions_set:
                        contradictions_set.add(lst_tuple)
                        reqErrors["CONTRADICTIONS"].append(list(lst_tuple))


def similarity(requirements):
    if len(requirements) > 1:
        similar_set = set()
        for i in range(len(requirements)):
            for j in range(i + 1, len(requirements)):
                check = similarities.similarCheck(requirements[i], requirements[j], glove_embeddings)
                if check:
                    lst_tuple = tuple(sorted([i, j]))
                    if lst_tuple not in similar_set:
                        similar_set.add(lst_tuple)
                        reqErrors["SIMILARITIES"].append(list(lst_tuple))

def overspecification(requirements):
    if len(requirements) > 0:
        indices_set = set()
        for i in range(len(requirements)):
            #check = Overspecification.overspecification_checker(requirements[i])
            check = False
            if check:
                indices_set.add(i)

        reqErrors["OVERSPECIFICATION"] = list(indices_set)


if __name__ == '__main__':
    app.run(debug=False)