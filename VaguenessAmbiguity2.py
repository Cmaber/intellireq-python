import spacy
from transformers import BertTokenizer, BertForMaskedLM
import torch
import string
import numpy as np

nlp = spacy.load("en_core_web_lg")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')
model.eval()


def recombine_hyphenated_words(tokens):
    i = 0
    while i < len(tokens) - 2:
        if tokens[i+1] == '-' and tokens[i].isalpha() and tokens[i+2].isalpha():
            tokens[i] = tokens[i] + tokens[i+1] + tokens[i+2]
            del tokens[i+1:i+3]
        else:
            i += 1
    return tokens

def combine_subwords(tokens):
    combined = []
    for token in tokens:
        if token.startswith('##'):
            combined[-1] += token[2:]
        else:
            combined.append(token)
    return combined

def load_glove_embeddings(path):
    embeddings_dict = {}
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

def get_glove_embedding(word, embeddings_dict):
    return embeddings_dict.get(word, np.zeros(300))

def cosine_similarity(vec1, vec2):
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_bert_predictions(sentence, interested_pos):
    doc = nlp(sentence)
    tokens = tokenizer.tokenize(sentence)
    tokens = recombine_hyphenated_words(tokens)
    tokens = combine_subwords(tokens)

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    results = {}
    for word in doc:
        if word.pos_ in interested_pos:
            try:
                i = tokens.index(word.text.lower())
                masked_ids = token_ids.copy()
                masked_ids[i] = tokenizer.mask_token_id
                input_ids = torch.tensor([masked_ids])

                with torch.no_grad():
                    outputs = model(input_ids)
                    predictions = outputs[0]

                top_50_indices = torch.topk(predictions[0, i], 10).indices.tolist()
                top_50_tokens = tokenizer.convert_ids_to_tokens(top_50_indices)

                filtered_tokens = [token for token in top_50_tokens if token not in string.punctuation]

                results[word.text] = filtered_tokens
            except ValueError:
                continue

    return results

def checkReq(requirement1, glove_embeddings):
    if requirement1 == '':
        return False
    sentence = requirement1
    interested_pos = ['ADJ', 'NOUN', 'PROPN']

    predictions = get_bert_predictions(sentence, interested_pos)

    filtered1 = []

    for word1, prediction_list1 in predictions.items():
        embedding1 = get_glove_embedding(word1.lower(), glove_embeddings)
        for word2 in prediction_list1:
            embedding2 = get_glove_embedding(word2.lower(), glove_embeddings)
            score = cosine_similarity(embedding1, embedding2)
            if score > 0.70:
                filtered1.append(word2)

    filtered1 = list(set(filtered1))

    print("Filtered words in sentence 1 based on similarity:", filtered1)

    if len(filtered1) > 0:
        print("Not Vague")
        return False
    else:
        print("Vague")
        return True


if __name__ == '__main__':

    glove_embeddings = load_glove_embeddings("glove.6B.300d.txt")

    checkReq("the system needs to be faster than 2 seconds", glove_embeddings)
    checkReq("the system needs to be slower than 2 seconds", glove_embeddings)
    checkReq("the system needs to be fast and smart", glove_embeddings)
    checkReq("the system needs to be smart", glove_embeddings)
    checkReq("the system needs to be available on multiple platforms", glove_embeddings)
    checkReq("the system needs to be quicker than 2 seconds", glove_embeddings)
    checkReq("the system shall be fast", glove_embeddings)
    checkReq("the system shall be responsive", glove_embeddings)
    checkReq("the system shall be innovative", glove_embeddings)
    checkReq("the system shall be flexible", glove_embeddings)
    checkReq("the system shall be user-friendly", glove_embeddings)
    checkReq("the system shall be secure", glove_embeddings)
    checkReq("the system shall be scalable", glove_embeddings)
    checkReq("the system shall be reliable", glove_embeddings)
    checkReq("the system shall be accessible", glove_embeddings)
    checkReq("the system shall be efficient", glove_embeddings)
    checkReq("the system shall be intuitive", glove_embeddings)
