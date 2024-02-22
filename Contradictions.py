import spacy
from transformers import BertTokenizer, BertForMaskedLM
import torch
import string
import numpy as np
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

nlp = spacy.load("en_core_web_md")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('BERT_finetuned_MLM')
model.eval()


def find_antonyms(word):
    antonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.extend([antonym.name() for antonym in lemma.antonyms()])
    return set(antonyms)

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

                top_50_indices = torch.topk(predictions[0, i], 150).indices.tolist()
                top_50_tokens = tokenizer.convert_ids_to_tokens(top_50_indices)

                filtered_tokens = [token for token in top_50_tokens if token not in string.punctuation]

                results[word.text] = filtered_tokens
            except ValueError:
                continue

    return results

def checkContradiction(requirement1, requirement2, glove_embeddings):
    sentence = requirement1
    sentence2 = requirement2
    interested_pos = ['ADJ', 'NOUN', 'PROPN', 'PART']

    predictions = get_bert_predictions(sentence, interested_pos)
    predictions2 = get_bert_predictions(sentence2, interested_pos)

    filtered1 = []
    filtered2 = []

    for word1, prediction_list1 in predictions.items():
        embedding1 = get_glove_embedding(word1.lower(), glove_embeddings)
        for word2 in prediction_list1:
            embedding2 = get_glove_embedding(word2.lower(), glove_embeddings)
            score = cosine_similarity(embedding1, embedding2)
            if score > 0.95:
                filtered1.append(word2)

    for word1, prediction_list1 in predictions2.items():
        embedding1 = get_glove_embedding(word1.lower(), glove_embeddings)
        for word2 in prediction_list1:
            embedding2 = get_glove_embedding(word2.lower(), glove_embeddings)
            score = cosine_similarity(embedding1, embedding2)
            if score > 0.95:
                filtered2.append(word2)

    filtered1 = list(set(filtered1))
    filtered2 = list(set(filtered2))

    print("Filtered words in sentence 1 based on similarity:", filtered1)
    print("Filtered words in sentence 2 based on similarity:", filtered2)

    scores = []
    opposites_dict = {'many': 'only', 'only': 'many'}

    for s in filtered1:
        for s2 in filtered2:
            if s in opposites_dict and s2 == opposites_dict[s]:
                print("Likely contradiction")
                return True
            if s2 in opposites_dict and s == opposites_dict[s2]:
                print("Likely contradiction")
                return True
            e1 = get_glove_embedding(s.lower(), glove_embeddings)
            e2 = get_glove_embedding(s2.lower(), glove_embeddings)
            if cosine_similarity(e1, e2) > 0.70:
                scores.append(cosine_similarity(e1, e2))

            x = lemmatizer.lemmatize(s, pos=wordnet.ADJ)
            y = lemmatizer.lemmatize(s2, pos=wordnet.ADJ)
            z = find_antonyms(s)
            v = find_antonyms(s2)

            if len(z) and len(v) > 0:
                if z.pop() == y:
                    print("contradiction from antonym")
                    return True
                if v.pop() == x:
                    print("contradiction from antonym")
                    return True

    if len(scores) > 0:
        print(sum(scores) / len(scores))
        similar = sum(scores) / len(scores)
        if similar > 0.70:
            if 'not' in filtered1 and 'not' not in filtered2:
                print("Likely contradiction from not")
                return True
            if 'not' in filtered2 and 'not' not in filtered1:
                print("Likely contradiction from not")
                return True
            else:
                print("Likely no contradiction")
                return False
        else:
            print("Likely no contradiction")
            return False
    else:
        print("Likely no contradiction")
        return False

if __name__ == '__main__':

    glove_embeddings = load_glove_embeddings("glove.6B.300d.txt")
    checkContradiction("the system needs to not be available on many platforms",
                 "the system needs to be available on many platforms", glove_embeddings)

    checkContradiction("the system needs to be available on many platforms",
                 "the system needs to be available on only Apple platforms", glove_embeddings)

    checkContradiction("the system needs to be faster than 2 seconds",
                 "the system needs to be slower than 2 seconds", glove_embeddings)
