import spacy
from transformers import BertTokenizer, BertModel
import torch

nlp_spacy = spacy.load("en_core_web_lg")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')


def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state


def checkReq(sentence):
    if (sentence == ''):
        return False
    # Syntactic analysis
    doc = nlp_spacy(sentence)

    embeddings = get_bert_embedding(sentence).squeeze(0)

    avg_similarity = 0
    count = 0

    for token in doc[:-1]:  # Exclude the last token to avoid index error
        # Only compare syntactically related words (e.g., subject-verb, verb-object)
        head_embedding = embeddings[token.head.i]
        child_embedding = embeddings[token.i]
        similarity = torch.cosine_similarity(head_embedding.unsqueeze(0), child_embedding.unsqueeze(0)).item()

        avg_similarity += similarity
        count += 1
    if (count == 0):
        return False
    avg_similarity /= count

    # Threshold might need to be adjusted
    if avg_similarity < 0.6:  # Arbitrary threshold for demonstration
        print(avg_similarity)
        print("unintelligible")
        return True  # Sentence might be nonsensical
    else:
        print(avg_similarity)
        print("intelligible")
        return False


if __name__ == '__main__':
    checkReq("the cool yes me")
