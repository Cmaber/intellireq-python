import spacy
import re

nlp = spacy.load("en_core_web_lg")

rules = [
    (r"\bmust be invoked\b", "Specifies method invocation"),
    (r"\bexactly [0-9]+(.[0-9]+)?\b", "Specifies an exact value"),
    (r"!\s*[A-Z]|!=" , "Contains programming-like condition or inequality"),
]

def apply_rules(text, rules):
    for pattern, explanation in rules:
        if re.search(pattern, text):
            return True, explanation
    return False, ""

def check_for_technical_language(text):
    doc = nlp(text)
    return any(token.text.lower() in technical_terms for token in doc)

technical_terms = {'method', 'function', 'class', 'interface', 'API', 'protocol', '>', '<', "=", "@", "#", "$", "^", "&", "*"}

def overspecification_checker(requirement):
    rule_flagged, explanation = apply_rules(requirement, rules)

    technical_flagged = check_for_technical_language(requirement)

    if rule_flagged or technical_flagged:
        return True
    else:
        return False

if __name__ == '__main__':
    sc = overspecification_checker("the system needs to be fast")
    print(sc)