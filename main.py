from fastapi import FastAPI
import spacy
import json
from typing import List
from bs4 import BeautifulSoup
from markdown import markdown

TAG_PROBABILITY_THRESHOLD = 0.5

app = FastAPI()

nlp_text = spacy.load("./spacy_text_model")
nlp_code = spacy.load("./spacy_code_model")

with open('selected_tags.json', 'r') as openfile:
    selected_tags = json.load(openfile)


def preprocess(texts):
    tokens = []
    removal = ['PUNCT', 'SPACE', 'NUM', 'SYM']
    cleaned_texts = []
    for summary in nlp_text.pipe(texts, disable=["transformer", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]):
        question_tokens = []
        for token in summary:
            if token.pos_ not in removal and token.is_alpha and len(question_tokens) < 512:
                question_tokens.append(token.lower_)
        cleaned_texts.append(" ".join(question_tokens))
    return cleaned_texts


def get_text_and_code(body):
    html = markdown(body)
    bs = BeautifulSoup(html)
    codes = bs.findAll('code')
    code = '\n'.join([x.text for x in codes])
    for x in codes:
        x.decompose()
    text = '\n'.join(bs.findAll(text=True))
    return text, code


@app.post("/infer_tags")
async def infer_tags(questions: List[str]):
    results = []
    texts = []
    codes = []
    for question in questions:
        text, code = get_text_and_code(question)
        texts.append(text)
        codes.append(code)

    texts_preprocessed = preprocess(texts)
    codes_preprocessed = preprocess(codes)

    pred_text = []
    pred_code = []
    for summary in nlp_text.pipe(texts_preprocessed):
        if summary.text != '':
            pred_text.append(summary.cats)
        else:
            pred_text.append(dict.fromkeys(selected_tags, 0))

    for summary in nlp_code.pipe(codes_preprocessed):
        if summary.text != '':
            pred_code.append(summary.cats)
        else:
            pred_code.append(dict.fromkeys(selected_tags, 0))

    text_tags = [[x for x in selected_tags if y[x] > TAG_PROBABILITY_THRESHOLD]
                 for y in pred_text]
    code_tags = [[x for x in selected_tags if y[x] > TAG_PROBABILITY_THRESHOLD]
                 for y in pred_code]
    union_tags = []
    for i in range(len(text_tags)):
        union_tags.append(list(set(text_tags[i]) | set(code_tags[i])))
    return union_tags
