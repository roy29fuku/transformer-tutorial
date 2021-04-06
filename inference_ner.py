"""
fine tuningしたモデルを読み込んで推論
"""
import argparse
import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

load_dotenv()


def get_chemicals(file_path):
    with open(file_path) as f:
        data = json.load(f)

    chemicals = set()
    for abstract in data['content']['abstract']:
        if abstract['heading'] == '':
            sentences = abstract['text']
        else:
            sentences = [abstract['heading']] + abstract['text']

        for sent in sentences:
            chemical = ''
            for res in ner(sent):
                ent = res['entity']
                word = res['word']
                start = res['start']
                end = res['end']

                if ent == 'B':
                    if chemical != '':
                        chemicals.add(chemical)
                        chemical = ''
                    chemical += word
                if ent == 'I':
                    if start == prev_end:
                        chemical += re.sub('^##', '', word)
                    else:
                        chemical += ' ' + word

                prev_end = end

            if chemical != '':
                chemicals.add(chemical)
    return list(chemicals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bc5cdr-chem/2021_03_25_02_11_06/')
    args = parser.parse_args()
    load_dir = Path(os.getenv('MODEL_DIR')) / args.model

    tokenizer = AutoTokenizer.from_pretrained(load_dir)
    model = AutoModelForTokenClassification.from_pretrained(load_dir)
    ner = pipeline('ner', model=model, tokenizer=tokenizer)

    pmcid2chemicals = {}
    paper_dir = Path(os.environ.get('PAPER_DIR'))
    file_paths = list(paper_dir.glob('*.json'))
    print(len(file_paths))
    for file_path in file_paths[:5]:
        chemicals = get_chemicals(file_path)
        pmcid = file_path.stem
        pmcid2chemicals[pmcid] = chemicals

    # for file_path in paper_dir.glob('*.json'):
    #     chemicals = get_chemicals(file_path)
    #     pmcid = file_path.stem
    #     pmcid2chemicals[pmcid] = chemicals

    with open('pmcid2chemicals.json', 'w') as f:
        json.dump(pmcid2chemicals, f, indent=4, ensure_ascii=False)
