"""
fine tuningしたモデルを読み込んで推論
"""
import argparse
import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

load_dotenv()
MODEL_DIR = Path(os.environ.get('MODEL_DIR'))
PAPER_DIR = Path(os.environ.get('PAPER_DIR'))
RESULT_DIR = Path(os.environ.get('RESULT_DIR'))


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

                if ent == 'I':
                    try:
                        if start == prev_end:
                            chemical += re.sub('^##', '', word)
                        else:
                            chemical += ' ' + word
                    except UnboundLocalError:
                        ent = 'B'  # Iから始まっていた場合はそれをBと見なす
                if ent == 'B':
                    if chemical != '':
                        chemicals.add(chemical)
                        chemical = ''
                    chemical += word

                prev_end = end

            if chemical != '':
                chemicals.add(chemical)
    return list(chemicals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bc5cdr-chem/2021_03_25_02_11_06/')
    args = parser.parse_args()
    load_dir = MODEL_DIR / args.model

    tokenizer = AutoTokenizer.from_pretrained(load_dir)
    model = AutoModelForTokenClassification.from_pretrained(load_dir)
    ner = pipeline('ner', model=model, tokenizer=tokenizer)

    pmcid2chemicals = {}
    error_pmcids = []
    file_path_list = list(PAPER_DIR.glob('*.json'))
    for file_path in tqdm(file_path_list):
        try:
            chemicals = get_chemicals(file_path)
            pmcid = file_path.stem
            pmcid2chemicals[pmcid] = chemicals
        except:
            error_pmcids.append(file_path.stem)

    with open(RESULT_DIR / 'pmcid2chemicals.json', 'w') as f:
        json.dump(pmcid2chemicals, f, ensure_ascii=False)
    with open(RESULT_DIR / 'error_pmcids.json', 'w') as f:
        json.dump(error_pmcids, f, ensure_ascii=False)
