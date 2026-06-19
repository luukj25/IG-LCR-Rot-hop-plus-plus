"""
Modified preprocessing script for IG-LCR-Rot-hop++.

Identical to main_preprocess.py but also stores token strings in each .pt file,
which are needed by ig_attribution.py to build the global attribution scores
and bias dictionary.

Usage:
    python main_preprocess_ig.py --year 2015 --phase Train
    python main_preprocess_ig.py --year 2015 --phase Test
    python main_preprocess_ig.py --year 2016 --phase Train
    python main_preprocess_ig.py --year 2016 --phase Test
"""

# https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus
import argparse
import os
from typing import Optional
import xml.etree.ElementTree as ElementTree

import torch
from tqdm import tqdm
from transformers import BertTokenizer

from model import EmbeddingsLayer
from utils import EmbeddingsDatasetIG, train_validation_split_ig
from rdflib import Graph

tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def clean_data(year: int, phase: str):
    """Clean a SemEval dataset by removing opinions with implicit targets."""
    filename = f"ABSA{year % 2000}_Restaurants_{phase}.xml"
    input_path = f"data/raw/{filename}"
    output_path = f"data/processed/{filename}"

    if os.path.isfile(output_path):
        print(f"Found cleaned file at {output_path}")
        return ElementTree.parse(output_path)

    tree = ElementTree.parse(input_path)

    n_null_removed = 0
    for opinions in tree.findall(".//Opinions"):
        for opinion in opinions.findall('./Opinion[@target="NULL"]'):
            opinions.remove(opinion)
            n_null_removed += 1

    n = 0
    n_positive = 0
    n_negative = 0
    n_neutral = 0
    for opinion in tree.findall(".//Opinion"):
        n += 1
        if opinion.attrib['polarity'] == "positive":
            n_positive += 1
        elif opinion.attrib['polarity'] == "negative":
            n_negative += 1
        elif opinion.attrib['polarity'] == "neutral":
            n_neutral += 1

    if n == 0:
        print(f"\n{filename} does not contain any opinions")
    else:
        print(f"\n{filename}")
        print(f"  Removed {n_null_removed} opinions with target NULL")
        print(f"  Total number of opinions remaining: {n}")
        print(f"  Fraction positive: {100 * n_positive / n:.3f} %")
        print(f"  Fraction negative: {100 * n_negative / n:.3f} %")
        print(f"  Fraction neutral: {100 * n_neutral / n:.3f} %")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree.write(output_path)
    print(f"Stored cleaned dataset in {output_path}")
    return tree


def generate_embeddings_with_tokens(embeddings_layer: EmbeddingsLayer, data: ElementTree, embeddings_dir: str):
    """
    Generate embeddings and store token strings alongside them.
    Each .pt file contains:
        - label: int sentiment label
        - embeddings: [N, d] tensor
        - target_pos: (start, end) tuple
        - hops: optional tensor
        - tokens: list of token strings (length N) -- NEW
    """
    os.makedirs(embeddings_dir, exist_ok=True)
    print(f"\nGenerating embeddings with token strings into {embeddings_dir}")

    labels = {
        'negative': 0,
        'neutral': 1,
        'positive': 2,
    }

    with torch.inference_mode():
        i = 0
        for node in tqdm(data.findall('.//sentence'), unit='sentence'):
            sentence = node.find('./text').text

            for opinion in node.findall('.//Opinion'):
                target_from = int(opinion.attrib['from'])
                target_to = int(opinion.attrib['to'])
                polarity = opinion.attrib['polarity']

                if polarity not in labels:
                    raise ValueError(f"Unknown polarity \"{polarity}\" found at sentence \"{sentence}\"")

                label = labels.get(polarity)
                embeddings, target_pos, hops = embeddings_layer.forward(sentence, target_from, target_to)

                # Get token strings for this sentence (matching the embedding preprocessing)
                sentence_with_cls = f"[CLS] {sentence} [SEP]"
                tokens_with_special = tokenizer.tokenize(sentence_with_cls)
                # Remove [CLS] and [SEP] tokens, matching embeddings_layer.py line: embeddings[0][1:-1]
                tokens = tokens_with_special[1:-1]

                # Verify token count matches embedding count
                assert len(tokens) == embeddings.shape[0], \
                    f"Token count mismatch: {len(tokens)} tokens vs {embeddings.shape[0]} embeddings"

                data_dict = {
                    'label': label,
                    'embeddings': embeddings,
                    'target_pos': target_pos,
                    'hops': hops,
                    'tokens': tokens,  # NEW: list of token strings
                }
                torch.save(data_dict, f"{embeddings_dir}/{i}.pt")
                i += 1

        print(f"Generated embeddings with tokens for {i} opinions")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2015, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--phase", default="Train", help="The phase of the dataset (Train or Test)")
    args = parser.parse_args()

    year: int = args.year
    phase: str = args.phase

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_default_device(device)

    data = clean_data(year, phase)
    embeddings_layer = EmbeddingsLayer(hops=None, ontology=None, device=device)

    # Store in a separate directory to avoid overwriting original embeddings
    embeddings_dir = f"data/embeddings_ig/{year}-{phase}"

    generate_embeddings_with_tokens(embeddings_layer, data, embeddings_dir)


if __name__ == "__main__":
    main()
