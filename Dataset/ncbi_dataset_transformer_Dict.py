#!/usr/bin/env python3
import csv
from datasets import load_dataset

def convert_ncbi_disease_to_tsv(text_output_file="ncbi_disease_converted.tsv", anno_output_file="ncbi_disease_gold_annotations.tsv"):
    """
    Lädt den ncbi/ncbi_disease-Datensatz und schreibt ihn in zwei Dateien:

    1. Eine TSV-Datei mit den Spalten:
         identifier, authors, journal, year, title, text
       (Da keine Metadaten vorhanden sind, werden diese Felder mit "NA" gefüllt).

    2. Eine zweite TSV-Datei mit den Goldannotationen. Für jede erkannte Entität
       (gemäß den numerischen ner_tags, wobei:
           0 => kein Entity,
           1 => Beginn einer Entität,
           2 => Fortsetzung einer Entität)
       werden die Zeichenoffsets (Start und Endposition im zusammengefügten Text)
       sowie das Label "DISEASE" ausgegeben.
    """
    ds = load_dataset("ncbi/ncbi_disease", split="train").select(range(1000))

    with open(text_output_file, "w", encoding="utf-8", newline="") as text_file, \
         open(anno_output_file, "w", encoding="utf-8", newline="") as anno_file:

        text_writer = csv.writer(text_file, delimiter="\t")
        anno_writer = csv.writer(anno_file, delimiter="\t")

        text_writer.writerow(["identifier", "authors", "journal", "year", "title", "text"])
        anno_writer.writerow(["id", "char_start", "char_end", "label"])

        for i, example in enumerate(ds):
            identifier = "PMID:" + str(i)
            authors = "NA"
            journal = "NA"
            year = "NA"
            title = "NA"

            tokens = example.get("tokens", [])
            text = " ".join(tokens)

            text_writer.writerow([identifier, authors, journal, year, title, text])

            token_spans = []
            current_offset = 0
            for token in tokens:
                start = current_offset
                end = start + len(token)
                token_spans.append((start, end))
                current_offset = end + 1

            ner_tags = example.get("ner_tags", [])

            entities = []
            current_entity = None

            for j, tag in enumerate(ner_tags):
                if tag == 0:
                    if current_entity is not None:
                        entities.append(current_entity)
                        current_entity = None
                    continue
                if tag == 1:
                    if current_entity is not None:
                        entities.append(current_entity)
                    current_entity = {
                        "start": token_spans[j][0],
                        "end": token_spans[j][1],
                        "label": "DISEASE"
                    }
                elif tag == 2:
                    if current_entity is not None:
                        current_entity["end"] = token_spans[j][1]
                    else:
                        current_entity = {
                            "start": token_spans[j][0],
                            "end": token_spans[j][1],
                            "label": "DISEASE"
                        }
            if current_entity is not None:
                entities.append(current_entity)

            for entity in entities:
                anno_writer.writerow([identifier, entity["start"], entity["end"], entity["label"]])

    print(f"Datensatz wurde in '{text_output_file}' und '{anno_output_file}' gespeichert.")


def main():
    convert_ncbi_disease_to_tsv()


if __name__ == "__main__":
    main()
