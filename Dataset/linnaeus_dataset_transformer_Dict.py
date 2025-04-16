#!/usr/bin/env python3
import csv
from datasets import load_dataset

def adjust_offset(original_text, offset):
    """
    Als Fallback: Falls die Offsets modifiziert werden müssen, werden hier CRLF-Sequenzen berücksichtigt.
    (Falls der Text nur "\n" enthält, bleibt die Länge gleich.)
    """
    crlf_count = original_text[:offset].count("\r\n")
    return offset - crlf_count

def convert_linnaeus_to_tsv(text_output_file="linnaeus_converted.tsv",
                            anno_output_file="linnaeus_gold_annotations.tsv",
                            split="test"):
    """
    Lädt den Linnaeus-Datensatz (bigbio/linnaeus) und schreibt ihn in zwei Dateien:

    1. Eine TSV-Datei mit den Spalten:
         identifier, authors, journal, year, title, text
       (Da in diesem Datensatz keine Metadaten vorhanden sind, werden diese Felder mit "NA"
        bzw. einem Beispielwert gefüllt. Der Dokumenten-Text wird durch Zusammenfügen
        der Passage-Texte erzeugt. Eventuelle Zeilenumbrüche im Text werden durch Tabs ersetzt.)

    2. Eine zweite TSV-Datei mit den Goldannotationen für SPECIES. Für jede Entität aus dem Feld
       `entities`, deren Typ "species" (case-insensitive) ist, wird folgendes gemacht:
         - Falls in der Annotation ein Feld "text" vorhanden ist, wird daraus ein String rekonstruiert
           und sein erstes Vorkommen im bereinigten Text gesucht.
         - Wird der Text gefunden, so werden die entsprechenden Offsets (new_start, new_end) genutzt.
         - Andernfalls wird als Fallback der ursprünglich angegebene Offset (nach adjust_offset) verwendet.
         - Das Label wird als "SPECIES" geschrieben.
    """
    ds = load_dataset("bigbio/linnaeus", "linnaeus_bigbio_kb", split="train")

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
            year = "2001"
            title = "NA"

            passages = example.get("passages", [])
            passage_texts = []
            for passage in passages:
                texts = passage.get("text", [])
                if isinstance(texts, list):
                    passage_texts.append(" ".join(texts))
                elif isinstance(texts, str):
                    passage_texts.append(texts)
            full_text = " ".join(passage_texts)
            clean_text = full_text.replace("\r\n", "  ").replace("\n", " ").replace("\r", " ").replace("\t", " ")

            text_writer.writerow([identifier, authors, journal, year, title, clean_text])

            entities = example.get("entities", [])
            for entity in entities:
                if entity.get("type", "").lower() == "species":
                    offsets_list = entity.get("offsets", [])
                    entity_text = ""
                    if "text" in entity:
                        et = entity["text"]
                        if isinstance(et, list):
                            entity_text = " ".join(et)
                        elif isinstance(et, str):
                            entity_text = et
                    for offset_pair in offsets_list:
                        if isinstance(offset_pair, list) and len(offset_pair) == 2:
                            orig_start, orig_end = offset_pair
                            fallback_start = adjust_offset(full_text, orig_start)
                            fallback_end = adjust_offset(full_text, orig_end)
                            new_start, new_end = fallback_start, fallback_end
                            if entity_text:
                                found_index = clean_text.find(entity_text)
                                if found_index != -1:
                                    new_start = found_index
                                    new_end = found_index + len(entity_text)
                            anno_writer.writerow([identifier, new_start, new_end, "SPECIES"])

    print(f"Datensatz wurde in '{text_output_file}' und '{anno_output_file}' gespeichert.")

def main():
    convert_linnaeus_to_tsv()

if __name__ == "__main__":
    main()
