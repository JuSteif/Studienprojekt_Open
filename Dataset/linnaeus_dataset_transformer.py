#!/usr/bin/env python3
import json
from datasets import load_dataset


def convert_linnaeus():
    """
    Lädt den LINNAEUS-Datensatz von HuggingFace (bigbio/linnaeus) und wandelt ihn in das Zielformat um.
    Für jedes Dokument werden folgende Felder erzeugt:
       - id: die Dokumenten-ID (aus document_id)
       - text: der rekonstruierte Fließtext, gebildet aus allen Passage-Texten (durch Leerzeichen verbunden)
       - gold_entities: Eine Liste von Entity-Dictionaries mit den Schlüsseln:
             "start": Startzeichenoffset,
             "end": Endzeichenoffset,
             "label": Entity-Label (hier "SPECIES")
    Es werden nur Entitäten berücksichtigt, deren Typ "species" (case-insensitive) ist.
    """
    ds = load_dataset("bigbio/linnaeus", "linnaeus_bigbio_kb", split="train")

    converted = []
    for example in ds:
        doc_id = example.get("document_id", "no_id")

        passages = example.get("passages", [])
        passage_texts = []
        for p in passages:
            if "text" in p:
                txt = p.get("text", [])
                if isinstance(txt, list):
                    joined = " ".join([s.strip() for s in txt if isinstance(s, str) and s.strip()])
                    if joined:
                        passage_texts.append(joined)
                elif isinstance(txt, str):
                    cleaned = txt.strip()
                    if cleaned:
                        passage_texts.append(cleaned)
        text = " ".join(passage_texts)

        if not text:
            text = example.get("document_text", "").strip()

        gold_entities = []
        for entity in example.get("entities", []):
            if entity.get("type", "").lower() == "species":
                offsets = entity.get("offsets", [])
                if offsets and isinstance(offsets, list) and len(offsets) > 0:
                    start, end = offsets[0]
                    gold_entities.append({
                        "start": start,
                        "end": end,
                        "label": entity.get("type", "").upper()
                    })

        converted.append({
            "id": doc_id,
            "text": text,
            "gold_entities": gold_entities
        })

    return converted


def main():
    converted_data = convert_linnaeus()
    output_file = "linnaeus_converted.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=4)
    print(f"Datensatz wurde in '{output_file}' gespeichert.")


if __name__ == "__main__":
    main()
