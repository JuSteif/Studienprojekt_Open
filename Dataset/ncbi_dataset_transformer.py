#!/usr/bin/env python3
import json
from datasets import load_dataset


def convert_ncbi_disease():
    """
    Lädt den NCBI Disease Corpus von HuggingFace (ncbi/ncbi_disease) und wandelt ihn
    in das Zielformat um. Für jedes Beispiel werden folgende Felder erzeugt:
       - id: die Beispiel-ID (aus "id")
       - text: der rekonstruierte Fließtext, gebildet aus den Tokens (durch Leerzeichen verbunden)
       - gold_entities: Eine Liste von Entity-Dictionaries mit den Schlüsseln:
             "start": Startzeichenoffset,
             "end": Endzeichenoffset,
             "label": Entity-Label (ohne IOB-Präfix, z. B. "DISEASE")

    Hinweis: Wir nutzen hier das Label-Mapping aus ds.features["ner_tags"].feature.names.
    Ein typisches Mapping könnte z. B. ["O", "B-DISEASE", "I-DISEASE", ...] lauten.
    """
    ds = load_dataset("ncbi/ncbi_disease", split="train").select(range(1000))
    label_names = ds.features["ner_tags"].feature.names

    converted = []
    for example in ds:
        doc_id = example.get("id", "no_id")
        tokens = example.get("tokens", [])
        tags = example.get("ner_tags", [])

        text = " ".join(tokens)

        token_offsets = []
        current_offset = 0
        for token in tokens:
            start = current_offset
            end = start + len(token)
            token_offsets.append((start, end))
            current_offset = end + 1

        gold_entities = []
        current_entity = None
        for i, tag_idx in enumerate(tags):
            tag = label_names[tag_idx]
            if tag == "O":
                if current_entity is not None:
                    gold_entities.append(current_entity)
                    current_entity = None
            elif tag.startswith("B-"):
                if current_entity is not None:
                    gold_entities.append(current_entity)
                current_entity = {
                    "start": token_offsets[i][0],
                    "end": token_offsets[i][1],
                    "label": tag[2:]
                }
            elif tag.startswith("I-") and current_entity is not None:
                current_entity["end"] = token_offsets[i][1]
            else:
                if current_entity is not None:
                    gold_entities.append(current_entity)
                    current_entity = None
        if current_entity is not None:
            gold_entities.append(current_entity)

        converted.append({
            "id": doc_id,
            "text": text,
            "gold_entities": gold_entities
        })

    return converted


def main():
    converted_data = convert_ncbi_disease()
    output_file = "ncbi_disease_converted.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=4)
    print(f"Datensatz wurde in '{output_file}' gespeichert.")


if __name__ == "__main__":
    main()
