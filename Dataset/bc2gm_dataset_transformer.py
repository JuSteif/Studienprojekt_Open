#!/usr/bin/env python3
import json
from datasets import load_dataset


def convert_bc2gm():
    """
    Lädt den Datensatz "spyysalo/bc2gm_corpus" und wandelt ihn in das Zielformat um.
    Jedes Beispiel enthält:
      - id: die Beispiel-ID
      - text: der rekonstruierte Fließtext (Tokens durch Leerzeichen verbunden)
      - gold_entities: eine Liste von Entity-Dictionaries mit den Schlüsseln:
            "start": Startzeichenoffset,
            "end": Endzeichenoffset,
            "label": Entity-Label (ohne IOB-Präfix)

    Hinweis: Diese Umwandlung erfolgt, indem die Tokens mit Leerzeichen verbunden werden.
    Für exaktere Alignment-Methoden kann eine weitergehende Anpassung nötig sein.
    """
    ds = load_dataset("spyysalo/bc2gm_corpus", split="test").select(range(1000))
    label_names = ds.features["ner_tags"].feature.names

    converted = []
    for example in ds:
        tokens = example["tokens"]
        tags = example["ner_tags"]
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
            if tag.startswith("B-"):
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
            "id": example["id"],
            "text": text,
            "gold_entities": gold_entities
        })

    return converted


def main():
    converted_data = convert_bc2gm()
    output_file = "bc2gm_converted.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=4)
    print(f"Datensatz wurde in '{output_file}' gespeichert.")


if __name__ == "__main__":
    main()
