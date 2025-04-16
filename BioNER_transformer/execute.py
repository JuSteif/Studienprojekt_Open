#!/usr/bin/env python3
import os
import sys
import csv
import json
import time
import torch
import dotenv
from transformers import AutoTokenizer, BertForTokenClassification

cur_dir = os.path.dirname(__file__)
top_dir = os.path.abspath(os.path.join(cur_dir, '..'))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

from EnergyMeter import EnergyMeter
from F1Score import compute_f1_score_contains_bipartite

dotenv.load_dotenv()


modelname = 'MilosKosRad/BioNER'


"""
json_file = "bc2gm_converted.json"
csv_file = "bc2gm_results.csv"
class_name = "GENE"
# entity_prompt = 'Protein'
entity_prompt = 'Gene'
# """

"""
json_file = "linnaeus_converted.json"
csv_file = "linnaeus_results.csv"
class_name = "SPECIES"
# entity_prompt = 'Organism'
entity_prompt = 'Species'
# """

# """
json_file = "ncbi_disease_converted.json"
csv_file = "ncbi_disease_results.csv"
class_name = "DISEASE"
entity_prompt = 'Disease'
# """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(modelname, token=os.getenv("HUGGINGFACE"))
model = BertForTokenClassification.from_pretrained(modelname, num_labels=2, token=os.getenv("HUGGINGFACE"))
model.to(device)
model.eval()

def load_converted_json(json_file):
    """
    Lädt den konvertierten BC2GM-Datensatz aus der JSON-Datei.
    Jeder Eintrag enthält:
      - id
      - text
      - gold_entities: Liste von Dictionaries mit "start", "end", "label"
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def predict_entities_bioNER(text, tokenizer, model, prompt=entity_prompt):
    """
    Verwendet das BioNER-Modell, um Entitäten im gegebenen Text zu erkennen.
    Da das Modell mit einem Satzpaar arbeitet (prompt, text), wird der erste Teil als
    Prompt übergeben, der zweite Teil (text) kommt für die Vorhersage in Frage.

    Mithilfe von return_offsets_mapping und sequence_ids() werden die Zeichenoffsets
    der Tokens des zweiten Segments (Text) ermittelt und anschließend zusammenhängende
    Tokens mit gleichem Label gruppiert.

    *Wichtig:* Nur Tokens, deren vorhergesagtes Label exakt "LABEL_1" ist, werden
    verarbeitet – diese werden auf "GENE" gemappt.

    Rückgabe:
       Eine Liste von Tupeln (start, end, label)
    """
    encoding = tokenizer(prompt, text,
                         padding=True,
                         truncation=True,
                         add_special_tokens=True,
                         return_offsets_mapping=True,
                         max_length=512,
                         return_tensors='pt')
    offset_mapping = encoding.pop("offset_mapping")
    sequence_ids = encoding.sequence_ids(0)
    encoding = {key: value.to(device) for key, value in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0].cpu())
    label_map = model.config.id2label if hasattr(model.config, "id2label") else {0: "O", 1: "LABEL_1"}

    entities = []
    current_entity = False
    current_start = None
    current_end = None
    current_label = None

    for token, pred, offsets, seq_id in zip(tokens, predictions, offset_mapping[0].tolist(), sequence_ids):
        if seq_id != 1:
            if current_entity:
                entities.append((current_start, current_end, current_label))
                current_entity = False
                current_start, current_end, current_label = None, None, None
            continue
        if offsets == [0, 0]:
            if current_entity:
                entities.append((current_start, current_end, current_label))
                current_entity = False
                current_start, current_end, current_label = None, None, None
            continue
        label = label_map[pred]
        if label != "LABEL_1":
            if current_entity:
                entities.append((current_start, current_end, current_label))
                current_entity = False
                current_start, current_end, current_label = None, None, None
            continue
        else:
            label = class_name
            if not current_entity:
                current_entity = True
                current_start, current_end, current_label = offsets[0], offsets[1], label
            else:
                if current_label == label:
                    current_end = offsets[1]
                else:
                    entities.append((current_start, current_end, current_label))
                    current_start, current_end, current_label = offsets[0], offsets[1], label
    if current_entity:
        entities.append((current_start, current_end, current_label))
    return entities

def main():
    num_runs = 10
    results = []

    for run in range(num_runs):
        corpus_run = load_converted_json(json_file)

        with EnergyMeter() as meter:
            for i, entry in enumerate(corpus_run):
                # print(f"Progress: {i/len(corpus_run)*100:.2f}%")
                text = entry["text"]
                pred_entities = predict_entities_bioNER(text, tokenizer, model, prompt=entity_prompt)
                entry["pred_entities"] = list(pred_entities)

        f1, precision, recall = compute_f1_score_contains_bipartite(corpus_run, class_name)
        runtime    = meter.runtime
        cpu_energy = meter.cpu_energy
        gpu_energy = meter.gpu_energy

        cpu_power_avg   = getattr(meter, "cpu_power_avg", 0)
        gpu_power_avg   = getattr(meter, "gpu_power_avg", 0)
        gpu_power_max   = getattr(meter, "gpu_power_max", 0)
        cpu_memory_avg  = getattr(meter, "cpu_memory_avg", 0)
        gpu_memory_avg  = getattr(meter, "gpu_memory_avg", 0)

        results.append((runtime, f1, precision, recall,
                        cpu_energy, gpu_energy,
                        cpu_power_avg, gpu_power_avg, gpu_power_max,
                        cpu_memory_avg, gpu_memory_avg))
        print(f"Run {run+1:3d}: Runtime {runtime:.4f} s, F1 {f1:.4f}, Precision {precision:.4f}, Recall {recall:.4f}")

    avg_runtime       = sum(r[0] for r in results) / num_runs
    avg_f1            = sum(r[1] for r in results) / num_runs
    avg_precision     = sum(r[2] for r in results) / num_runs
    avg_recall        = sum(r[3] for r in results) / num_runs
    avg_cpu_energy    = sum(r[4] for r in results) / num_runs
    avg_gpu_energy    = sum(r[5] for r in results) / num_runs
    avg_cpu_power_avg = sum(r[6] for r in results) / num_runs
    avg_gpu_power_avg = sum(r[7] for r in results) / num_runs
    avg_gpu_power_max = sum(r[8] for r in results) / num_runs
    avg_cpu_memory    = sum(r[9] for r in results) / num_runs
    avg_gpu_memory    = sum(r[10] for r in results) / num_runs

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Run", "Runtime (s)", "F1 Score (bipartite contains rule)", "Precision", "Recall",
            "CPU Energy (J)", "GPU Energy (J)", "CPU Power (avg, W)",
            "GPU Power (avg, W)", "GPU Power (max, W)",
            "CPU RAM (avg, MB)", "GPU VRAM (avg, MB)"
        ])
        for i, res in enumerate(results):
            writer.writerow([
                i+1,
                f"{res[0]:.4f}",
                f"{res[1]:.4f}",
                f"{res[2]:.4f}",
                f"{res[3]:.4f}",
                f"{res[4]:.4f}",
                f"{res[5]:.4f}",
                f"{res[6]:.4f}",
                f"{res[7]:.4f}",
                f"{res[8]:.4f}",
                f"{res[9]:.2f}",
                f"{res[10]:.2f}"
            ])
        writer.writerow([
            "Average",
            f"{avg_runtime:.4f}",
            f"{avg_f1:.4f}",
            f"{avg_precision:.4f}",
            f"{avg_recall:.4f}",
            f"{avg_cpu_energy:.4f}",
            f"{avg_gpu_energy:.4f}",
            f"{avg_cpu_power_avg:.4f}",
            f"{avg_gpu_power_avg:.4f}",
            f"{avg_gpu_power_max:.4f}",
            f"{avg_cpu_memory:.2f}",
            f"{avg_gpu_memory:.2f}"
        ])
    print(f"Evaluation results saved to {csv_file}")

if __name__ == "__main__":
    main()
