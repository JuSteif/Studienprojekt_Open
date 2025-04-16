#!/usr/bin/env python3
import csv
import json
import time
import os
import sys

from flair.data import Sentence
from flair.nn import Classifier

cur_dir = os.path.dirname(__file__)
top_dir = os.path.abspath(os.path.join(cur_dir, '..'))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

# Importiere die Auslagerungen für Energie-Messung und F1-Berechnung (bipartites Matching)
from EnergyMeter import EnergyMeter
from F1Score import nx, compute_f1_score_contains_bipartite, build_bipartite_graph

"""
json_file = "bc2gm_converted.json"
csv_file = "bc2gm_no_splitter_results.csv"
class_name = "GENE"
# """

"""
json_file = "linnaeus_converted.json"
csv_file = "linnaeus_no_splitter_results.csv"
class_name = "SPECIES"
# """

# """
json_file = "ncbi_disease_converted.json"
csv_file = "ncbi_disease_no_splitter_results.csv"
class_name = "DISEASE"
# """

COLOR_GREEN  = "\033[92m"
COLOR_RED    = "\033[91m"
COLOR_YELLOW = "\033[93m"
RESET_COLOR  = "\033[0m"

def highlight_entities_different_colors(text, colored_entities):
    """
    Erzeugt einen Text, in dem die Bereiche, die in colored_entities angegeben sind,
    mithilfe des jeweiligen Farbcodes hervorgehoben werden.
    Erwartet wird eine Liste von Tupeln (start, end, color_code).
    """
    sorted_entities = sorted(colored_entities, key=lambda x: x[0])
    highlighted = ""
    last_index = 0
    for (start, end, color) in sorted_entities:
        highlighted += text[last_index:start]
        highlighted += f"{color}{text[start:end]}{RESET_COLOR}"
        last_index = end
    highlighted += text[last_index:]
    return highlighted

def load_converted_json(json_file):
    """
    Lädt den konvertierten Datensatz aus der JSON-Datei.
    Jeder Eintrag enthält:
      - id
      - text
      - gold_entities: Liste von Dictionaries mit "start", "end", "label"
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def run_hunflair2(text, tagger):
    """
    Erzeugt ein flair.data.Sentence aus dem Text, führt die NER-Vorhersage mit HunFlair2 durch
    und gibt das annotierte Sentence-Objekt zurück.
    """
    sentence = Sentence(text)
    tagger.predict(sentence)
    return sentence

def extract_predicted_entities(sentence):
    """
    Extrahiert aus dem flair.data.Sentence-Objekt die vom HunFlair2-Modell berechneten Entitäten
    als Menge von Tupeln im Format (start_pos, end_pos, label). Dabei werden gängige Präfixe
    (B-, I-, E-, S-) entfernt und das Label in Großbuchstaben umgewandelt.
    """
    entities = set()
    sent_dict = sentence.to_dict()
    for ent in sent_dict.get("entities", []):
        label = ent["labels"][0]["value"] if ent["labels"] else ""
        if label.startswith(("B-", "I-", "E-", "S-")):
            label = label[2:]
        label = label.upper()
        entities.add((ent["start_pos"], ent["end_pos"], label))
    return entities

def main():
    num_runs = 10
    results = []

    tagger = Classifier.load("hunflair2")

    for run in range(num_runs):
        corpus = load_converted_json(json_file)

        with EnergyMeter() as meter:
            for entry in corpus:
                text = entry["text"]
                sentence = run_hunflair2(text, tagger)
                pred_entities = extract_predicted_entities(sentence)
                entry["pred_entities"] = list(pred_entities)

        f1, precision, recall = compute_f1_score_contains_bipartite(corpus, class_name)
        runtime    = meter.runtime
        cpu_energy = meter.cpu_energy
        gpu_energy = meter.gpu_energy

        cpu_power_avg = getattr(meter, "cpu_power_avg", 0)
        gpu_power_avg = getattr(meter, "gpu_power_avg", 0)
        gpu_power_max = getattr(meter, "gpu_power_max", 0)
        cpu_memory_avg = getattr(meter, "cpu_memory_avg", 0)
        gpu_memory_avg = getattr(meter, "gpu_memory_avg", 0)

        results.append((runtime, f1, precision, recall,
                        cpu_energy, gpu_energy,
                        cpu_power_avg, gpu_power_avg, gpu_power_max,
                        cpu_memory_avg, gpu_memory_avg))
        print(f"Run {run+1:3d}: Runtime {runtime:.4f} s, F1 {f1:.4f}, Precision {precision:.4f}, Recall {recall:.4f}")

    avg_runtime    = sum(r[0] for r in results) / num_runs
    avg_f1         = sum(r[1] for r in results) / num_runs
    avg_precision  = sum(r[2] for r in results) / num_runs
    avg_recall     = sum(r[3] for r in results) / num_runs
    avg_cpu_energy = sum(r[4] for r in results) / num_runs
    avg_gpu_energy = sum(r[5] for r in results) / num_runs
    avg_cpu_power  = sum(r[6] for r in results) / num_runs
    avg_gpu_power  = sum(r[7] for r in results) / num_runs
    avg_gpu_power_max = sum(r[8] for r in results) / num_runs
    avg_cpu_memory = sum(r[9] for r in results) / num_runs
    avg_gpu_memory = sum(r[10] for r in results) / num_runs

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
            f"{avg_cpu_power:.4f}",
            f"{avg_gpu_power:.4f}",
            f"{avg_gpu_power_max:.4f}",
            f"{avg_cpu_memory:.2f}",
            f"{avg_gpu_memory:.2f}"
        ])
    print(f"Evaluation results saved to {csv_file}")

if __name__ == "__main__":
    main()
