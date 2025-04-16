#!/usr/bin/env python3
import csv
import json
import os
import sys

from tagger.tagger_swig import Tagger, GetMatchesParams

cur_dir = os.path.dirname(__file__)
top_dir = os.path.abspath(os.path.join(cur_dir, '..'))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)
from EnergyMeter import EnergyMeter
from F1Score import compute_f1_score_contains_bipartite

"""
entities_file = "/home/julius/Schreibtisch/Uni/Studienprojekt/PREON/human_dictionary/human_entities.tsv"
names_file = "/home/julius/Schreibtisch/Uni/Studienprojekt/PREON/human_dictionary/human_names.tsv"
types_file = "/home/julius/Schreibtisch/Uni/Studienprojekt/PREON/human_dictionary/human_local.tsv"

# Pfad zum bc2gm-Datensatz (TSV) und zu den goldenen Annotationen
input_filename = "bc2gm_converted.tsv"
gold_annotations_file = "bc2gm_gold_annotations.tsv"
# Ausgabedatei für die Energie/Laufzeit-Ergebnisse
results_csv = "bc2gm_results.csv"

class_name = "GENE"
# """

"""
entities_file = "/home/julius/Schreibtisch/Uni/Studienprojekt/PREON/organisms_dictionary/organisms_entities.tsv"
names_file = "/home/julius/Schreibtisch/Uni/Studienprojekt/PREON/organisms_dictionary/organisms_names.tsv"
types_file = "/home/julius/Schreibtisch/Uni/Studienprojekt/PREON/organisms_dictionary/organisms_groups.tsv"

entities_file = "/home/julius/Schreibtisch/Uni/Studienprojekt/PREON/tagger_dictionary/tagger_entities.tsv"
names_file = "/home/julius/Schreibtisch/Uni/Studienprojekt/PREON/tagger_dictionary/tagger_names.tsv"
types_file = "/home/julius/Schreibtisch/Uni/Studienprojekt/PREON/tagger_dictionary/tagger_groups.tsv"


# Pfad zum bc2gm-Datensatz (TSV) und zu den goldenen Annotationen
input_filename = "linnaeus_converted.tsv"
gold_annotations_file = "linnaeus_gold_annotations.tsv"
# Ausgabedatei für die Energie/Laufzeit-Ergebnisse
results_csv = "linnaeus_results.csv"

class_name = "SPECIES"
# """

# """
entities_file = "/home/julius/Schreibtisch/Uni/Studienprojekt/PREON/diseases_dictionary/diseases_entities.tsv"
names_file = "/home/julius/Schreibtisch/Uni/Studienprojekt/PREON/diseases_dictionary/diseases_names.tsv"
types_file = "/home/julius/Schreibtisch/Uni/Studienprojekt/PREON/diseases_dictionary/diseases_groups.tsv"

# Pfad zum bc2gm-Datensatz (TSV) und zu den goldenen Annotationen
input_filename = "ncbi_disease_converted.tsv"
gold_annotations_file = "ncbi_disease_gold_annotations.tsv"
# Ausgabedatei für die Energie/Laufzeit-Ergebnisse
results_csv = "ncbi_disease_results.csv"

class_name = "DISEASE"
# """


def load_converted_tsv(tsv_file):
    """
    Lädt den bc2gm-Datensatz im TSV-Format.
    Erwartetes Format: identifier, authors, journal, year, title, text
    """
    data = []
    with open(tsv_file, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) < 6:
                continue
            identifier, authors, journal, year, title, text = row[:6]
            data.append({
                "id": identifier,
                "text": text
            })
    return data


def load_gold_annotations(tsv_file):
    """
    Lädt die goldenen Annotationen aus der separaten Datei bc2gm_gold_annotations.tsv.
    Erwartetes Format: id, char_start, char_end, label
    Hier werden die Keys "start" und "end" genutzt, damit sie zum F1Score-Format passen.
    """
    gold = {}
    with open(tsv_file, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            doc_id, char_start, char_end, label = row[:4]
            if doc_id not in gold:
                gold[doc_id] = []
            gold[doc_id].append({
                "start": int(char_start),
                "end": int(char_end),
                "label": label
            })
    return gold


def main():
    tag = Tagger(serials_only=False)
    tag.load_names(entities_file, names_file)
    tag.load_local(types_file)

    corpus = load_converted_tsv(input_filename)
    gold_annos = load_gold_annotations(gold_annotations_file)
    for entry in corpus:
        entry["gold_entities"] = gold_annos.get(entry["id"], [])

    num_runs = 10
    results = []

    for run in range(num_runs):
        corpus = load_converted_tsv(input_filename)
        for entry in corpus:
            entry["gold_entities"] = gold_annos.get(entry["id"], [])

        with EnergyMeter() as meter:
            for entry in corpus:
                text = entry["text"]

                params = GetMatchesParams()
                params.auto_detect = True
                params.allow_overlap = False
                params.protect_tags = True
                params.max_tokens = 5
                params.tokenize_characters = False
                params.ignore_blacklist = False
                if class_name == "SPECIES":
                    params.add_entity_type(-2)
                    params.auto_detect = False

                matches = tag.get_matches(text, entry["id"], params)
                converted_matches = []
                for match in matches:
                    if len(match) >= 3:
                        converted_matches.append((int(match[0]), int(match[1]), class_name))
                entry["pred_entities"] = converted_matches

        f1, precision, recall = compute_f1_score_contains_bipartite(corpus, class_name)
        runtime = meter.runtime
        cpu_energy = meter.cpu_energy
        gpu_energy = meter.gpu_energy
        cpu_power_avg = getattr(meter, "cpu_power_avg", 0)
        gpu_power_avg = getattr(meter, "gpu_power_avg", 0)
        gpu_power_max = getattr(meter, "gpu_power_max", 0)
        cpu_memory_avg = getattr(meter, "cpu_memory_avg", 0)
        gpu_memory_avg = getattr(meter, "gpu_memory_avg", 0)

        results.append((runtime, f1, precision, recall,
                        cpu_energy, gpu_energy,
                        cpu_power_avg,
                        gpu_power_avg, gpu_power_max,
                        cpu_memory_avg, gpu_memory_avg))
        print(f"Run {run + 1:3d}: Runtime: {runtime:.4f} s, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    avg_runtime = sum(r[0] for r in results) / num_runs
    avg_f1 = sum(r[1] for r in results) / num_runs
    avg_precision = sum(r[2] for r in results) / num_runs
    avg_recall = sum(r[3] for r in results) / num_runs
    avg_cpu_energy = sum(r[4] for r in results) / num_runs
    avg_gpu_energy = sum(r[5] for r in results) / num_runs
    avg_cpu_power_avg = sum(r[6] for r in results) / num_runs
    avg_gpu_power_avg = sum(r[7] for r in results) / num_runs
    avg_gpu_power_max = sum(r[8] for r in results) / num_runs
    avg_cpu_memory = sum(r[9] for r in results) / num_runs
    avg_gpu_memory = sum(r[10] for r in results) / num_runs

    with open(results_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Run", "Runtime (s)", "F1 Score", "Precision", "Recall",
                         "CPU Energy (J)", "GPU Energy (J)",
                         "CPU Power (avg, W)",
                         "GPU Power (avg, W)", "GPU Power (max, W)",
                         "CPU RAM (avg, MB)", "GPU VRAM (avg, MB)"])
        for i, res in enumerate(results):
            writer.writerow([i + 1,
                             f"{res[0]:.4f}",
                             f"{res[1]:.4f}",
                             f"{res[2]:.4f}",
                             f"{res[3]:.4f}",
                             f"{res[4]:.4f}",
                             f"{res[5]:.4f}",
                             f"{res[6]:.4f}",
                             f"{res[7]:.4f}",
                             f"{res[8]:.4f}",
                             f"{res[9]:.4f}",
                             f"{res[10]:.2f}"])
        writer.writerow(["Average",
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
                         f"{avg_gpu_memory:.2f}"])

    print(f"Energy- und Laufzeitergebnisse wurden in '{results_csv}' gespeichert.")


if __name__ == "__main__":
    main()
