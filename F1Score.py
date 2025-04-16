import networkx as nx


def is_gold_contained_in_pred(gold, pred, text):
    """
    Prüft, ob der Textbereich der Gold-Annotation (gold: (start, end, label))
    vollständig oder teilweise im Textbereich der Prediction (pred: (start, end, label)) enthalten ist.
    Es wird wechselseitig geprüft, ob (gold_sub in pred_sub) oder (pred_sub in gold_sub).
    """
    if gold[2] != pred[2]:
        return False
    gold_sub = text[gold[0]:gold[1]]
    pred_sub = text[pred[0]:pred[1]]
    return (gold_sub in pred_sub) or (pred_sub in gold_sub)


def build_bipartite_graph(gold_entities, pred_entities, text):
    """
    Baut einen bipartiten Graphen, in dem:
      - Knoten: Gold-Annotationen (linke Seite) und Predictions (rechte Seite)
      - Es wird eine Kante ( ("gold", i), ("pred", j) ) angelegt, falls die gold-Entity und
        die Prediction gemäß is_gold_contained_in_pred matchen.
    """
    G = nx.Graph()
    for i in range(len(gold_entities)):
        G.add_node(("gold", i))
    for j in range(len(pred_entities)):
        G.add_node(("pred", j))
    for i, g in enumerate(gold_entities):
        for j, p in enumerate(pred_entities):
            if is_gold_contained_in_pred(g, p, text):
                G.add_edge(("gold", i), ("pred", j))
    return G


def compute_f1_score_contains_bipartite(corpus, target_type="GENE"):
    """
    Berechnet den F1-Score für einen gewünschten Entity-Typ (z.B. "GENE" oder "SPECIES")
    mithilfe eines bipartiten Maximum Matchings. Dabei wird zunächst aus jedem Beispiel in
    corpus nur die Gold- und Predicted-Annotationen des gewünschten Typs ausgewählt.
    Anschließend wird ein bipartiter Graph erstellt, und über nx.max_weight_matching
    das maximale Matching ermittelt.
    Die Anzahl der Matchings wird als Anzahl True Positives gewertet.
    """
    total_gold = 0
    total_pred = 0
    total_tp = 0

    for entry in corpus:
        text = entry["text"]
        gold_entities = [(ent["start"], ent["end"], ent["label"].upper())
                         for ent in entry.get("gold_entities", [])
                         if ent["label"].upper() == target_type.upper()]
        pred_entities = [pe for pe in entry.get("pred_entities", [])
                         if pe[2].upper() == target_type.upper()]

        total_gold += len(gold_entities)
        total_pred += len(pred_entities)

        G = build_bipartite_graph(gold_entities, pred_entities, text)
        matching = nx.max_weight_matching(G, maxcardinality=True)
        tp_example = len(matching)
        total_tp += tp_example

    precision = total_tp / total_pred if total_pred > 0 else 0
    recall = total_tp / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return f1, precision, recall
