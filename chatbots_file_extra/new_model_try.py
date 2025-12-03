from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import torch

model_name = "Babelscape/rebel-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def extract_triples(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=256)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    triples = []


    for t in decoded.split("<triplet>"):
        parts = [p.strip() for p in t.split("|")]
        if len(parts) == 3:
            subj, rel, obj = parts
            triples.append((subj, rel, obj))

    return triples


def triples_to_graphjson(triples):
    nodes = {}
    edges = []

    for s, r, o in triples:
        if s not in nodes:
            nodes[s] = {"id": s.replace(" ", "_"), "type": "Entity", "properties": {"label": s}}
        if o not in nodes:
            nodes[o] = {"id": o.replace(" ", "_"), "type": "Entity", "properties": {"label": o}}

        edges.append({
            "from": s.replace(" ", "_"),
            "to": o.replace(" ", "_"),
            "type": r.replace(" ", "_").upper()
        })

    return {
        "nodes": list(nodes.values()),
        "edges": edges
    }



text = """
Goods procurement is a process with multiple steps. It requires procurement leaders to source, negotiate, manage contracts, and handle order placement to ensure their businesses have a steady and high-quality supply of the goods they need. This article covers the particulars of the procurement process for goods and how, when done efficiently and responsibly, can bring maximum value to an organization. After reading this article, youll understand the distinction between goods and services in the procurement process, and how cost savings and sustainability can co-exist to drive equal value and impact to an organization. I have created a free-to-download editable cost price breakdown template. Its a PowerPoint file, together with an Excel file, that can help you save cost. I even created a video where Ill explain how you can use this template. Download Cost Price Breakdown Template Overview of Goods Procurement Goods procurement specifically refers to the process of sourcing and buying goods, such as raw materials and components. Once acquired, these goods will be used within a business to produce an end product or to aid company operations. Procuring goods requires careful planning around product quality, cost, delivery times, supplier stability, and regulatory adherence. This process can vary depending on the industry, company, or type of goods being sourced.
"""

triples = extract_triples(text)
graph = triples_to_graphjson(triples)

print(json.dumps(graph, indent=2))