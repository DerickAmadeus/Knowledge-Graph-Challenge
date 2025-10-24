Knowledge Graph Construction from Text with Personality Modeling

1. Introduction & Objective

This project implements a Python pipeline to automatically construct Knowledge Graphs (KGs) from unstructured text documents. The primary goal is to extract factual entities and relationships while also inferring and representing the Big Five (OCEAN) personality traits of individuals mentioned in the text. The output is a structured KG, visualized interactively, capturing both knowledge and personality insights.

2. Features

Text Input: Processes plain text files (.txt).

Entity Recognition: Uses spaCy (en_core_web_sm) to identify entities like PERSON, ORG, GPE.

Entity Linking & Coreference Resolution: Implements heuristics (create_canonical_name_map) to unify variations of person names (e.g., "Dr. Vance", "Elias Vance") and resolve basic pronouns ("He", "She") to a canonical entity name.

Relation Extraction: Employs spaCy's dependency parser (extract_triples_spacy_enhanced) to extract Subject-Predicate-Object (SPO) triples, "IS_A", and "OWNS" relationships.

Personality Modeling: Assigns Big Five personality scores (1-5) based on configurable keyword matching (assess_personality_from_text, DEFAULT_TRAIT_CLUES) within sentences related to identified persons.

Graph Construction: Builds a NetworkX Labeled Property Graph (LPG), storing personality scores and evidence as edge properties.

Evaluation Metrics: Calculates structural graph metrics (nodes, edges, density, degree, types) and intrinsic quality scores (coherence, richness, completeness, etc.). Includes optional ground truth comparison (Precision, Recall, F1).

Interactive Visualization: Generates an HTML file using pyvis for interactive graph exploration.

3. Technologies Used

Python 3.x

spaCy: For core NLP tasks (NER, POS tagging, dependency parsing).

NetworkX: For creating and managing the graph structure.

Pyvis: For generating interactive HTML graph visualizations.

Matplotlib: (Used in previous versions, potentially needed for static plots if Pyvis fails).

Pandas: For formatting the evaluation report.

4. Setup & Usage

Prerequisites:

Ensure Python 3 is installed.

Install required libraries:

pip install spacy networkx pyvis pandas matplotlib


Download the spaCy English model:

python -m spacy download en_core_web_sm


Input Files:

Create text files (e.g., 1.txt, 2.txt, 3.txt) containing the documents you want to process. Place them in the same directory as the script.

Run the Script:

Modify the FILE_NAMES list in kg_processor.py to include the files you want to process.

Execute the script from your terminal:

python kg_processor.py


Output:

The script will print evaluation metrics for each processed file to the console.

An interactive HTML file (e.g., knowledge_graph_1.html) will be generated for each input file and automatically opened in your web browser (if possible).

A summary evaluation_report.csv file will be created containing metrics for all processed documents.

5. Configuration

Personality Keywords: The DEFAULT_TRAIT_CLUES dictionary at the beginning of kg_processor.py defines the keywords and score adjustments used for personality assessment. You can modify this dictionary to adapt the scoring to different domains or add more nuanced rules.

Titles/Prefixes: The TITLE_PREFIXES regex variable controls which titles are stripped during name cleaning.

6. Limitations

Rule-Based Brittleness: Both relation extraction and personality assessment rely on rules (dependency patterns, keywords) which may fail on complex or unexpected sentence structures. Implicit relations are often missed.

Coreference/Entity Linking: The heuristic-based name mapping and pronoun resolution are not perfect and can fail, potentially leading to fragmented or incorrect entity representations. A dedicated coreference model is recommended for higher accuracy.

Personality Model Simplicity: Keyword matching lacks deep semantic understanding and context sensitivity (e.g., negation, sarcasm).

Evaluation Dependency: While intrinsic metrics are provided, robust evaluation requires comparison against comprehensive, human-annotated ground truth data.