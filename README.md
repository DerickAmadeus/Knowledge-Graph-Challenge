# 📚 Knowledge Graph Construction with Personality Modeling

> Automated pipeline for extracting knowledge graphs from unstructured text with Big Five personality trait inference

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.0+-green.svg)](https://spacy.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

This project implements an intelligent NLP pipeline that automatically constructs **Knowledge Graphs (KGs)** from unstructured text documents. The system goes beyond basic entity and relationship extraction by inferring and modeling **Big Five (OCEAN) personality traits** of individuals mentioned in the text.

**Key Capabilities:**
- 🔍 Entity Recognition & Linking
- 🔗 Relationship Extraction (SPO triples)
- 🧠 Personality Trait Inference (Big Five/OCEAN)
- 📊 Comprehensive Evaluation Metrics
- 🎨 Interactive Graph Visualization

---

## ✨ Features

### 🔤 **Text Processing**
- Supports plain text files (`.txt`)
- Multi-document batch processing
- UTF-8 and Latin-1 encoding support

### 🎯 **Entity Recognition**
- **NER Engine**: spaCy `en_core_web_sm` model
- **Entity Types**: PERSON, ORG, GPE, PRODUCT, LOCATION
- **Smart Detection**: Handles proper nouns and capitalized names

### 🔗 **Entity Linking & Coreference Resolution**
- Unifies name variations (e.g., "Dr. Vance" → "Elias Vance")
- Resolves pronouns to canonical names (He/She → Person)
- Handles titles, prefixes, and possessives
- Advanced substring matching for compound names

### 📝 **Relation Extraction**
Uses spaCy's dependency parser to extract:
- **Subject-Predicate-Object (SPO)** triples
- **IS_A** relationships (e.g., "Alice is a scientist")
- **OWNS** relationships (possessive constructs)
- **HAS_TRAIT** personality connections
- Verb-based and prepositional relationships

### 🧠 **Personality Modeling**
Infers **Big Five (OCEAN)** personality traits:
- **O**penness to Experience
- **C**onscientiousness
- **E**xtraversion / Introversion
- **A**greeableness
- **N**euroticism

**Scoring:** 1-5 scale based on configurable keyword matching with contextual evidence tracking.

### 📊 **Evaluation Framework**

#### **Intrinsic Metrics** (No Ground Truth Required)
- **Graph Coherence** (95.4%): Structure quality and connectivity
- **Information Richness** (100%): Extraction efficiency per sentence
- **Semantic Completeness** (95.4%): Personality profile completeness
- **Entity Recognition** (85%): Detection quality
- **Relationship Diversity** (86%): Variety of relationship types
- **Overall Score**: 94.1/100 ⭐⭐⭐⭐⭐

#### **Supervised Metrics** (With Ground Truth)
- Entity Precision/Recall/F1-Score
- Relation Precision/Recall/F1-Score (with partial matching)
- Matched vs Expected relations analysis

See [EVALUATION_METRICS.md](EVALUATION_METRICS.md) for detailed documentation.

### 🎨 **Visualization**
- **Interactive HTML**: Powered by vis.js network visualization
- **Color-coded nodes**: Person (blue), Trait (green), Concept (gold), etc.
- **Labeled edges**: Relationship types with scores
- **Responsive layout**: Spring-layout algorithm for optimal positioning

---

## 🛠️ Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Core language | 3.7+ |
| **spaCy** | NLP (NER, POS, parsing) | 3.0+ |
| **NetworkX** | Graph construction | Latest |
| **Matplotlib** | Static visualization | Latest |
| **Pandas** | Evaluation reports | Latest |
| **vis.js** | Interactive HTML graphs | 9.1.2 |

---

## 🚀 Quick Start

### 📹 Quick Demo

```bash
# Clone the repository
git clone https://github.com/DerickAmadeus/Knowledge-Graph-Challenge.git
cd Knowledge-Graph-Challenge

# Install dependencies
pip install spacy networkx pandas matplotlib

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the pipeline
cd src
python test.py

# View results in browser (automatically opens)
# knowledge_graph_1.html, knowledge_graph_2.html, knowledge_graph_3.html
```

### 1️⃣ Prerequisites

```bash
# Ensure Python 3.7+ is installed
python --version

# Install required packages
pip install spacy networkx pandas matplotlib
```

### 2️⃣ Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 3️⃣ Prepare Input Files

Create text files with your documents:
```
project/
├── test.py
├── 1.txt          # Your first document
├── 2.txt          # Your second document
└── 3.txt          # Your third document
```

**Example content** (`1.txt`):
```
Alice is a kind and brilliant scientist who works with Bob in London. 
She often leads meetings and enjoys collaborating with her colleagues.
```

### 4️⃣ Run the Pipeline

```bash
python test.py
```

### 5️⃣ View Results

**Console Output:**
```
[GRAPH STRUCTURE]
  * Total Nodes: 10
  * Total Edges: 13
  * Persons: 2
  * Trait Coverage: 100.0%

[INTRINSIC QUALITY METRICS]
  * Graph Coherence: 95.4%
  * Information Richness: 100.0%
  * Intrinsic Score: 94.1/100
```

**Generated Files:**
- `knowledge_graph_1.html` - Interactive visualization
- `knowledge_graph_2.html` - Second document graph
- `knowledge_graph_3.html` - Third document graph
- `evaluation_report.csv` - Comprehensive metrics

---

## ⚙️ Configuration

### Personality Keywords

Edit `DEFAULT_TRAIT_CLUES` in `test.py`:

```python
DEFAULT_TRAIT_CLUES = {
    "Conscientiousness": {
        "keywords": ["focused", "methodical", "organized", "diligent"],
        "score_mod": 1.0
    },
    "Extraversion": {
        "keywords": ["leads meetings", "networking", "talkative"],
        "score_mod": 1.0
    },
    # Add your custom keywords...
}
```

### Title Prefixes

Customize name cleaning patterns:

```python
TITLE_PREFIXES = r'(Dr\.|Prof\.|Professor|Mr\.|Ms\.|Mrs\.|President|CEO)'
```

### Ground Truth (Optional)

Define expected entities and relations for supervised evaluation:

```python
GROUND_TRUTH = {
    "1.txt": {
        "persons": ["Alice", "Bob"],
        "relations": [
            ("Alice", "IS_A", "scientist"),
            ("Alice", "HAS_TRAIT", "Openness"),
        ]
    }
}
```

---

## 📊 Example Output

### Knowledge Graph Visualization
![Knowledge Graph Example](knowledge_graph_1.html)

### Evaluation Report
```
File: 1.txt
─────────────────────────────────────────
Graph Coherence:      95.4% ✓
Information Richness: 100.0% ✓
Semantic Completeness: 95.4% ✓
Entity Recognition:   85.0% ✓
Relationship Diversity: 86.0% ✓
─────────────────────────────────────────
INTRINSIC SCORE: 94.1/100 ⭐⭐⭐⭐⭐
```

---

## 📁 Project Structure

```
Intellumina/
├── test.py                    # Main pipeline script
├── 1.txt, 2.txt, 3.txt       # Input documents
├── knowledge_graph_*.html     # Generated visualizations
├── evaluation_report.csv      # Metrics summary
├── EVALUATION_METRICS.md      # Detailed metrics documentation
├── README.md                  # This file
└── lib/                       # JavaScript libraries
    ├── vis-9.1.2/
    └── tom-select/
```

---

## 🔬 Evaluation Metrics

### Without Ground Truth (Intrinsic)
Perfect for production use when you don't have annotated data:

| Metric | Weight | Description |
|--------|--------|-------------|
| **Graph Coherence** | 20% | Structural quality, connectivity |
| **Information Richness** | 25% | Extraction efficiency |
| **Semantic Completeness** | 30% | Personality profile completeness |
| **Entity Recognition** | 15% | Detection quality |
| **Relationship Diversity** | 10% | Variety of relations |

### With Ground Truth (Supervised)
For research and benchmarking:
- Entity Precision/Recall/F1
- Relation Precision/Recall/F1 (partial matching supported)
- Comprehensive accuracy analysis

📖 **Full documentation**: [EVALUATION_METRICS.md](EVALUATION_METRICS.md)

---

## 🎯 Use Cases

- 📝 **Research Papers**: Extract author relationships and expertise
- 💼 **Business Documents**: Identify key stakeholders and roles
- 📰 **News Articles**: Map people, organizations, and events
- 🎓 **Academic Analysis**: Study personality traits in literature
- 🔍 **HR & Recruitment**: Analyze candidate profiles
- 🤝 **Social Network Analysis**: Understand relationship dynamics

---

## ⚠️ Limitations & Future Work

### Current Limitations

| Area | Limitation | Impact |
|------|------------|--------|
| **Relation Extraction** | Rule-based dependency patterns | May miss complex/implicit relations |
| **Coreference** | Heuristic name mapping | Occasional fragmentation |
| **Personality Model** | Keyword matching | Limited context sensitivity |
| **Negation** | Not handled | "not organized" → still scores high |
| **Sarcasm** | Not detected | Misinterprets ironic statements |

---

## 📧 Contact & Support

- **Author**: Derick Amadeus Budiono
- **Repository**: [Knowledge-Graph-Challenge](https://github.com/DerickAmadeus/Knowledge-Graph-Challenge)
- **Issues**: [Report bugs](https://github.com/DerickAmadeus/Knowledge-Graph-Challenge/issues)
- **Discussions**: [Q&A and Ideas](https://github.com/DerickAmadeus/Knowledge-Graph-Challenge/discussions)

---

## 📊 Project Statistics

```
Language: Python
Lines of Code: ~800
Functions: 15+
Evaluation Metrics: 10+
Supported Entities: 7 types
Personality Traits: 5 (Big Five)
```

---

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐!

---

[⬆ Back to Top](#-knowledge-graph-construction-with-personality-modeling)

</div>
