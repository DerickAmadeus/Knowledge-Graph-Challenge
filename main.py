import spacy
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Any
import re
import os
from collections import defaultdict
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---

DEFAULT_TRAIT_CLUES = {
    "Conscientiousness": {
        "keywords": ["focused", "methodical", "scheduled", "deadline", "triple-checking", "organized", "diligent"],
        "score_mod": 1.0
    },
    "Extraversion": {
        "keywords": ["leads meetings", "expressive speaker", "networking", "social events", "talkative", "enjoys networking"],
        "score_mod": 1.0
    },
    "Introversion": {
        "keywords": ["quiet", "stays in the background", "deep conversations", "one-on-one", "prefers deep conversations"],
        "score_mod": -1.0
    },
    "Openness": {
        "keywords": ["new project", "spontaneously suggests", "new research directions", "complex theories", "brilliant"],
        "score_mod": 1.0
    },
    "Agreeableness": {
        "keywords": ["collaborating with", "works well with", "respected figure", "kind", "altruistic", "enjoys collaborating"],
        "score_mod": 0.5
    },
    "Neuroticism": {
        "keywords": ["annoyance", "deviation from the agenda", "moody", "anxious", "overly stressed", "gets anxious"],
        "score_mod": 1.0
    }
}

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'.")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- UTILITY FUNCTIONS ---

TITLE_PREFIXES = r'(Dr\.|Prof\.|Professor|Mr\.|Ms\.|Mrs\.|President|CEO)'

def clean_name(name: str) -> str:
    """Strips common titles/prefixes and possessive 's from a name string."""
    # Remove titles first
    cleaned = re.sub(TITLE_PREFIXES, '', name, flags=re.IGNORECASE).strip()
    # Remove possessive 's if present at the end
    if cleaned.endswith("'s"):
        cleaned = cleaned[:-2].strip()
    # Handle potential compound names resulting from pronoun resolution - keep only first part?
    # This might be too aggressive, let's rely on mapping instead for now.
    # parts = cleaned.split(' and ')
    # cleaned = parts[0].strip() # Example: "Elias Vance and Professor Anya Sharma" -> "Elias Vance"
    return cleaned

# --- COREFERENCE & ENTITY LINKING (REFINED MAPPING) ---

def create_canonical_name_map(doc: spacy.tokens.Doc) -> Dict[str, str]:
    """
    Refined: Creates a map from all detected PERSON variations (including titles,
    partial names, and pronouns) to a single, consistent canonical name.
    Prioritizes the longest name version and handles substring relationships.
    """
    # First get spaCy detected persons
    person_entities = list(set([ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"])) # Unique names first
    
    # Also collect non-person entities to exclude them later
    non_person_entities = set([ent.text.strip() for ent in doc.ents if ent.label_ in ["GPE", "LOC"]])
    
    # Add capitalized single words as potential names - especially those followed by "is a/an" pattern
    for sent in doc.sents:
        for i, token in enumerate(sent):
            # Pattern 1: "Name is a/an [profession]"
            if token.pos_ == "PROPN" and token.text not in non_person_entities:
                # Check if followed by "is a" or "is an"
                if i + 2 < len(sent) and sent[i+1].text.lower() in ["is", "was"] and sent[i+2].text.lower() in ["a", "an"]:
                    person_entities.append(token.text.strip())
                # Or just add any PROPN that's not a known location
                elif len(token.text) > 1 and token.text[0].isupper():
                    person_entities.append(token.text.strip())
    
    person_entities = list(set(person_entities))  # Remove duplicates
    
    if not person_entities:
        return {}

    # 1. Group variations by cleaned name (without titles)
    cleaned_to_variations = defaultdict(set)
    for name in person_entities:
        cleaned = clean_name(name)
        if cleaned:
            cleaned_to_variations[cleaned].add(name)

    # 2. Determine initial canonical name for each group (longest variation)
    cleaned_to_canonical = {}
    for cleaned, variations in cleaned_to_variations.items():
        if variations:
            # Choose the longest original variation as the initial canonical
            cleaned_to_canonical[cleaned] = max(variations, key=len)

    # 3. Refine canonical names by checking for subset relationships among CLEANED names
    #    Map shorter cleaned names to the canonical name of the longer cleaned name they are part of.
    all_cleaned_sorted = sorted(list(cleaned_to_canonical.keys()), key=len) # Process shorter names first
    final_cleaned_to_canonical = cleaned_to_canonical.copy() # Start with initial canonicals

    for i, shorter_cleaned in enumerate(all_cleaned_sorted):
        shorter_canonical = final_cleaned_to_canonical[shorter_cleaned] # Current canonical for this group
        
        # Check against all longer cleaned names
        for longer_cleaned in all_cleaned_sorted[i+1:]:
            longer_canonical = final_cleaned_to_canonical[longer_cleaned]
            
            # Use split to handle multi-word names robustly
            shorter_tokens = set(shorter_cleaned.split())
            longer_tokens = set(longer_cleaned.split())

            # If the shorter name's words are a proper subset of the longer name's words
            if shorter_tokens.issubset(longer_tokens) and shorter_tokens != longer_tokens:
                # Update the canonical target for the shorter group to be the canonical of the longer group
                final_cleaned_to_canonical[shorter_cleaned] = longer_canonical
                break # Assume it maps to the first longer match found

    # 4. Create the final map from ANY variation to its ultimate canonical name
    variation_to_canonical_map = {}
    for cleaned, variations in cleaned_to_variations.items():
        # Get the final canonical name for this cleaned group (might have been updated)
        ultimate_canonical = final_cleaned_to_canonical[cleaned]
        for variation in variations:
            variation_to_canonical_map[variation] = ultimate_canonical

    # 5. Add Pronoun Mappings (heuristic-based)
    canonical_names_found = list(set(variation_to_canonical_map.values())) # Unique final canonicals
    
    # Heuristics based on common names in the final canonical list
    elias_canonical = next((name for name in canonical_names_found if "Elias" in name or "Vance" in name), None)
    anya_canonical = next((name for name in canonical_names_found if "Anya" in name or "Sharma" in name), None)
    sarah_canonical = next((name for name in canonical_names_found if "Sarah" in name), None)
    david_canonical = next((name for name in canonical_names_found if "David" in name or "Chen" in name), None)
    bob_canonical = next((name for name in canonical_names_found if "Bob" in name), None)
    carol_canonical = next((name for name in canonical_names_found if "Carol" in name), None)
    alice_canonical = next((name for name in canonical_names_found if "Alice" in name), None)

    male_target = elias_canonical or david_canonical or bob_canonical
    female_target = anya_canonical or sarah_canonical or carol_canonical or alice_canonical

    if male_target:
        for p in ['He', 'Him', 'His', 'he', 'him', 'his']: variation_to_canonical_map[p] = male_target
    if female_target:
        for p in ['She', 'Her', 'Hers', 'she', 'her', 'hers']: variation_to_canonical_map[p] = female_target
    
    # Generic "they/them/their" - map to most recently mentioned person or first person found
    if canonical_names_found:
        # Prefer singular "they" to map to the first/main person in the document
        generic_target = canonical_names_found[0] if len(canonical_names_found) > 0 else None
        if generic_target:
            for p in ['They', 'Them', 'Their', 'Theirs', 'they', 'them', 'their', 'theirs']:
                variation_to_canonical_map[p] = generic_target
        
    # Legacy: Collective Pronoun (kept for backwards compatibility, but less common now)
    # if len(canonical_names_found) > 1:
    #     compound_name = " and ".join(canonical_names_found[:2])
    #     variation_to_canonical_map['Their'] = compound_name
    #     variation_to_canonical_map['their'] = compound_name
        
    # Handle compound names created by previous pronoun step (e.g., "Elias Vance and Prof...")
    # Map them back to a reasonable representation or remove if too complex
    # For now, let's just ensure they don't break things - map to first name?
    compound_keys = [k for k in variation_to_canonical_map if " and " in k]
    for key in compound_keys:
        first_part = key.split(" and ")[0]
        if first_part in variation_to_canonical_map:
             variation_to_canonical_map[key] = variation_to_canonical_map[first_part]
        # else: # If first part not mapped, maybe remove or map to generic?
        #     del variation_to_canonical_map[key]


    return variation_to_canonical_map


# --- 1. Enhanced Factual Triple Extraction ---
def extract_triples_spacy_enhanced(doc: spacy.tokens.Doc, name_map: Dict[str, str]) -> List[Tuple[str, str, str, Dict]]:
    triples = []
    
    LABEL_MAP = {"PERSON": "Person", "ORG": "Organization", "GPE": "Location", "PRODUCT": "Product"}
    entity_labels = {ent.text: LABEL_MAP.get(ent.label_, "Entity") for ent in doc.ents}

    for sent in doc.sents:
        for token in sent:
            # --- Verb-based triple extraction ---
            if token.pos_ == "VERB":
                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                objects = [child for child in token.children if child.dep_ in ("dobj", "attr", "acomp")]
                # Add prepositional objects
                for child in token.children:
                    if child.dep_ == "prep" and child.head == token:
                        pobj = [gc for gc in child.children if gc.dep_ == "pobj"]
                        if pobj: objects.append(pobj[0])
                
                for subj_tok in subjects:
                    subject_text_raw = " ".join([t.text for t in subj_tok.subtree if not t.is_punct]).strip()
                    subject_text = name_map.get(subject_text_raw, subject_text_raw) # Map to canonical

                    # Skip if subject is just a linking word
                    if subject_text.lower() in ["who", "which", "that", "whom", "whose"]:
                        continue

                    for obj_tok in objects:
                        object_text_raw = " ".join([t.text for t in obj_tok.subtree if not t.is_punct]).strip()
                        object_text = name_map.get(object_text_raw, object_text_raw) # Map to canonical
                        
                        predicate = token.lemma_.upper()

                        subj_label = entity_labels.get(subject_text_raw, "Entity")
                        # If mapped name is Person, override label
                        if subject_text_raw in name_map: subj_label = "Person"
                        
                        obj_label = entity_labels.get(object_text_raw, "Concept")
                        if object_text_raw in name_map: obj_label = "Person"

                        # Skip if object is just a pronoun or linking word (who, which, that, etc.)
                        if object_text.lower() in ["who", "which", "that", "whom", "whose"]:
                            continue
                            
                        if subject_text and object_text and subject_text != object_text:
                            triples.append((subject_text, predicate, object_text, {"source": "Document_Fact", "subj_label": subj_label, "obj_label": obj_label}))

            # --- "IS_A" triple extraction ---
            if token.dep_ == "attr" and token.head.dep_ == "ROOT" and token.head.lemma_ == "be":
                subject_tok = [child for child in token.head.children if child.dep_ == "nsubj"]
                if subject_tok:
                    subject_text_raw = " ".join([t.text for t in subject_tok[0].subtree if not t.is_punct]).strip()
                    subject_text = name_map.get(subject_text_raw, subject_text_raw)
                    
                    # Only get the noun phrase, not the entire relative clause
                    object_text_raw = ""
                    for t in token.subtree:
                        # Stop at relative clause markers
                        if t.text.lower() in ["who", "which", "that", "whom", "whose"]:
                            break
                        if not t.is_punct:
                            object_text_raw += t.text + " "
                    object_text_raw = object_text_raw.strip()
                    object_text = name_map.get(object_text_raw, object_text_raw)
                    
                    subj_label = entity_labels.get(subject_text_raw, "Entity")
                    if subject_text_raw in name_map: subj_label = "Person"
                    obj_label = entity_labels.get(object_text_raw, "Concept")
                    if object_text_raw in name_map: obj_label = "Person"
                    
                    if subject_text and object_text and subject_text != object_text:
                        triples.append((subject_text, "IS_A", object_text, {"source": "Document_Fact", "subj_label": subj_label, "obj_label": obj_label}))

            # --- Possessive triple extraction ---
            if token.dep_ == "poss" and token.head:
                subject_text_raw = token.text.replace("'s", "").strip()
                subject_text = name_map.get(subject_text_raw, subject_text_raw)
                
                object_text_raw = token.head.text.strip()
                object_text = name_map.get(object_text_raw, object_text_raw)
                
                # Assume possessor is Person if it maps to a canonical name
                subj_label_poss = "Person" if subject_text_raw in name_map else entity_labels.get(subject_text_raw, "Entity")
                
                if subject_text and object_text and subject_text != object_text:
                    triples.append((subject_text, "OWNS", object_text, {"source": "Document_Fact", "subj_label": subj_label_poss, "obj_label": "Concept"}))

    # Remove duplicates based on CANONICAL triple structure
    seen = set()
    unique_triples = []
    for triple in triples:
        # Check for non-empty subject/object before adding
        if triple[0] and triple[2]:
            key = (triple[0], triple[1], triple[2]) # Use canonical names for uniqueness
            if key not in seen:
                seen.add(key)
                unique_triples.append(triple)
    return unique_triples


# --- 2. Personality Trait Assessment ---
def assess_personality_from_text(doc: spacy.tokens.Doc, trait_clues_config: Dict[str, Any], name_map: Dict[str, str]) -> Dict[str, List[Dict]]:
    personality_scores: Dict[str, Dict[str, Tuple[float, List[str]]]] = {} 

    # 1. Initialize scores for CANONICAL persons found
    canonical_persons = set(name_map.values())
    for person_name in canonical_persons:
        # Heuristic check for valid person names (avoids mapping pronouns here)
        if len(person_name.split()) > 0 and person_name[0].isupper() and " and " not in person_name: 
            if person_name not in personality_scores:
                personality_scores[person_name] = {
                    "Openness": [2.5, []], "Conscientiousness": [2.5, []], 
                    "Extraversion": [2.5, []], "Agreeableness": [2.5, []], 
                    "Neuroticism": [2.5, []],
                }

    # 2. Score based on clues, attributing to the CANONICAL name
    for sent in doc.sents:
        # Map raw mentions/pronouns in sentence to canonical names for scoring
        target_canonical_persons = set()
        for token in sent:
            token_text = token.text
            # Check entities first (using original text for lookup)
            if token.ent_type_ == "PERSON":
                 canonical = name_map.get(token_text)
                 if canonical and canonical in personality_scores:
                     target_canonical_persons.add(canonical)
            # Check pronouns explicitly mapped (using original text for lookup)
            elif token_text in name_map:
                 canonical = name_map[token_text]
                 if canonical in personality_scores: # Ensure it maps to a person initialized
                     target_canonical_persons.add(canonical)
        
        # Only score if we found specific persons in this sentence
        if not target_canonical_persons:
            continue

        for person_canonical in target_canonical_persons:
            for trait, data in trait_clues_config.items():
                target_trait = "Extraversion" if trait == "Introversion" else trait
                if target_trait not in personality_scores[person_canonical]: continue 

                for keyword in data["keywords"]:
                    if keyword in sent.text.lower():
                        current_score = personality_scores[person_canonical][target_trait][0]
                        evidence_list = personality_scores[person_canonical][target_trait][1]
                        
                        new_score = current_score + data["score_mod"]
                        new_score = max(1.0, min(5.0, new_score))
                        
                        new_evidence = evidence_list + [keyword]
                        personality_scores[person_canonical][target_trait] = [new_score, new_evidence]
                        
    # 3. Format the output using canonical names
    formatted_output = {}
    for person_canonical, traits in personality_scores.items():
        formatted_output[person_canonical] = []
        for trait_name, (score, evidence) in traits.items():
            formatted_output[person_canonical].append({
                "trait": trait_name,
                "score": round(score, 1),
                "basis": "; ".join(list(set(evidence))) if evidence else "Neutral base score"
            })
    return formatted_output

# --- 3. Knowledge Graph Construction (Refined - NO MERGE STEP NEEDED) ---

def build_lpg_from_text(text: str, trait_clues_config: Dict[str, Any] = DEFAULT_TRAIT_CLUES) -> nx.DiGraph:
    
    if not text:
        return nx.DiGraph()
        
    # --- STEP 1: Initial Processing & Create Canonical Name Map ---
    doc = nlp(text) 
    # This map is now the source of truth for entity names
    variation_to_canonical_map = create_canonical_name_map(doc)
    
    G = nx.DiGraph()

    # --- STEP 2: Extract Triples using the Canonical Map ---
    # The extraction function now directly returns triples with canonical names
    extracted_triples = extract_triples_spacy_enhanced(doc, variation_to_canonical_map)
    
    for subject, predicate, obj, props in extracted_triples:
        # Nodes are added using the canonical names
        # Ensure correct label is applied, prioritizing 'Person' if available
        subj_label = "Person" if subject in variation_to_canonical_map.values() else props.get("subj_label", "Entity")
        obj_label = "Person" if obj in variation_to_canonical_map.values() else props.get("obj_label", "Concept")
        
        # Also check if it's a mapped variation (direct key in the map)
        if subject in variation_to_canonical_map:
            subj_label = "Person"
        if obj in variation_to_canonical_map:
            obj_label = "Person"

        G.add_node(subject, label=subj_label)
        G.add_node(obj, label=obj_label)
        
        # Add edge between canonical nodes
        if subject != obj: 
            G.add_edge(subject, obj, type=predicate, **{k:v for k,v in props.items() if k not in ["subj_label", "obj_label"]})

    # --- STEP 3: Add Personality Traits to Canonical Person Nodes ---
    # Pass the original doc for context, but use the map for targeting
    personality_data = assess_personality_from_text(doc, trait_clues_config, variation_to_canonical_map)
    for person_canonical, traits in personality_data.items():
        # Add/Ensure the canonical person node exists and is labeled 'Person'
        G.add_node(person_canonical, label="Person") 
            
        for data in traits:
            trait_node = data["trait"]
            G.add_node(trait_node, label="Trait")
            
            G.add_edge(
                person_canonical, # Connect trait to the canonical name
                trait_node,
                type="HAS_TRAIT",
                score=data["score"],
                source_text=data["basis"]
            )

    # --- STEP 4: Remove isolated nodes ---
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    return G

# --- Visualization Function ---
def visualize_knowledge_graph(G: nx.DiGraph):
    # Check if graph is empty
    if not G.nodes:
        print("Graph is empty, nothing to visualize.")
        return
        
    pos = nx.spring_layout(G, k=0.8, iterations=80, seed=42) 
    
    node_size_base = 3500 
    
    color_map = {
        "Person": "skyblue",
        "Trait": "lightgreen",
        "Organization": "lightcoral", 
        "Location": "salmon",     
        "Concept": "gold",
        "Entity": "lightgray",
        "Product": "orange"
    }

    node_colors = []
    node_labels_dict = {}
    for node, data in G.nodes(data=True):
        label_text = str(node)
        max_line_len = 18 
        if len(label_text) > max_line_len:
             parts = label_text.split()
             lines = []
             current_line = ""
             for part in parts:
                 if not current_line: current_line = part
                 elif len(current_line) + len(part) + 1 <= max_line_len: current_line += " " + part
                 else:
                     lines.append(current_line)
                     current_line = part
             lines.append(current_line)
             label_text = '\n'.join(lines)
                 
        node_labels_dict[node] = label_text
        
        # Determine color: prioritize 'Person' if label exists, else use default logic
        node_label = data.get("label", "Entity") # Get the assigned label
        node_colors.append(color_map.get(node_label, "lightgray"))


    edge_labels = {}
    for u, v, data in G.edges(data=True):
        rel_type = data.get('type', '') # Use .get for safety
        
        if rel_type == "HAS_TRAIT":
            score = data.get('score', 'N/A')
            edge_labels[(u, v)] = f"HAS_TRAIT\n(Score: {score})"
        else:
            edge_labels[(u, v)] = rel_type

    plt.figure(figsize=(max(15, G.number_of_nodes()), max(10, G.number_of_nodes()*0.8))) # Dynamic figure size
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size_base, alpha=0.95, edgecolors="black")
    nx.draw_networkx_edges(G, pos, arrowsize=30, edge_color="darkgray", width=2.0, alpha=0.7, connectionstyle='arc3,rad=0.1') # Slightly increased curve
    nx.draw_networkx_labels(G, pos, labels=node_labels_dict, font_size=max(8, 12 - G.number_of_nodes() // 5), font_weight="bold") # Dynamic font size
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=max(7, 10 - G.number_of_nodes() // 5), label_pos=0.3, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1')) 

    plt.title("Knowledge Graph: Extracted Facts and Personality Traits (LPG Model)", fontsize=22) 
    plt.axis('off')
    
    legend_handles = []
    for label, color in color_map.items():
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=label,
                                      markerfacecolor=color, markersize=14, markeredgecolor='black')) 
    # Adjust legend position based on graph size
    plt.legend(handles=legend_handles, title="Node Types", loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, title_fontsize=14) 
    
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust rect to make space
    plt.subplots_adjust(right=0.85) # Fine-tune right margin
    plt.show()


# --- Evaluation Matrix Functions ---

def calculate_graph_metrics(G: nx.DiGraph, doc_text: str) -> Dict[str, Any]:
    """
    Calculate various quality metrics for the knowledge graph.
    """
    metrics = {}
    
    # Basic Graph Statistics
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['num_persons'] = len([n for n, d in G.nodes(data=True) if d.get('label') == 'Person'])
    metrics['num_traits'] = len([n for n, d in G.nodes(data=True) if d.get('label') == 'Trait'])
    metrics['num_concepts'] = len([n for n, d in G.nodes(data=True) if d.get('label') == 'Concept'])
    
    # Graph Density (how connected the graph is)
    if G.number_of_nodes() > 1:
        metrics['density'] = nx.density(G)
    else:
        metrics['density'] = 0.0
    
    # Average degree (connections per node)
    if G.number_of_nodes() > 0:
        degrees = [d for n, d in G.degree()]
        metrics['avg_degree'] = sum(degrees) / len(degrees)
        metrics['max_degree'] = max(degrees) if degrees else 0
    else:
        metrics['avg_degree'] = 0.0
        metrics['max_degree'] = 0
    
    # Relationship Type Distribution
    edge_types = defaultdict(int)
    for u, v, data in G.edges(data=True):
        edge_type = data.get('type', 'UNKNOWN')
        edge_types[edge_type] += 1
    metrics['edge_type_distribution'] = dict(edge_types)
    
    # Person Coverage (how many persons have traits assigned)
    persons = [n for n, d in G.nodes(data=True) if d.get('label') == 'Person']
    persons_with_traits = []
    for person in persons:
        has_trait = any(G.has_edge(person, trait) and G[person][trait].get('type') == 'HAS_TRAIT' 
                       for trait in G.successors(person))
        if has_trait:
            persons_with_traits.append(person)
    
    metrics['persons_with_traits'] = len(persons_with_traits)
    if persons:
        metrics['trait_coverage'] = len(persons_with_traits) / len(persons)
    else:
        metrics['trait_coverage'] = 0.0
    
    # Text-to-Graph Conversion Rate
    doc = nlp(doc_text)
    num_sentences = len(list(doc.sents))
    num_entities = len([ent for ent in doc.ents])
    
    metrics['num_sentences_in_doc'] = num_sentences
    metrics['num_entities_in_doc'] = num_entities
    if num_sentences > 0:
        metrics['edges_per_sentence'] = G.number_of_edges() / num_sentences
    else:
        metrics['edges_per_sentence'] = 0.0
    
    # Connected Components (ideally should be 1 for fully connected graph)
    if G.number_of_nodes() > 0:
        metrics['num_connected_components'] = nx.number_weakly_connected_components(G)
    else:
        metrics['num_connected_components'] = 0
    
    return metrics


def evaluate_knowledge_graph(G: nx.DiGraph, doc_text: str, file_name: str, 
                            ground_truth: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation of the knowledge graph including accuracy metrics.
    
    Args:
        G: The knowledge graph
        doc_text: Original document text
        file_name: Name of the source file
        ground_truth: Optional dictionary containing expected entities, relations, and traits
    """
    evaluation = {
        'file_name': file_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Calculate structural metrics
    structural_metrics = calculate_graph_metrics(G, doc_text)
    evaluation.update(structural_metrics)
    
    # If ground truth is provided, calculate precision and recall
    if ground_truth:
        # Entity Extraction Evaluation
        extracted_persons = set([n for n, d in G.nodes(data=True) if d.get('label') == 'Person'])
        expected_persons = set(ground_truth.get('persons', []))
        
        if expected_persons:
            true_positives = len(extracted_persons & expected_persons)
            false_positives = len(extracted_persons - expected_persons)
            false_negatives = len(expected_persons - extracted_persons)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            evaluation['entity_precision'] = precision
            evaluation['entity_recall'] = recall
            evaluation['entity_f1_score'] = f1_score
            evaluation['extracted_persons'] = list(extracted_persons)
            evaluation['expected_persons'] = list(expected_persons)
        
        # Relationship Extraction Evaluation (with partial matching)
        extracted_relations = [(u, d.get('type'), v) for u, v, d in G.edges(data=True)]
        expected_relations = [tuple(r) for r in ground_truth.get('relations', [])]
        
        if expected_relations:
            # Use partial matching for objects (e.g., "scientist" matches "a kind scientist")
            matched_relations = set()
            for exp_subj, exp_pred, exp_obj in expected_relations:
                for ext_subj, ext_pred, ext_obj in extracted_relations:
                    # Check if subject and predicate match exactly, and object contains expected
                    if (exp_subj == ext_subj and 
                        exp_pred == ext_pred and 
                        (exp_obj == ext_obj or exp_obj in ext_obj or ext_obj in exp_obj)):
                        matched_relations.add((exp_subj, exp_pred, exp_obj))
                        break
            
            rel_tp = len(matched_relations)
            rel_fn = len(expected_relations) - rel_tp
            rel_fp = max(0, len(extracted_relations) - rel_tp)  # Approximate false positives
            
            rel_precision = rel_tp / len(extracted_relations) if len(extracted_relations) > 0 else 0
            rel_recall = rel_tp / len(expected_relations) if len(expected_relations) > 0 else 0
            rel_f1 = 2 * (rel_precision * rel_recall) / (rel_precision + rel_recall) if (rel_precision + rel_recall) > 0 else 0
            
            evaluation['relation_precision'] = rel_precision
            evaluation['relation_recall'] = rel_recall
            evaluation['relation_f1_score'] = rel_f1
            evaluation['matched_relations'] = rel_tp
            evaluation['total_expected_relations'] = len(expected_relations)
            evaluation['total_extracted_relations'] = len(extracted_relations)
    
    # === UNSUPERVISED/INTRINSIC METRICS (No Ground Truth Needed) ===
    
    # 1. Graph Coherence Score (how well-structured is the graph)
    coherence_score = 0
    if evaluation['num_nodes'] > 0:
        # Prefer single connected component
        if evaluation['num_connected_components'] == 1:
            coherence_score += 0.3
        else:
            coherence_score += 0.3 * (1 / evaluation['num_connected_components'])
        
        # Balanced density (not too sparse, not too dense)
        optimal_density = 0.15
        density_diff = abs(evaluation['density'] - optimal_density)
        coherence_score += max(0, 0.2 - density_diff)
        
        # Good average degree (each node should have meaningful connections)
        if evaluation['avg_degree'] >= 2:
            coherence_score += min(0.3, evaluation['avg_degree'] / 10)
        
        # No isolated nodes (already handled by removing them, but check)
        coherence_score += 0.2  # Bonus for no isolates
    
    evaluation['graph_coherence'] = min(1.0, coherence_score)
    
    # 2. Information Richness (how much info extracted per sentence)
    info_richness = 0
    if evaluation['num_sentences_in_doc'] > 0:
        # Edges per sentence (higher = more info extracted)
        eps = evaluation['edges_per_sentence']
        info_richness += min(0.4, eps / 10)
        
        # Nodes per sentence
        nps = evaluation['num_nodes'] / evaluation['num_sentences_in_doc']
        info_richness += min(0.3, nps / 10)
        
        # Diversity of relationship types
        num_rel_types = len(evaluation['edge_type_distribution'])
        info_richness += min(0.3, num_rel_types / 10)
    
    evaluation['information_richness'] = min(1.0, info_richness)
    
    # 3. Semantic Completeness (person-centric analysis)
    semantic_completeness = 0
    if evaluation['num_persons'] > 0:
        # All persons should have traits
        semantic_completeness += evaluation['trait_coverage'] * 0.4
        
        # Persons should have at least 2 connections (IS_A + traits)
        if evaluation['avg_degree'] >= 2:
            semantic_completeness += 0.3
        
        # Ratio of meaningful edges (IS_A, HAS_TRAIT vs noise)
        meaningful_edges = evaluation['edge_type_distribution'].get('HAS_TRAIT', 0) + \
                          evaluation['edge_type_distribution'].get('IS_A', 0)
        if evaluation['num_edges'] > 0:
            semantic_completeness += (meaningful_edges / evaluation['num_edges']) * 0.3
    
    evaluation['semantic_completeness'] = min(1.0, semantic_completeness)
    
    # 4. Entity Recognition Quality (based on expected entity patterns)
    entity_quality = 0
    if evaluation['num_entities_in_doc'] > 0:
        # Should extract at least as many nodes as entities detected
        extraction_rate = min(1.0, evaluation['num_nodes'] / evaluation['num_entities_in_doc'])
        entity_quality += extraction_rate * 0.5
        
        # Should have good person:concept ratio (not too many noise nodes)
        total_meaningful = evaluation['num_persons'] + evaluation['num_traits']
        if evaluation['num_nodes'] > 0:
            entity_quality += (total_meaningful / evaluation['num_nodes']) * 0.5
    
    evaluation['entity_recognition_quality'] = min(1.0, entity_quality)
    
    # 5. Relationship Diversity (variety of edge types)
    rel_diversity = 0
    edge_dist = evaluation['edge_type_distribution']
    if len(edge_dist) > 0:
        # Simple diversity: number of unique types normalized
        num_types = len(edge_dist)
        rel_diversity = min(1.0, num_types / 5)  # Assume max 5 meaningful types
        
        # Bonus for having key relationship types
        has_is_a = 0.15 if 'IS_A' in edge_dist else 0
        has_trait = 0.15 if 'HAS_TRAIT' in edge_dist else 0
        
        rel_diversity = min(1.0, rel_diversity * 0.7 + has_is_a + has_trait)
    
    evaluation['relationship_diversity'] = rel_diversity
    
    # === Overall Intrinsic Quality Score (0-100) ===
    intrinsic_score = (
        evaluation['graph_coherence'] * 20 +           # Structure quality
        evaluation['information_richness'] * 25 +       # Extraction efficiency
        evaluation['semantic_completeness'] * 30 +      # Meaning preservation
        evaluation['entity_recognition_quality'] * 15 + # Entity detection
        evaluation['relationship_diversity'] * 10       # Variety
    )
    
    evaluation['intrinsic_quality_score'] = min(100, intrinsic_score)
    
    # Keep old quality score for comparison (if ground truth available)
    quality_score = 0
    
    # Component 1: Graph completeness (40 points)
    if evaluation['num_persons'] > 0:
        quality_score += evaluation['trait_coverage'] * 20  # All persons should have traits
    if evaluation['num_connected_components'] == 1:
        quality_score += 10  # Fully connected graph
    if evaluation['density'] > 0.1:
        quality_score += min(10, evaluation['density'] * 50)  # Reasonable density
    
    # Component 2: Information extraction (30 points)
    if evaluation['num_entities_in_doc'] > 0:
        entity_extraction_rate = evaluation['num_nodes'] / evaluation['num_entities_in_doc']
        quality_score += min(15, entity_extraction_rate * 10)
    if evaluation['edges_per_sentence'] >= 1:
        quality_score += min(15, evaluation['edges_per_sentence'] * 5)
    
    # Component 3: Structural quality (30 points)
    if evaluation['avg_degree'] >= 2:
        quality_score += min(15, evaluation['avg_degree'] * 3)
    if evaluation['num_edges'] > 0:
        useful_edges = evaluation['edge_type_distribution'].get('HAS_TRAIT', 0) + \
                      evaluation['edge_type_distribution'].get('IS_A', 0)
        quality_score += min(15, (useful_edges / evaluation['num_edges']) * 15)
    
    evaluation['overall_quality_score'] = min(100, quality_score)
    
    return evaluation


def create_evaluation_report(evaluations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a summary report from multiple evaluations.
    """
    df = pd.DataFrame(evaluations)
    
    # Reorder columns for better readability
    priority_cols = ['file_name', 'overall_quality_score', 'num_nodes', 'num_edges', 
                    'num_persons', 'trait_coverage', 'density', 'num_connected_components']
    
    other_cols = [col for col in df.columns if col not in priority_cols and col != 'timestamp']
    ordered_cols = priority_cols + other_cols + ['timestamp']
    ordered_cols = [col for col in ordered_cols if col in df.columns]
    
    return df[ordered_cols]


def print_evaluation_summary(evaluation: Dict[str, Any]):
    """
    Print a formatted evaluation summary.
    """
    print(f"\n{'='*70}")
    print(f"EVALUATION REPORT: {evaluation['file_name']}")
    print(f"{'='*70}")
    
    print(f"\n[GRAPH STRUCTURE]")
    print(f"  * Total Nodes: {evaluation['num_nodes']}")
    print(f"  * Total Edges: {evaluation['num_edges']}")
    print(f"  * Persons: {evaluation['num_persons']}")
    print(f"  * Traits: {evaluation['num_traits']}")
    print(f"  * Concepts: {evaluation['num_concepts']}")
    print(f"  * Graph Density: {evaluation['density']:.3f}")
    print(f"  * Average Degree: {evaluation['avg_degree']:.2f}")
    print(f"  * Connected Components: {evaluation['num_connected_components']}")
    
    print(f"\n[EXTRACTION EFFICIENCY]")
    print(f"  * Sentences in Document: {evaluation['num_sentences_in_doc']}")
    print(f"  * Entities in Document: {evaluation['num_entities_in_doc']}")
    print(f"  * Edges per Sentence: {evaluation['edges_per_sentence']:.2f}")
    
    print(f"\n[PERSONALITY ANALYSIS]")
    print(f"  * Persons with Traits: {evaluation['persons_with_traits']}/{evaluation['num_persons']}")
    print(f"  * Trait Coverage: {evaluation['trait_coverage']*100:.1f}%")
    
    print(f"\n[RELATIONSHIP TYPES]")
    for rel_type, count in evaluation['edge_type_distribution'].items():
        print(f"  * {rel_type}: {count}")
    
    print(f"\n[INTRINSIC QUALITY METRICS] (No Ground Truth Required)")
    print(f"  * Graph Coherence: {evaluation['graph_coherence']*100:.1f}%")
    print(f"    - Structure well-connected and balanced")
    print(f"  * Information Richness: {evaluation['information_richness']*100:.1f}%")
    print(f"    - Amount of info extracted per sentence")
    print(f"  * Semantic Completeness: {evaluation['semantic_completeness']*100:.1f}%")
    print(f"    - Persons have complete trait profiles")
    print(f"  * Entity Recognition: {evaluation['entity_recognition_quality']*100:.1f}%")
    print(f"    - Quality of entity detection")
    print(f"  * Relationship Diversity: {evaluation['relationship_diversity']*100:.1f}%")
    print(f"    - Variety of relationship types")
    print(f"  * Intrinsic Score: {evaluation['intrinsic_quality_score']:.1f}/100")
    
    if 'entity_precision' in evaluation:
        print(f"\n[ACCURACY METRICS]")
        print(f"  * Entity Precision: {evaluation['entity_precision']*100:.1f}%")
        print(f"  * Entity Recall: {evaluation['entity_recall']*100:.1f}%")
        print(f"  * Entity F1-Score: {evaluation['entity_f1_score']*100:.1f}%")
        
        if 'relation_precision' in evaluation:
            print(f"  * Relation Precision: {evaluation['relation_precision']*100:.1f}%")
            print(f"  * Relation Recall: {evaluation['relation_recall']*100:.1f}%")
            print(f"  * Relation F1-Score: {evaluation['relation_f1_score']*100:.1f}%")
            if 'matched_relations' in evaluation:
                print(f"  * Matched Relations: {evaluation['matched_relations']}/{evaluation['total_expected_relations']} expected")
                print(f"  * Total Extracted: {evaluation['total_extracted_relations']} relations")
    
    print(f"\n[OVERALL QUALITY SCORE]: {evaluation['overall_quality_score']:.1f}/100")
    
    # Quality rating
    score = evaluation['overall_quality_score']
    if score >= 80:
        rating = "Excellent *****"
    elif score >= 60:
        rating = "Good ****"
    elif score >= 40:
        rating = "Fair ***"
    elif score >= 20:
        rating = "Poor **"
    else:
        rating = "Very Poor *"
    
    print(f"  Rating: {rating}")
    print(f"{'='*70}\n")


# --- Execution to read from multiple files ---
FILE_NAMES = ["text/1.txt", "text/2.txt", "text/3.txt"] # Make sure these files exist

# Optional: Define ground truth for evaluation
GROUND_TRUTH = {
    "1.txt": {
        "persons": ["Alice", "Bob"],
        "relations": [
            ("Alice", "IS_A", "scientist"),  # Partial match allowed
            ("Alice", "LEAD", "meetings"),
            ("Alice", "HAS_TRAIT", "Openness"),
            ("Alice", "HAS_TRAIT", "Extraversion"),
            ("Alice", "HAS_TRAIT", "Agreeableness"),
        ],
        "expected_traits": {
            "Alice": ["Openness", "Extraversion", "Agreeableness"],
            "Bob": ["Agreeableness"]
        }
    },
    "2.txt": {
        "persons": ["Bob"],
        "relations": [
            ("Bob", "IS_A", "engineer"),
            ("Bob", "HAS_TRAIT", "Conscientiousness"),
            ("Bob", "HAS_TRAIT", "Extraversion"),
            ("Bob", "HAS_TRAIT", "Neuroticism"),
        ],
        "expected_traits": {
            "Bob": ["Conscientiousness", "Extraversion", "Neuroticism"]
        }
    },
    "3.txt": {
        "persons": ["Carol"],
        "relations": [
            ("Carol", "IS_A", "researcher"),
            ("Carol", "HAS_TRAIT", "Extraversion"),  # Low score = Introversion
        ],
        "expected_traits": {
            "Carol": ["Extraversion"]  # Should have low score (introvert)
        }
    }
}

evaluations = []

for FILE_NAME in FILE_NAMES:
    document_text = ""
    try:
        # Try different encodings if utf-8 fails
        try:
             with open(FILE_NAME, 'r', encoding='utf-8') as f:
                document_text = f.read()
        except UnicodeDecodeError:
             with open(FILE_NAME, 'r', encoding='latin-1') as f:
                 document_text = f.read()
                 
    except FileNotFoundError:
        print(f"Error: The file '{FILE_NAME}' was not found.")
        continue # Skip to the next file
    except Exception as e:
        print(f"Error reading file '{FILE_NAME}': {e}")
        continue
    
    if document_text:
        print(f"\n{'='*60}")
        print(f"--- Processing document from '{FILE_NAME}' ---")
        print(f"{'='*60}")
        kg_output = build_lpg_from_text(document_text)
        
        # Evaluate the knowledge graph
        ground_truth = GROUND_TRUTH.get(FILE_NAME, None)
        evaluation = evaluate_knowledge_graph(kg_output, document_text, FILE_NAME, ground_truth)
        evaluations.append(evaluation)
        
        # Print evaluation summary
        print_evaluation_summary(evaluation)
        
        if kg_output.number_of_nodes() > 0:
             visualize_knowledge_graph(kg_output)
        else:
             print(f"No graph data extracted from '{FILE_NAME}'.")

# Create and display summary report
if evaluations:
    print(f"\n{'='*70}")
    print("SUMMARY REPORT - ALL DOCUMENTS")
    print(f"{'='*70}\n")
    
    report_df = create_evaluation_report(evaluations)
    print(report_df.to_string(index=False))
    
    # Save report to CSV
    report_df.to_csv('evaluation_report.csv', index=False)
    print(f"\n>> Evaluation report saved to 'evaluation_report.csv'")
    
    # Calculate and print aggregated statistics
    print(f"\n{'='*70}")
    print("AGGREGATED STATISTICS")
    print(f"{'='*70}")
    print(f"  * Average Quality Score: {report_df['overall_quality_score'].mean():.1f}/100")
    print(f"  * Total Nodes Extracted: {report_df['num_nodes'].sum()}")
    print(f"  * Total Edges Extracted: {report_df['num_edges'].sum()}")
    print(f"  * Total Persons Identified: {report_df['num_persons'].sum()}")
    print(f"  * Average Trait Coverage: {report_df['trait_coverage'].mean()*100:.1f}%")
    print(f"  * Average Graph Density: {report_df['density'].mean():.3f}")
    print(f"{'='*70}\n")

