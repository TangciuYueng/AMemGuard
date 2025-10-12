import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Set, Tuple, Any
import traceback 

SCENARIOS = [
    "Web Browse", "Task Automation", "Financial Operations", 
    "Support, Evaluation & Diagnosis", "Information Retrieval & Analysis", 
    "Email Management", "Security & Access Management", 
    "Content Publishing & Communication", "Software Development & Support", 
    "Social Media Management", "Health & Wellness Support", "Data Management", 
    "Autonomous Navigation & Robotics", "Planning, Scheduling & Optimization", 
    "IT System & Network Operations", "Content Creation & Processing", 
    "Device & Environment Control"
]
OUTPUT_DIR = './graph_edge_networkx'

def load_graph_documents(filename: str):
   
    with open(filename, 'rb') as f:
        graph_documents = pickle.load(f)
    return graph_documents

def draw_graph(G: nx.DiGraph, scenario: str, node_labels: Dict[Tuple, Set]):
    
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50) 
    plt.figure(figsize=(18, 14))

    color_label1_only = '#F6997B' 
    color_label0_only = '#92D4BE' 
    color_both_labels = '#FF0000' 
    color_other = '#BDBDBD'      
    
    red_edges, blue_edges, green_edges, other_edges = [], [], [], []
    for u, v, data in G.edges(data=True):
        labels = data.get('labels', set())
        is_positive = 1 in labels
        is_negative = 0 in labels

        if is_positive and is_negative:
            green_edges.append((u, v))
        elif is_positive:
            red_edges.append((u, v))
        elif is_negative:
            blue_edges.append((u, v))
        else:
            other_edges.append((u, v))

    total_edges = len(G.edges)
    print(f"--- Scenario: {scenario} ---")
    print(f"Total nodes in graph: {len(G.nodes())}")
    print(f"Total edges: {total_edges}")
    if total_edges > 0:
        print(f"  - Edges only in label=1 (Soft Red): {len(red_edges)} ({len(red_edges)/total_edges:.2%})")
        print(f"  - Edges only in label=0 (Soft Blue): {len(blue_edges)} ({len(blue_edges)/total_edges:.2%})")
        print(f"  - Edges in both (Soft Purple): {len(green_edges)} ({len(green_edges)/total_edges:.2%})")
        print(f"  - Other edges (Gray): {len(other_edges)} ({len(other_edges)/total_edges:.2%})")

    node_size = 180 
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color='#cccccc', node_size=node_size, alpha=0.9)

    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color=color_label1_only, width=1.0, arrowstyle='->', arrowsize=10, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=blue_edges, edge_color=color_label0_only, width=1.0, arrowstyle='->', arrowsize=10, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=green_edges, edge_color=color_both_labels, width=2.5, arrowstyle='->', arrowsize=10, alpha=0.8)

    edge_patches = [
        mpatches.Patch(color=color_label1_only, label=f'Malicious ({len(red_edges)})'),
        mpatches.Patch(color=color_label0_only, label=f'Benign ({len(blue_edges)})'),
        mpatches.Patch(color=color_both_labels, label=f'Overlap ({len(green_edges)})'),
    ]
    
    plt.legend(
        handles=edge_patches, 
        loc='upper center',         
        bbox_to_anchor=(0.5, 0.01), 
        fontsize=24, 
        ncol=3                      
    )
    plt.axis('off')
    plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    safe_scenario_name = scenario.replace(" ", "_").replace("/", "_").replace("&", "and")
    output_filename = os.path.join(OUTPUT_DIR, f'graph_edge_{safe_scenario_name}.pdf')
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Graph saved to {output_filename}\n")


def merge_graph_documents(graph_documents, scenario: str) -> Tuple[nx.DiGraph, Dict[Tuple, Set]]:
    G = nx.DiGraph()
    node_labels: Dict[Tuple, Set] = {}
    
    for g in graph_documents:
        source_meta = g.source.metadata if hasattr(g, "source") and hasattr(g.source, "metadata") else {}
        
        if scenario != source_meta.get('application_scenario'):
            continue
            
        label = source_meta.get('label', None)
        
        for node in g.nodes:
            key = (node.id, node.type)
            
            if key not in G:
                G.add_node(key, data=node)
            
            if key not in node_labels:
                node_labels[key] = set()
            
            if label is not None:
                node_labels[key].add(label)
        
        for rel in g.relationships:
            src_key = (rel.source.id, rel.source.type)
            tgt_key = (rel.target.id, rel.target.type)

            if not G.has_edge(src_key, tgt_key):
                G.add_edge(src_key, tgt_key, data=rel, labels=set())
            
            if label is not None:
                G[src_key][tgt_key]['labels'].add(label)

    return G, node_labels


def main():
    filename = 'graph_documents.pkl' 
    
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        graph_documents = load_graph_documents(filename)
        
        for scenario in SCENARIOS:
            G, node_labels = merge_graph_documents(graph_documents, scenario)
            
            if not G.nodes():
                continue

            draw_graph(G, scenario, node_labels)
            
    except FileNotFoundError:
        print(f"File '{filename}' not found")
    except Exception as e:
        traceback.print_exc()

if __name__ == '__main__':
    main()