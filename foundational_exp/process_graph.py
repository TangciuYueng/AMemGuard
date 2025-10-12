import pickle
from collections import defaultdict

def load_graph_documents(filename):
    with open(filename, 'rb') as f: 
        graph_documents = pickle.load(f)
    return graph_documents

def merge_graph_documents(graph_documents):
    all_nodes = {}
    all_relationships = []
    node_sources = {}

    for g in graph_documents:
        source_meta = g.source.metadata if hasattr(g, "source") and hasattr(g.source, "metadata") else {}
        for node in g.nodes:
            key = (node.id, node.type)
            if key not in all_nodes:
                all_nodes[key] = node
            node_sources[key] = source_meta

        for rel in g.relationships:
            key = (rel.source.id, rel.target.id, rel.type)
            if key not in {(r.source.id, r.target.id, r.type) for r in all_relationships}:
                all_relationships.append(rel)

    return list(all_nodes.values()), all_relationships, node_sources

def merge_graph_documents_by_label(graph_documents, label=1):
    all_nodes = {}
    all_relationships = []
    node_sources = {}

    for g in graph_documents:
        source_meta = g.source.metadata if hasattr(g, "source") and hasattr(g.source, "metadata") else {}
        for node in g.nodes:
            if source_meta['label'] != label:
                continue
            key = (node.id, node.type)
            if key not in all_nodes:
                all_nodes[key] = node
            node_sources[key] = source_meta

        for rel in g.relationships:
            key = (rel.source.id, rel.target.id, rel.type)
            if key not in {(r.source.id, r.target.id, r.type) for r in all_relationships}:
                all_relationships.append(rel)

    return list(all_nodes.values()), all_relationships, node_sources

filename = 'graph_documents.pkl'
graph_documents = load_graph_documents(filename)

all_nodes, all_relationships, node_sources = merge_graph_documents(graph_documents)

output_filename = 'merged_graph.pkl'
with open(output_filename, 'wb') as f:
    pickle.dump((all_nodes, all_relationships, node_sources), f)

all_nodes_label_1, all_relationships_label_1, node_sources_label_1 = merge_graph_documents_by_label(graph_documents, label=1)
output_filename_label_1 = 'merged_graph_label_1.pkl'
with open(output_filename_label_1, 'wb') as f:
    pickle.dump((all_nodes_label_1, all_relationships_label_1, node_sources_label_1), f)

all_nodes_label_0, all_relationships_label_0, node_sources_label_0 = merge_graph_documents_by_label(graph_documents, label=0)
output_filename_label_0 = 'merged_graph_label_0.pkl'
with open(output_filename_label_0, 'wb') as f:
    pickle.dump((all_nodes_label_0, all_relationships_label_0, node_sources_label_0), f)