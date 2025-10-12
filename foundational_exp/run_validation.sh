#!/bin/bash

# This script runs the validation experiments for the project.
# Preprocess the data
python extract_raw.py
# Build knowledge graphs with langchain
python get_graph.py
# Merge the knowledge graphs 
python process_graph.py
# Draw the knowledge graphs
python postprocess_network.py