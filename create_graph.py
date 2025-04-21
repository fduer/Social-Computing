import json
import os
import pandas as pd
import networkx as nx
from tqdm import tqdm
import numpy as np
import pickle


def create_graph_from_json(csv_file, target_file, feature_file,  output_file):
    """
    Create a graph from a JSON file containing user interactions.

    Args:
        csv_file (str): Path to the CSV file containing user interactions.
        target_file (str): Path to the CSV file containing user labels.
        feature_file (str): Path to the JSON file containing user features.
        output_file (str): Path to save the generated graph in GraphML format.
    """
    try:
        with open(output_file, 'rb') as f:
            graph = pickle.load(f)
            print("load graph successfully.")
    except:
        # Load the JSON data
        df = pd.read_csv(csv_file, header=0)
        labels = pd.read_csv(target_file, header=0)

        # create un-directed graph
        G = nx.Graph()
        # Add nodes and edges to the graph
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc = "adding edges"):
            G.add_edge(row['node_1'], row['node_2'])
        # Add labels to the nodes
        for _, row in tqdm(labels.iterrows(), total=labels.shape[0], desc = "adding node labels"):
            G.nodes[row['id']]['label'] = row['target']
        # Add features to the nodes
        with open(feature_file, 'r') as f:
            features = json.load(f)
        feature_set = set()
        for node, feature in features.items():
            for f in feature:
                feature_set.add(f)
        for node, feature in features.items():
            node = int(node)
            G.nodes[node]['feature'] = np.zeros(len(feature_set))
            G.nodes[node]['feature'][feature] = 1
            G.nodes[node]['feature'] = G.nodes[node]['feature'].tolist()
        # Add node attributes
        for node in G.nodes():
            G.nodes[node]['degree'] = G.degree(node)
        # Save the graph to a file
        nx.write_graphml(G, output_file)
        print(G.nodes[0]['feature'].shape, G.nodes[0]['feature'])
        print(f"Graph saved to {output_file}")
    

def compute_graph_metric(graph, output_file):
    """
    Compute graph metrics and save them to a CSV file.

    Args:
        graph (nx.Graph): The input graph.
        output_file (str): Path to the output CSV file.
    """
    # compute the clustering_coefficient
    try:
        with open(output_file + "/clustering_coeff.json", 'r') as f:
            clustering_coeff = json.load(f)
        print("loading clustering_coeff.json successfully.")
    except:
        print("computing clustering coefficient...")
        clustering_coeff = nx.clustering(graph)
        max_coeff = max(clustering_coeff.values())
        min_coeff = min(clustering_coeff.values())
        clustering_coeff = {str(k): (v - min_coeff) / (max_coeff - min_coeff) for k, v in clustering_coeff.items()}
        with open(output_file + "/clustering_coeff.json", 'w') as f:
            json.dump(clustering_coeff, f)
        print("clustering_coeff.json created successfully.")
        
    # compute the degree_centrality
    try:
        with open(output_file + "/degree_centrality.json", 'r') as f:
            degree_centrality = json.load(f)
        print("loading degree_centrality.json successfully.")
    except:
        print("computing degree centrality...")
        degree_centrality = nx.degree_centrality(graph)
        max_degree = max(degree_centrality.values())
        min_degree = min(degree_centrality.values())
        degree_centrality = {str(k): (v - min_degree) / (max_degree - min_degree) for k, v in degree_centrality.items()}
        with open(output_file + "/degree_centrality.json", 'w') as f:
            json.dump(degree_centrality, f)
        print("degree_centrality.json created successfully.")
        
    # compute the betweenness_centrality
    try:
        with open(output_file + "/betweenness_centrality.json", 'r') as f:
            betweenness_centrality = json.load(f)
        print("loading betweenness_centrality.json successfully.")
    except:
        print("computing betweenness centrality...")
        betweenness_centrality = nx.betweenness_centrality(graph, normalized=True, k = graph.number_of_nodes()//10)
        betweenness_centrality = {str(k): v for k, v in betweenness_centrality.items()}
        with open(output_file + "/betweenness_centrality.json", 'w') as f:
            json.dump(betweenness_centrality, f)
        print("betweenness_centrality.json created successfully.")
        
    # compute the eigenvector_centrality
    try:
        with open(output_file + "/eigenvector_centrality.json", 'r') as f:
            eigenvector_centrality = json.load(f)
        print("loading eigenvector_centrality.json successfully.")
    except:
        print("computing eigenvector centrality...")
        eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=100)
        max_eigen = max(eigenvector_centrality.values())
        min_eigen = min(eigenvector_centrality.values())
        eigenvector_centrality = {str(k): (v - min_eigen) / (max_eigen - min_eigen) for k, v in eigenvector_centrality.items()}
        with open(output_file + "/eigenvector_centrality.json", 'w') as f:
            json.dump(eigenvector_centrality, f)
        print("eigenvector_centrality.json created successfully.")

def main():
    # Define the paths to the input files
    csv_file = r'/root/social_computing_group/lasftm_asia/lastfm_asia_edges.csv'
    target_file = r'/root/social_computing_group/lasftm_asia/lastfm_asia_target.csv'
    feature_file = r'/root/social_computing_group/lasftm_asia/lastfm_asia_features.json'
    output_file = r'/root/social_computing_group/graph.pkl'

    # Create the graph
    create_graph_from_json(csv_file, target_file, feature_file, output_file)

    # Load the graph from the file
    with open(output_file, 'rb') as f:
        graph = pickle.load(f)

    # Compute and save graph metrics
    compute_graph_metric(graph, r'/root/social_computing_group')

if __name__ == "__main__":
    main()
    