import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import tqdm
from dgl.data.utils import load_graphs
import json
import os
from sklearn.metrics import roc_auc_score
import dgl.function as fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def negative_sampling(graph, num_samples):
    neg_src = []
    neg_dst = []
    for _ in range(num_samples):
        u = np.random.randint(0, graph.number_of_nodes())
        v = np.random.randint(0, graph.number_of_nodes())
        while graph.has_edges_between(u, v):
            u = np.random.randint(0, graph.number_of_nodes())
            v = np.random.randint(0, graph.number_of_nodes())
        neg_src.append(u)
        neg_dst.append(v)
    return torch.tensor(neg_src).to(device), torch.tensor(neg_dst).to(device)

def train_test_graph(graph, train_pos_src = None, train_pos_dst = None, test_pos_src = None, test_pos_dst = None):
    try:
        train_graph, _ = dgl.load_graphs("/home/xzhoucs/my_files/social_computing_group/link_prediction/train_graph.bin")
        test_graph, _ = dgl.load_graphs("/home/xzhoucs/my_files/social_computing_group/link_prediction/test_graph.bin")
        print("load train_graph and test_graph success.")
        print("train_graph number of nodes:", train_graph[0].number_of_nodes(), "train_graph number of edges:", train_graph[0].number_of_edges())
        print("test_graph number of nodes:", test_graph[0].number_of_nodes(), "test_graph number of edges:", test_graph[0].number_of_edges())
        return train_graph[0], test_graph[0]
    except:
        print("start sampling...")
        test_neg_src, test_neg_dst = negative_sampling(graph, len(test_pos_src))
        train_neg_src, train_neg_dst = negative_sampling(graph, len(train_pos_src))

        train_graph = dgl.graph((torch.cat([train_pos_src, train_neg_src]), torch.cat([train_pos_dst, train_neg_dst])), num_nodes=graph.number_of_nodes())
        train_graph.edata['label'] = torch.cat([torch.ones(len(train_pos_src)), torch.zeros(len(train_neg_src))])

        test_graph = dgl.graph((torch.cat([test_pos_src, test_neg_src]), torch.cat([test_pos_dst, test_neg_dst])), num_nodes=graph.number_of_nodes())
        test_graph.edata['label'] = torch.cat([torch.ones(len(test_pos_src)), torch.zeros(len(test_neg_src))])
        
        # save train_graph and test_graph
        try:
            dgl.save_graphs("/home/xzhoucs/my_files/social_computing_group/train_graph.bin", train_graph)
            dgl.save_graphs("/home/xzhoucs/my_files/social_computing_group/test_graph.bin", test_graph)
            print("save train_graph and test_graph success.")
            print("train_graph number of nodes:", train_graph.number_of_nodes(), "train_graph number of edges:", train_graph.number_of_edges())
            print("test_graph number of nodes:", test_graph.number_of_nodes(), "test_graph number of edges:", test_graph.number_of_edges())
        except:
            print("save train_graph and test_graph failed.")
        return train_graph, test_graph

def load_graph_metrics(file_path):
    try:
        with open(file_path + "/clustering_coeff.json", 'r') as f:
            clustering_coeff = json.load(f)
        clustering_coeff = dict(sorted(clustering_coeff.items(), key=lambda item: int(item[0])))
    except FileNotFoundError:
        print(f"File {file_path + '/clustering_coeff.json'} not found.")
        return None
    
    try:
        with open(file_path + "/degree_centrality.json", 'r') as f:
            degree_centrality = json.load(f)
        degree_centrality = dict(sorted(degree_centrality.items(), key=lambda item: int(item[0])))
    except FileNotFoundError:
        print(f"File {file_path + '/degree_centrality.json'} not found.")
        return None
    
    try:
        with open(file_path + "/betweenness_centrality.json", 'r') as f:
            betweenness_centrality = json.load(f)
        betweenness_centrality = dict(sorted(betweenness_centrality.items(), key=lambda item: int(item[0])))
    except FileNotFoundError:
        print(f"File {file_path + '/betweenness_centrality.json'} not found.")
        return None
    
    try:
        with open(file_path + "/eigenvector_centrality.json", 'r') as f:
            eigenvector_centrality = json.load(f)
        eigenvector_centrality = dict(sorted(eigenvector_centrality.items(), key=lambda item: int(item[0])))
    except FileNotFoundError:
        print(f"File {file_path + '/eigenvector_centrality.json'} not found.")
        return None
    
    try:
        with open(file_path + "/degree.json", 'r') as f:
            degree = json.load(f)
        degree = dict(sorted(degree.items(), key=lambda item: int(item[0])))
    except FileNotFoundError:
        print(f"File {file_path + '/degree.json'} not found.")
        return None
    
    return clustering_coeff, degree_centrality, betweenness_centrality, eigenvector_centrality, degree
        
class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = dgl.nn.SAGEConv(in_features, hidden_features, 'mean')
        self.conv2 = dgl.nn.SAGEConv(hidden_features, hidden_features, 'mean')
        self.conv3 = dgl.nn.SAGEConv(hidden_features, out_features, 'mean')
        self.pred = DotProductPredictor()
    def forward(self, g, x):
        h = self.conv1(g, x)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        # h = self.conv3(g, h)
        h = self.pred(g, h)
        return h
    
def compute_loss(scores, labels):
    pred_pos = scores[:scores.shape[0] // 2]
    pred_neg = scores[scores.shape[0] // 2:]
    # labels = torch.cat([torch.ones_like(pred_pos), torch.zeros_like(pred_neg)], dim=0)
    labels = labels.float().unsqueeze(1)
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    return loss

def compute_auc(scores, labels):
    pred_pos = scores[:scores.shape[0] // 2]
    pred_neg = scores[scores.shape[0] // 2:]
    # labels = torch.cat([torch.ones_like(pred_pos), torch.zeros_like(pred_neg)], dim=0)
    labels = labels.float().unsqueeze(1)
    auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().detach().numpy())
    return auc
def generate_node_feature(graph, clustering_coeff, degree_centrality, betweenness_centrality, eigenvector_centrality, degree):
    # features = graph.ndata['feature'].to(device)
    features = torch.eye(graph.number_of_nodes()).to(device)
    clustering_coeff = torch.tensor(list(clustering_coeff.values())).to(device)
    degree_centrality = torch.tensor(list(degree_centrality.values())).to(device)
    betweenness_centrality = torch.tensor(list(betweenness_centrality.values())).to(device)
    eigenvector_centrality = torch.tensor(list(eigenvector_centrality.values())).to(device)
    degree = torch.tensor(list(degree.values())).to(device)
    print("before cat features shape:", features.shape)
    features = torch.cat([features, clustering_coeff.unsqueeze(1), degree_centrality.unsqueeze(1), betweenness_centrality.unsqueeze(1), eigenvector_centrality.unsqueeze(1), degree.unsqueeze(1)], dim=1)
    print("after cat features shape:", features.shape)
    return features

def train_model(model, train_graph, optimizer, num_epochs=400):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        scores = model(train_graph, train_graph.ndata['feat'])
        # scores.shape : torch.Size([100100, 1])
        loss = compute_loss(scores, train_graph.edata['label'])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
    return model

def test_model(model, test_graph):
    model.eval()
    with torch.no_grad():
        scores = model(test_graph, test_graph.ndata['feat'])
        # scores.shape : torch.Size([98346, 1])
        auc = compute_auc(scores, test_graph.edata['label'])
        print(f"Test AUC: {auc:.4f}")
    return auc
# train_graph number of nodes: 7624
# train_graph number of edges: 100100
# test_graph number of nodes: 7624
# test_graph number of edges: 11124


def main(graph_path, metrics_path, model_path):
    graph = load_graphs(graph_path)[0][0]
    print("number of nodes:", graph.num_nodes())
    print("number of edges:", graph.num_edges())
    
    # Set random seed for reproducibility
    torch.manual_seed(5008)
    np.random.seed(5008)
    
    # train_data, test_data = train_test_split(
    # np.arange(graph.number_of_edges()), 
    # test_size=0.1, random_state=42)
    # train_mask = torch.zeros(graph.number_of_edges(), dtype=torch.bool).to(device)
    # train_mask[train_data] = True
    # test_mask = torch.zeros(graph.number_of_edges(), dtype=torch.bool).to(device)
    # test_mask[test_data] = True
    # graph.edata['train_mask'] = train_mask
    # graph.edata['test_mask'] = test_mask
    # src, dst = graph.edges()
    # test_pos_src, test_pos_dst = src[test_mask], dst[test_mask]
    # train_pos_src, train_pos_dst = src[train_mask], dst[train_mask]
    train_graph, test_graph = train_test_graph(graph)
    clustering_coeff, degree_centrality, betweenness_centrality, eigenvector_centrality, degree = load_graph_metrics(metrics_path)
    features = generate_node_feature(graph, clustering_coeff, degree_centrality, betweenness_centrality, eigenvector_centrality, degree)
    
    train_graph = train_graph.to(device)
    test_graph = test_graph.to(device)
    train_graph.ndata['feat'] = features
    test_graph.ndata['feat'] = features
    features_dim = features.shape[1]
    hidden_features = 256
    out_features = 256
    model = Model(features_dim, hidden_features, out_features).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model = train_model(model, train_graph, optimizer, num_epochs=1600)
    
    test_auc = test_model(model, test_graph)
    
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    graph_path = "/home/xzhoucs/my_files/social_computing_group/dgl_graph.bin"
    metrics_path = "/home/xzhoucs/my_files/social_computing_group"
    model_path = "/home/xzhoucs/my_files/social_computing_group/link_prediction/link_prediction_model.pth"
    main(graph_path, metrics_path, model_path)