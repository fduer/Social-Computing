import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import json

# 读取 graph.pkl 文件
# import pickle
# with open('/root/social_computing_group/graph.pkl', 'rb') as f:
#     graph = pickle.load(f)
# print("number of nodes:", graph.number_of_nodes())
# print("number of edges:", graph.number_of_edges())

# indice = [node[0] for node in graph.nodes(data=True)]
# print(indice[:10]) # [0, 747, 1, 4257, 2194, 580, 6478, 1222, 5735, 7146]
# 转换为 DGL 图
# dgl_graph = dgl.from_networkx(graph, node_attrs=['feature', 'degree', 'label'])

# dgl.save_graphs('dgl_graph.bin', dgl_graph)
# print("Graph saved successfully.")
# print("number of nodes:", dgl_graph.number_of_nodes())
# print("number of edges:", dgl_graph.number_of_edges())

# 读取dgl_graph.bin文件
graph = dgl.load_graphs('/root/social_computing_group/dgl_graph.bin')[0][0]
print("number of nodes:", graph.number_of_nodes())
print("number of edges:", graph.number_of_edges())
# 打印前10个节点的度数和标签
limit = 10
for node in graph.nodes():
    if limit == 0:
        break
    print("node id:", node.item())
    print("node degree:", graph.ndata['degree'][node].item())
    print("node label:", graph.ndata['label'][node].item())
    limit -= 1
