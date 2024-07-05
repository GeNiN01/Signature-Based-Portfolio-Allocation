import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from utils import *

def network_metrics(graph, network_metric:str, edge_weight:str="weight"):
    """
    Parameters
    ----------
    graph : nx.Graph
        Graph object defined using networkx.
    network_metric : str
        type of network metric used consider.
    edge_weight : str, optional
        edge name feature. The default is "weight". Otherwise put None

    Returns
    -------
    val : float or dictionary
        value of the network metrics consider

    """
    if edge_weight != "weight": 
        edge_weight = None
    if network_metric == "Clustering Coefficient": 
        val = nx.clustering(graph, weight=edge_weight)
    elif network_metric == "Degree Centrality": 
        val = nx.degree_centrality(graph)
    elif network_metric == "Betweenness Centrality": 
        val = nx.betweenness_centrality(graph, weight=edge_weight)
    elif network_metric == "Eigenvector Centrality": 
        val = nx.eigenvector_centrality(graph, weight=edge_weight)
    elif network_metric == "Closeness Centrality": 
        val = nx.closeness_centrality(graph, distance=edge_weight)
    else: 
        print("Networks properties not implemented")
    return val

def plot_graph(df:pd.DataFrame, adjacency_matrix:list):
    """
    Parameters
    ----------
    df : pd.DataFrame
        dataframe 
    adjacency_matrix : list
        adjacency matrix.

    Returns
    -------
    G : nx.Graph()
        return the graph associated to the adjacency matrix and plot the type of graph
    """
    G = nx.Graph() # Create empty graph
    G.add_nodes_from(np.linspace(0, df.shape[1] - 1, df.shape[1], dtype=int)) # Create the node of the graph
    Gx = nx.from_numpy_array(np.matrix(adjacency_matrix))
    edge = nx.edges(Gx) # Find the links among nodes
    G.add_edges_from(edge) # Add links to the graph
    nx.draw(G, with_labels=True) # Plot the results
    plt.show()
    return G

def filtered_matrix(eigenvalues:list, eigenvectors:list, df:pd.DataFrame):
    """
    Parameters
    ----------
    eigenvalues : list
        list of the eigenvalues of a correlation matrix or a similarity matrix.
    eigenvectors : list
        list of the eigenvector associated to the eigenvalues
    df : pd.DataFrame
        dataframe of the original data.

    Returns
    -------
    c_m : list
        market components.
    c_g : list
        significant components.
    This filtering approach is based on the Random Matrix Theory
    """
    max_eigenvalue = eigenvalues[0]
    lambda_max = (1+np.sqrt(df.shape[1]/df.shape[0]))**2
    eigenvalue_list, eigenvector_list = [], []
    for pos, eig in enumerate(eigenvalues):
        if eig>lambda_max:
            eigenvalue_list.append(eig)
            eigenvector_list.append(eigenvectors[pos])
        else:
            break
    if len(eigenvector_list)==0: 
        c_m = 0
    c_g = 0
    for pos, eig in enumerate(eigenvalue_list):
        if pos==0: 
            c_m = (eigenvectors[pos].reshape(-1,1)*eigenvectors[pos].reshape(1,-1))*eig
        else: 
            c_g += (eigenvectors[pos].reshape(-1,1)*eigenvectors[pos].reshape(1,-1))*eig
    return c_m, c_g