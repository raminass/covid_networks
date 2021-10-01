import pandas as pd
import networkx as nx
import numpy as np
import scipy as sp
import math
import os
import joblib

try:
    import mygene
except ImportError:
    os.system('pip install mygene')
finally:
    import mygene
mg = mygene.MyGeneInfo()

# Global Variables
PROPAGATE_ALPHA = 0.9
PROPAGATE_ITERATIONS = 200
PROPAGATE_EPSILON = 10 ** (-4)

def read_network(network_filename):
    network = pd.read_table(network_filename, header=None, usecols=[0, 1, 2])
    return nx.from_pandas_edgelist(network, 0, 1, 2)


def generate_similarity_matrix(network):
    genes = sorted(network.nodes)
    matrix = nx.to_scipy_sparse_matrix(network, genes, weight=2)
    norm_matrix = sp.sparse.diags(1 / sp.sqrt(matrix.sum(0).A1), format="csr")
    matrix = norm_matrix * matrix * norm_matrix
    return PROPAGATE_ALPHA * matrix, genes


def propagate(seeds, matrix, gene_indexes, num_genes):
    F_t = np.zeros(num_genes)
    F_t[[gene_indexes[seed] for seed in seeds if seed in gene_indexes]] = 1
    Y = (1 - PROPAGATE_ALPHA) * F_t

    for _ in range(PROPAGATE_ITERATIONS):
        F_t_1 = F_t
        F_t = matrix.dot(F_t_1) + Y

        if math.sqrt(sp.linalg.norm(F_t_1 - F_t)) < PROPAGATE_EPSILON:
            break

    return F_t


def generate_propagate_data(network, interactors=None):
    matrix, genes = generate_similarity_matrix(network)
    num_genes = len(genes)
    gene_indexes = dict([(gene, index) for (index, gene) in enumerate(genes)])
    if interactors:
        gene_scores = {gene: propagate(
            [gene], matrix, gene_indexes, num_genes) for gene in interactors}
    else:
        gene_scores = {gene: propagate(
            [gene], matrix, gene_indexes, num_genes) for gene in genes}

    return matrix, num_genes, gene_indexes, gene_scores

def load_network_scores(g):
  # propogate Network for all Genes
  network_path = f"artifacts/network_scores.pkl.gz"
  # check if already on disk
  if os.path.exists(network_path):
      print('loading propagated network from disk')
      network_scores = joblib.load(network_path)
      W, num_genes, gene_indexes, gene_scores = (
          network_scores["W"],
          network_scores["num_genes"],
          network_scores["gene_indexes"],
          network_scores["gene_scores"],
      )
  else:
      print('start propagating network')
      W, num_genes, gene_indexes, gene_scores = util.generate_propagate_data(g)
      network_scores = {"W": W, "num_genes": num_genes, "gene_indexes": gene_indexes, "gene_scores": gene_scores}
      joblib.dump(network_scores, network_path)
  return W, num_genes, gene_indexes, gene_scores

def load_random_networks(g, interactions):
  random_networks_path = f"artifacts/random_networks_score.pkl.gz"
  E = g.number_of_edges()
  Q = 10
  inter_genes = list(interactions["entrezgene"].unique())
  random_networks = {}
  if os.path.exists(random_networks_path):
    print('loading random networks from disk')
    random_networks = joblib.load(random_networks_path)
  else:
    for i in range(100):
        H = g.copy()
        nx.swap.double_edge_swap(H, nswap=Q*E, max_tries=Q*E*2)
        W_temp, num_genes_temp, gene_indexes_temp, gene_scores_temp = generate_propagate_data(H, inter_genes)
        random_networks[i] = gene_scores_temp
        print(f"network {i} generated")
  random_networks_path = f"artifacts/random_networks_score.pkl.gz"
  joblib.dump(random_networks, random_networks_path)
  return random_networks
  
def load_and_map_interactions():
  # https://www.nature.com/articles/s41586-020-2286-9#Sec36
  interactions = pd.read_csv("covid_files/data/inputs/interactions.csv")
  # mapping to entrez id
  xli = interactions["PreyGene"].unique().tolist()
  out = pd.DataFrame(mg.querymany(xli, scopes="symbol", fields="entrezgene", species="human"))
  interactions = pd.merge(interactions, out[["query", "entrezgene"]], left_on="PreyGene", right_on="query")
  interactions["entrezgene"] = interactions["entrezgene"].astype(np.float).astype("Int32")
  return interactions

def load_all_targets():
  all_targets = {'enterocytes': pd.read_csv("covid_files/data/inputs/Enterocytes.csv", dtype={'entrez':pd.Int64Dtype()}).dropna().set_index('gene').to_dict()['entrez'],
               'proximal': pd.read_csv("covid_files/data/inputs/Proximal_tubule_cells.csv", dtype={'entrez':pd.Int64Dtype()}).dropna().set_index('gene').to_dict()['entrez'],
               'cardiomyocytes':pd.read_csv("covid_files/data/inputs/cardiomyocytes.csv", dtype={'entrez':pd.Int64Dtype()}).dropna().set_index('gene').to_dict()['entrez'],
               'bronchial':pd.read_csv("covid_files/data/inputs/human_bronchial_epithelial.csv", dtype={'entrez':pd.Int64Dtype()}).dropna().set_index('gene').to_dict()['entrez'],
               'lung':pd.read_csv("covid_files/data/inputs/lung.csv", dtype={'entrez':pd.Int64Dtype()}).set_index('gene').dropna().to_dict()['entrez'],
               'lymphocytes':pd.read_csv("covid_files/data/inputs/lymphocytes.csv", dtype={'entrez':pd.Int64Dtype()}).dropna().set_index('gene').to_dict()['entrez'],
               'neuronal':pd.read_csv("covid_files/data/inputs/neuronal.csv", dtype={'entrez':pd.Int64Dtype()}).dropna().set_index('gene').to_dict()['entrez'],
               'vascular':pd.read_csv("covid_files/data/inputs/vascular.csv", dtype={'entrez':pd.Int64Dtype()}).dropna().set_index('gene').to_dict()['entrez']}
  return all_targets

