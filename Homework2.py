from scipy.sparse import csgraph
import numpy as np
from numpy import linalg
import networkx as nx
import pylab
import matplotlib.pyplot as plt
import math
from random import shuffle
A = np.matrix([[0, 1, 1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 0, 1, 0], [0, 0, 0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 1, 0, 1, 0]])
G = nx.from_numpy_matrix(A)
names = {0:'Zypman', 1:'Cwilich', 2:'Prodan', 3:'Buldyrev', 4:'Bastuscheck', 5:'Asherie', 6:'Edelman', 7:'Santos'}
# G = nx.Graph()
# G.add_edges_from([(1,2), (1, 6), (1, 4), (2, 4), (2, 3), (3, 5), (3, 4), (5, 4), (6, 5)])
# names = {1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6'}
def get_global_clustering(matrix):
	def get_triplets(row_index, column_index):
		x = column_index
		connected = 0
		triplets = 0
		while x < len(matrix) - 1:
			x += 1;
			if matrix[row_index][x] == 1:
				triplets += 1
				if matrix[column_index][x] == 1:
					connected += 1
		return [triplets, connected]
	size = len(matrix)
	triplets = 0
	connectec_triplets = 0
	for row_index in range(size):
		for column_index in range(size):
			if matrix[row_index][column_index] == 1:
				trips = get_triplets(row_index, column_index)
				triplets += trips[0]
				connectec_triplets += trips[1]
	return connectec_triplets / triplets
def plot_info(G, names):
	def get_spread(dictionary):
		min_val = dictionary[1]
		max_val = dictionary[1]
		for key in dictionary:
			if min_val > dictionary[key]:
				min_val = dictionary[key]
			if max_val < dictionary[key]:
				max_val = dictionary[key]
		if min_val == 0:
			dictionary['Spread'] = 'infinity'
		else:
			dictionary['Spread'] = max_val/min_val
		return dictionary
	def get_katz_alpha(matrix):
		largest = max(linalg.eigvals(matrix))
		return 1/largest - 0.01

	L = nx.laplacian_matrix(G)
	print([round(a, 5) for a in linalg.eigvals(L.A)])
	nx.draw_networkx(G, show_labels=True, labels=names)
	degree_centralities = get_spread(nx.degree_centrality(G))
	eigenvector_centralities = get_spread(nx.eigenvector_centrality(G))
	katz_centralities = get_spread(nx.katz_centrality(G, alpha=get_katz_alpha(nx.to_numpy_matrix(G))))
	page_rank_centralities = get_spread(nx.pagerank(G, alpha=0.85))
	closeness_centralities = get_spread(nx.closeness_centrality(G))
	betweeness_centralities = get_spread(nx.betweenness_centrality(G))
	data = []
	for key in degree_centralities:
		data.append([degree_centralities[key], eigenvector_centralities[key], katz_centralities[key], page_rank_centralities[key], closeness_centralities[key], betweeness_centralities[key]])
	row_lables =[]
	for x in range(len(names)):
		row_lables.append(names[x])
	row_lables.append('Spread')
	centralities = ['Degree', 'Eigenvector', 'Katz', 'Page Rank', 'Closeness', 'Betweenness']
	the_table = plt.table(cellText=data, rowLabels=row_lables, colLabels=centralities, loc='bottom')
	plt.tight_layout()
	plt.subplots_adjust(left=0.29, bottom=0.46, right=0.75, top=None,
	                wspace=None, hspace=None)
	the_table.set_fontsize(24)
	the_table.scale(2,2)
	plt.axis('off')
	plt.show()
def draw_balanence_graph(G, edges, names):
	nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G), edge_labels=edges)
	nx.draw_networkx(G, show_labels=True, labels=names, pos=nx.spring_layout(G))
	plt.tight_layout()
	plt.axis('off')
	plt.show()

def dot_rowproduct(row1, row2):
	result = 0 
	for i in range(len(row1)):
		result += row1[i] * row2[i]
	return result
def cosine_similarity(matrix, i, j):
	return dot_rowproduct(matrix[i], matrix[j]) / math.sqrt(sum(matrix[i]) * sum(matrix[j]))



def degree_and_closeness(G, names):
	print(nx.average_clustering(G))
	degree_list = list(G.degree())
	degree_list.sort(key=lambda x : x[1])
	clustering_coefficients = nx.clustering(G)
	row_labels = []
	data = []
	for pair in degree_list:
		row_labels.append(names[pair[0]])
		data.append([pair[1], clustering_coefficients[pair[0]]])
	the_table = plt.table(cellText=data, rowLabels=row_labels, colLabels=['Degree', 'Local Clustering'], loc='bottom')

	the_table.set_fontsize(24)
	the_table.scale(2,2)
	plt.tight_layout()
	plt.subplots_adjust(left=0.29, bottom=0.5, right=0.48, top=None,
	                wspace=None, hspace=None)
	plt.axis('off')
	plt.show()
def kernigham_lin(G):
	def cut_set(nodes1, nodes2, graph):
		cut_set_size = 0
		for edge in graph.edges():
			if edge[0] in nodes1:
				if edge[1] in nodes2:
					cut_set_size += 1
			else:
				if edge[1] in nodes1:
					cut_set_size += 1
		return cut_set_size
	nodes_list = list(G.nodes())
	shuffle(nodes_list)
	size = int(len(nodes_list) / 2)
	nodes1 = list(nodes_list[:size])
	nodes2 = list(nodes_list[size:])
	print(nodes1)
	print(nodes2)
	min_cut = cut_set(nodes1, nodes2, G)
	min_nodes1 = nodes1
	min_nodes2 = nodes2
	for x in range(size):
		for y in range(size):
			min_nodes1 = nodes1[:]
			min_nodes2 = nodes2[:]
			min_nodes1[x] = nodes2[y]
			min_nodes2[y] = nodes1[x]
			if min_cut > cut_set(min_nodes1, min_nodes2, G):
				nodes1 = min_nodes1
				nodes2 = min_nodes2
				min_cut = cut_set(nodes1, nodes2, G)
	return min_cut, nodes1, nodes2
			

def pearson_correlation_coefficeint(matrix, i, j):
	size = len(matrix)
	average_i = sum(matrix[i]) / size
	average_j = sum(matrix[j]) / size
	numerator = 0
	denom1 = 0
	denom2 = 0
	for k in range(size):
		numerator += (matrix[i][k] - average_i) * (matrix[j][k] - average_j)
		denom1 += math.pow((matrix[i][k] - average_i), 2)
		denom2 += math.pow((matrix[j][k] - average_j), 2)
	return numerator / (math.sqrt(denom1) * math.sqrt(denom2))
	# L = csgraph.laplacian(G, normed=False)
	# print(L)
	# print([round(a, 5) for a in linalg.eigvals(L)])
def get_similarity_over_all_nodes(matrix, similarity_name, similarity_function, names):
	x = 0
	for key in range(len(names)):
		for key2 in range(key, len(names)):
			if key2 != key:
				x += 1
				print(str(x) + " " +  names[key] + " " + names[key2] + " " + similarity_name + ": " + str(similarity_function(A.tolist(), key, key2)))
# plot_info(G, names)
# get_similarity_over_all_nodes(A.tolist(), "cosine_similarity", cosine_similarity, names)
# get_similarity_over_all_nodes(A.tolist(), "pearson similarity", pearson_correlation_coefficeint, names)
# # for key in range(len(names)):
# # 	for key2 in range(key, len(names)):
# # 		if key2 != key:
# # 			x += 1
# # 			print(str(x) + " " +  names[key] + " " + names[key2] + " cosine similarity: " + str(cosine_similarity(A.tolist(), key, key2)))
# #draw_balanence_graph(G, {(0, 1):'+', (0,2):'-'}, names)
# plot_info(G, names)
# degree_and_closeness(G, names)
# print(get_global_clustering(A.tolist()))
cut_size, set1, set2 = kernigham_lin(G)
print("cut_size {} {} {} ".format(cut_size, [names[a] for a in set1], [names[b] for b in set2]))