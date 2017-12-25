import networkx as nx
import matplotlib.pyplot as plt
import random
import community
import numpy as np
def delta_m(G, node, component, total_weightx2):
	sum_in = 0
	sum_total = 0
	component_temp = component.copy()
	# print(component_temp)
	for edge in G.edges(nbunch=component_temp, data=True):
		sum_total += edge[2]['weight']	
		if edge[0] in component_temp and edge[1] in component_temp:
			sum_in += edge[2]['weight']
	nbunch = [node]
	sum_incedent = 0
	sum_incedent_inside = 0
	component_temp.add(node)
	for edge in G.edges(nbunch=nbunch, data=True):
		#print("edge: {} component: {}".format(edge, component_temp))
		sum_incedent += edge[2]['weight']
		if edge[0] in component_temp and edge[1] in component_temp:
			sum_incedent_inside += edge[2]['weight'] 
	# print("node: {} component {} sum_in: {} sum_total: {} sum_incedent: {} sum_incedent_inside: {} twice_sum_of_weights: {}".format(node, component, sum_in, sum_total, sum_incedent,sum_incedent_inside,total_weightx2))
	part1 = (sum_in + (2 * sum_incedent_inside)) / total_weightx2
	part2 = (sum_total + sum_incedent) / total_weightx2
	part2 *= part2 
	part3 = sum_in / total_weightx2
	part4 = sum_total / total_weightx2
	part4 *= part4
	part5 = sum_incedent / total_weightx2
	part5 *= part5
	return (part1 - part2) - (part3 - part4 - part5)
def modularity(G, sList):
		modularity = 0
		A = nx.to_numpy_matrix(G)
		edges = nx.number_of_edges(G)
		degrees = [x[1] for x in list(G.degree())]
		for x in range(len(sList)):
			for y in range(x):
				if sList[x] == sList[y]:
					delta = 1
				else:
					delta = 0
				modularity += (A[x,y] - (degrees[x] * degrees[y]) / (2 * edges)) * delta
		return modularity * (1 / (2 * edges))

def twice_sum_of_weights(G):
	return 2 *sum([edge[2]['weight'] for edge in G.edges(data=True)])
def comparison(G):
	comparison = (community.best_partition(G))
	buckets = {}
	for key in comparison:
		if comparison[key] in buckets:
			buckets[comparison[key]].append(key)
		else:
			buckets[comparison[key]] = [key]
	print (buckets)
def find_best_place_to_move_with_labels(G, node, communities, community_numbers, total_weightx2):
	labels = {}
	max_delta_m = 0
	best_community = -1
	for neighbor in G.neighbors(node):
		if node not in communities[community_numbers[neighbor]]:
			communities[community_numbers[node]].remove(node)
			if (communities[community_numbers[node]]):
				loss = delta_m(G, node, communities[community_numbers[node]], total_weightx2)
			else:
				loss = 0
			mod_gain = delta_m(G, node, communities[community_numbers[neighbor]], total_weightx2) - loss
			# print('mod_gain {} loss {}'.format(mod_gain, loss))
			communities[community_numbers[node]].add(node)
			labels[(node, neighbor)] = round(mod_gain, 3)
			if mod_gain > max_delta_m:
				max_delta_m = mod_gain
				best_community = community_numbers[neighbor]
	return None if best_community <= 0 else best_community, labels
def calculate_comunitites_part_I(G, pos):
	def find_best_place_to_move(G, node, communities, community_numbers, total_weightx2):
		max_delta_m = 0
		best_community = -1
		for neighbor in G.neighbors(node):
			if node not in communities[community_numbers[neighbor]]:
				communities[community_numbers[node]].remove(node)
				if (communities[community_numbers[node]]):
					loss = delta_m(G, node, communities[community_numbers[node]], total_weightx2)
				else:
					loss = 0
				mod_gain = delta_m(G, node, communities[community_numbers[neighbor]], total_weightx2) - loss
				# print('mod_gain {} loss {}'.format(mod_gain, loss))
				communities[community_numbers[node]].add(node)
				if mod_gain > max_delta_m:
					max_delta_m = mod_gain
					best_community = community_numbers[neighbor]
		#print(max_delta_m)
		# print("max_delta_m: {} node: {} best_community: {}".format(max_delta_m, node, best_community))
		return None if best_community <= 0 else best_community
	total_weightx2 = twice_sum_of_weights(G)
	nodes = list(G)
	communities = {int(node):set([int(node)]) for node in nodes}
	community_numbers = {int(node) : int(node) for node in nodes}
	start_over = len(nodes)
	index = 0
	while start_over >= 0:
		node = nodes[index]
		community_number, labels = find_best_place_to_move_with_labels(G, node, communities,community_numbers, total_weightx2)
		#print(community_number)
		print("communities: {}".format(communities))
		if labels:
			print_graph(G, communities, labels, pos, node)
			plt.show()
			plt.clf()
		if community_number != None:
			start_over = len(nodes)
			community_numbers[node], communities = move(node, community_numbers[node],community_number, communities)
			# if communities[community_number] & set([node]):
			# 	loss = delta_m(G, node, communities[community_number] & set([node]), total_weightx2)
			# else:
			# 	loss = 0
			# mod_gain = delta_m(G, node, communities[community_numbers[node]], total_weightx2) - loss
			# print("communities: {} mod_gain for moving back: {} loss {} unadjusted {}".format(communities, mod_gain, loss, delta_m(G, node, communities[community_numbers[community_number]], total_weightx2)))
			
			# print("communities: {} modularity: {}".format(communities, modularity(G, [community_numbers[x] for x in nodes])))
		else:
			start_over -= 1
		index = (index + 1) % len(nodes)
		# index = random.randrange(0, len(nodes))
	return communities, community_numbers
def move(node, source, dest, communities):
	communities[dest].add(node)
	communities[source].remove(node)
	if not communities[source]:
		del communities[source]
	return dest, communities
def add_edge_to_graph(G, edge_tup):
	G.add_edge(edge_tup[0], edge_tup[1], weight=1)
	G.add_edge(edge_tup[1], edge_tup[0], weight=1)
def merge_communities(communities, community_numbers, oldG):
	newG = nx.Graph()
	newG.add_nodes_from(communities)
	edges = [edge for edge in oldG.edges(data=True)]
	for edge in edges:
		com_num = community_numbers[edge[0]]
		if edge[1] in communities[com_num]:
			if com_num in newG.neighbors(com_num):
				newG[com_num][com_num]['weight'] += 2
			else:
				newG.add_edge(com_num, com_num, weight=2)
		else:
			com_num1 = community_numbers[edge[0]]
			com_num2 = community_numbers[edge[1]]
			if com_num2 in newG.neighbors(com_num1):
				newG[com_num1][com_num2]['weight'] += 1
			else:
				newG.add_edge(com_num1, com_num2, weight=1)
	return newG

def simple_mod_max_equal_size_start(G, rounds):
	nodes_list = list(G.nodes())
	size = len(nodes_list)
	random.shuffle(nodes_list)
	nodes1 = list(nodes_list[:int(size / 2)])
	nodes2 = list(nodes_list[int(size / 2):])
	sList = [1 if x in nodes1 else -1 for x in range(size)]
	# print(sList)
	for time in range(rounds):
		current_modularity = modularity(G, sList)
		#print("current_modularity = {}".format(current_modularity))
		modlist = {current_modularity : sList}
		done_nodes = [False for x in range(size)]
		while not all(x for x in done_nodes):
			best_mod_change = -10
			best_mod_change_index = 0
			temp_modularity = modularity(G, sList)
			for x in range(size):
				if not done_nodes[x]:
					sList[x] *= -1
					mod_change =  modularity(G, sList) - temp_modularity
					if mod_change > best_mod_change:
					 	best_mod_change = mod_change
					 	best_mod_change_index = x
					sList[x] *= -1
			sList[best_mod_change_index] *= -1
			done_nodes[best_mod_change_index] = True
			modlist[modularity(G, sList)] = sList.copy()
		best_state = max(modlist)
		sList = modlist[best_state]
	return sList
def print_graph(G, communities, labels, pos, main=-1):
	node_colors = list(G)
	i = 0
	for key in communities:
		for node in communities[key]:
			node_colors[node] = key
		i += 1
	nx.draw(G, with_labels=True, pos=pos, cmap=plt.get_cmap('jet'), node_color=node_colors)
	nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
	if node != -1:
		plt.text(0,0, main)
G = nx.Graph()
G.add_edge(0, 2, weight=1)
G.add_edge(0, 4, weight=1)
G.add_edge(0, 5, weight=1)
G.add_edge(0, 3, weight=1)
G.add_edge(1, 2, weight=1)
G.add_edge(1, 4, weight=1)
G.add_edge(1, 7, weight=1)
G.add_edge(2, 4, weight=1)
G.add_edge(2, 5, weight=1)
G.add_edge(2, 6, weight=1)
G.add_edge(3, 7, weight=1)
G.add_edge(4, 10, weight=1)
G.add_edge(5, 11, weight=1)
G.add_edge(5, 7, weight=1)
G.add_edge(6, 7, weight=1)
G.add_edge(6, 11, weight=1)
G.add_edge(8, 9, weight=1)
G.add_edge(8, 10, weight=1)
G.add_edge(8, 11, weight=1)
G.add_edge(8, 14, weight=1)
G.add_edge(8, 15, weight=1)
G.add_edge(9, 12, weight=1)
G.add_edge(9, 14, weight=1)
G.add_edge(10, 12, weight=1)
G.add_edge(10, 13, weight=1)
G.add_edge(10, 14, weight=1)
G.add_edge(11, 13, weight=1)
total_weightx2 = twice_sum_of_weights(G)
nodes = list(G)
# communities = {int(node):set([int(node)]) for node in nodes}
# community_numbers = {int(node) : int(node) for node in nodes}
# for component in G.neighbors(0):
# 	print(delta_m(G, 0, communities[component], total_weightx2))
# calculate_comunitites_part_I(G)
# print(list(G))
# move()
# comparison(G)
communities, community_numbers = calculate_comunitites_part_I(G, pos = nx.circular_layout(G)
)
# sList = simple_mod_max_equal_size_start(G, 1)
# print([a for a in range(len(sList)) if sList[a] == 1], [a for a in range(len(sList)) if sList[a] == -1])
# print(modularity(G, [community_numbers[x] for x in list(G)]))
# print(modularity(G, sList))
# while len(list(G)) > 2:
# 	communities, community_numbers = calculate_comunitites_part_I(G)
# 	print(communities)
# 	print(G.edges(data=True))
# 	G = merge_communities(communities, community_numbers, G)
# print(nx.get_edge_attributes(G , 'weight'))
color_list = plt.cm.Set3(np.linspace(0, 1, len(list(G)) + 1))
print_graph(G, communities, nx.get_edge_attributes(G , 'weight'), pos= nx.circular_layout(G))
# print(communities)
# print(newG.edges())
# nx.draw(newG, pos=pos, with_labels=True)
plt.axis('on')
plt.show()