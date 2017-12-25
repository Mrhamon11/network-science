import networkx as nx
import matplotlib.pyplot as plt
import community
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
	return(community.best_partition(G))
	buckets = {}
	comparison = comparison(G)
	for key in comparison:
		if comparison[key] in buckets:
			buckets[comparison[key]].append(key)
		else:
			buckets[comparison[key]] = [key]
	print (buckets)
def calculate_comunitites_part_I(G):
	def find_best_place_to_move(G, node, communities, community_numbers, total_weightx2):
		max_delta_m = 0
		best_community = -1
		for neighbor in G.neighbors(node):
			if node not in communities[community_numbers[neighbor]]:
				if communities[community_numbers[node]] & set([node]):
					loss = delta_m(G, node, communities[community_numbers[node]] & set([node]), total_weightx2)
				else:
					loss = 0
				mod_gain = delta_m(G, node, communities[community_numbers[neighbor]], total_weightx2) - loss
				#print('mod_gain {}'.format(mod_gain))
				if mod_gain > max_delta_m:
					max_delta_m = mod_gain
					best_community = community_numbers[neighbor]
		#print(max_delta_m)
		print("max_delta_m: {}".format(max_delta_m))
		return None if best_community <= 0 else best_community
	total_weightx2 = twice_sum_of_weights(G)
	nodes = list(G)
	communities = {int(node):set([int(node)]) for node in nodes}
	community_numbers = {int(node) : int(node) for node in nodes}
	start_over = len(nodes)
	index = 0
	while start_over >= 0:
		node = nodes[index]
		community_number = find_best_place_to_move(G, node, communities,community_numbers, total_weightx2)
		#print(community_number)
		if community_number != None:
			communities[community_number].add(node)
			communities[community_numbers[node]].remove(node)
			if not communities[community_numbers[nodes[index]]]:
				del communities[community_numbers[nodes[index]]]
			community_numbers[node] = community_number
			start_over = len(nodes)
			# if communities[community_number] & set([node]):
			# 	loss = delta_m(G, node, communities[community_number] & set([node]), total_weightx2)
			# else:
			# 	loss = 0
			# mod_gain = delta_m(G, node, communities[community_numbers[node]], total_weightx2) - loss
			# print("communities: {} mod_gain for moving back: {} loss {} unadjusted {}".format(communities, mod_gain, loss, delta_m(G, node, communities[community_numbers[community_number]], total_weightx2)))
			
			print("community_numbers: {} communities: {} best move : {} modularity: {}".format(community_numbers, communities, find_best_place_to_move(G, node, communities, community_numbers, total_weightx2), modularity(G, [community_numbers[x] for x in nodes])))
		else:
			start_over -= 1
		index = (index + 1) % len(nodes)
	return communities, community_numbers

def add_edge_to_graph(G, edge_tup):
	G.add_edge(edge_tup[0], edge_tup[1], weight=1)
	G.add_edge(edge_tup[1], edge_tup[0], weight=1)

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
communities = {int(node):set([int(node)]) for node in nodes}
community_numbers = {int(node) : int(node) for node in nodes}
# for component in G.neighbors(0):
# 	print(delta_m(G, 0, communities[component], total_weightx2))
# calculate_comunitites_part_I(G)
# print(list(G))
communities, community_numbers = calculate_comunitites_part_I(G)
print(communities)