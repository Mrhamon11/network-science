import pickle
from sys import argv
input_file = open(argv[1], 'rb')
temp_list = pickle.load(input_file)
com_pair = temp_list[10]
input_file = open(argv[2], 'r')
output_file = open(argv[3], "w")
input_file2 = open(argv[4], 'r')
zero_based = {}
nums = 0
for node in com_pair[1]:
	if com_pair[1][node] not in zero_based:
		zero_based[com_pair[1][node]]= nums
		nums += 1
actual_dict = {}
for line in input_file2:
	items = line.split()
	actual_dict[int(items[0])] = items[1]
new_list = []

for line in input_file:
     if ":" in line and "#" not in line:
             items = line.split()
             Louvain_community = zero_based[com_pair[1][int(items[3])]] 
             #" " + str(len(com_pair[0][Louvain_community]))
             final_lines =  actual_dict[int(items[3])] + " " + str(Louvain_community) + " " + items[0].split(":")[0] + " " + items[3] + "\n"
             # line2 = line + " " + str(Louvain_community)  + " actual:" + actual_dict[int(items[3])]
             # line2 = line2.replace("\n","")
             # line2 += "\n"
             # print(line2)
             new_list.append(final_lines)
def find_group(x):
	split = x.split()
	return int(split[0])
def find_Louvain(x):
	items = x.split()
	return int(items[1])
def find_info(x):
	items = x.split()
	return int(items[2])
new_list.sort(key=lambda x : (find_group(x), find_Louvain(x), find_info(x)))
def add_to_dept_dict(diction, item, section):
	if section not in diction:
		diction[section] = {}
		diction[section][item] = 1
	elif item in diction[section]:
		diction[section][item] += 1
	else:
		diction[section][item] = 1
	return diction
dept_dict = {}
output_file.write('actual_group Louvain_group infoMap_group node_name\n')
for item in new_list:
	dept_dict = add_to_dept_dict(dept_dict, find_Louvain(item), find_group(item))
	output_file.write(item)
max_lens = [] 
for key in dept_dict:
	max_lens.append((dept_dict[key][max(dept_dict[key], key=lambda x  : dept_dict[key][x])], sum(dept_dict[key][x] for x in dept_dict[key])))
print (max_lens)
print(dept_dict)
print
