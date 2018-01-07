import pickle
from sys import argv
input_file = open(argv[1], 'rb')
temp_list = pickle.load(input_file)
com_pair = temp_list[10]
	
input_file = open(argv[2], 'r')
output_file = open(argv[3], "w")
input_file2 = open(argv[4], 'r')
actual_dict = {}
for line in input_file2:
	items = line.split()
	actual_dict[int(items[0])] = items[1]
new_list = []

for line in input_file:
     if ":" in line and "#" not in line:
             items = line.split()
             Louvain_community = com_pair[1][int(items[3])] 
             #" " + str(len(com_pair[0][Louvain_community]))
             line2 = line + " " + str(Louvain_community)  + " actual:" + actual_dict[com_pair[1][int(items[3])]]
             line2 = line2.replace("\n","")
             line2 += "\n"
             # print(line2)
             new_list.append(line2)
def find_group(x):
	split = x.split(":")
	print(split)
	return int(split[2])
new_list.sort(key=find_group)
output_file.close
for item in new_list:
	output_file.write(item)
