import pickle
input_file = open("./temp.txt", 'rb')
temp_list = pickle.load(input_file)
com_pair = temp_list[10]
input_file = open("./firstcom.txt", 'r')
output_file = open("./both_communitites.txt", "w")
for line in input_file:
     if "#" not in line:
             items = line.split()
             Louvain_community = com_pair[1][int(items[0])] 
             line2 = line + " " + str(Louvain_community) + " " + str(len(com_pair[0][Louvain_community]))
             line2 = line2.replace("\n","")
             line2 += "\n"
             print(line2)
             output_file.write(line2)