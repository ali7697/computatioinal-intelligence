import random
import math
import numpy as np
import operator
import statistics
import matplotlib.pyplot as plt
import networkx as nx

num_initial_generation = 20 # 300
num_iterations = 140# 50
recom_prob = 1  # 0.8



# This section reads the input
infile = r"E:\educational\9th Semester\comp\proj1\steiner_in.txt"
lines = []
steiner_xs = []
steiner_ys = []
terminal_xs = []
terminal_ys = []
edges_1 = []
edges_2 = []
vertices_type = []  # demonstrates the types of vertices
vertices_x = []
vertices_y = []
edge_weights = []
edge_mst = []
with open(infile) as f:
    for line in f:
        lines.append(line)
num_steiner = lines[0].split(" ")[0]
num_steiner = int(num_steiner)
num_terminal = lines[0].split(" ")[1]
num_terminal = int(num_terminal)
num_edges = lines[0].split(" ")[2]
num_edges = int(num_edges)

for i in range(num_steiner):
    vertices_type.append("steiner")
    steiner_xs.append(lines[i + 1].split(" ")[0])
    steiner_ys.append(lines[i + 1].split(" ")[1].split('\n')[0])
    vertices_x.append(int(lines[i + 1].split(" ")[0]))
    vertices_y.append(int(lines[i + 1].split(" ")[1].split('\n')[0]))
for i in range(num_terminal):
    vertices_type.append("terminal")
    terminal_xs.append(lines[i + 1 + num_steiner].split(" ")[0])
    terminal_ys.append(lines[i + 1 + num_steiner].split(" ")[1].split('\n')[0])
    vertices_x.append(int(lines[i + 1 + num_steiner].split(" ")[0]))
    vertices_y.append(int(lines[i + 1 + num_steiner].split(" ")[1].split('\n')[0]))

edges_dict = {}

for i in range(num_edges):
    edges_1.append(int(lines[i + 1 + num_steiner + num_terminal].split(" ")[0]))
    edges_2.append(int(lines[i + 1 + num_steiner + num_terminal].split(" ")[1].split('\n')[0]))
    edge_weights.append(math.sqrt(
        (vertices_y[int(edges_1[i])] - vertices_y[int(edges_2[i])]) ** 2 +
        (vertices_x[int(edges_1[i])] - vertices_x[int(edges_2[i])]) ** 2)
    )
    edge_mst.append([i, edge_weights[i]])
    edges_dict[i] = edge_weights[i]


# sorting based on the weights
edges_dict_sorted = {k: v for k, v in sorted(edges_dict.items(), key=lambda item: item[1])}

# This function is used in calculating the MST
def FindParent(parent, node):
    if parent[node] == node:
        return node
    return FindParent(parent, parent[node])

# This function is used in calculating the MST
def Add_edge(ver1, ver2):
    if ver1 not in gr1:
        gr1[ver1] = []
    if ver2 not in gr1:
        gr1[ver2] = []
    gr1[ver1].append(ver2)
    gr1[ver2].append(ver1)

# This function is used in calculating the MST
def dfs1(x, vis1):
    vis1[x] = True
    if x not in gr1:
        gr1[x] = {}

    for u in gr1[x]:
        if not vis1[u]:
            dfs1(u, vis1)

# This function is used in calculating the MST
# It checks if the connected or not
# and returns the number of separated parts the
# graph is consisted from (it does up to 5)
def Is_Connected(vis1):
    # global vis1
    num_parts = 1
    key_list = list(vis1.keys())
    # Call for correct direction
    dfs1(key_list[0], vis1)
    for k in range(len(key_list)):
        if not vis1[key_list[k]]:
            num_parts = num_parts + 1
            break
    if num_parts == 1:
        return num_parts
    else:
        dfs1(key_list[k], vis1)

    for k in range(len(key_list)):
        if not vis1[key_list[k]]:
            num_parts = num_parts + 1
            break
    if num_parts == 2:
        return num_parts
    else:
        dfs1(key_list[k], vis1)
    for k in range(len(key_list)):
        if not vis1[key_list[k]]:
            num_parts = num_parts + 1
            break
    if num_parts == 3:
        return num_parts
    else:
        dfs1(key_list[k], vis1)
    for k in range(len(key_list)):
        if not vis1[key_list[k]]:
            num_parts = num_parts + 1
            break
    if num_parts == 4:
        return num_parts
    return 5


visited = {}

# calculates the MST
def mst(encode):
    global visited
    # get rid of the extra edges
    edges_dict_sorted_mst = edges_dict_sorted.copy()
    existing_edges = []
    for edge in edges_dict_sorted.keys():
        existing_edges.append(edge)
    for vert in range(num_steiner):
        if encode[vert] == 0:
            tmp_to_be_removed = []
            for edge in existing_edges:
                if edges_1[edge] == vert or edges_2[edge] == vert:
                    edges_dict_sorted_mst.pop(edge)
                    tmp_to_be_removed.append(edge)
            for i in tmp_to_be_removed:
                existing_edges.remove(i)

    # figuring out the vertices
    vertices = []
    parent = [None] * (num_steiner + num_terminal)
    rank = [None] * (num_steiner + num_terminal)
    for ii in range(num_steiner):
        if encode[ii] == 1:
            vertices.append(ii)
            parent[ii] = ii
            rank[ii] = 0
    for q in range(num_steiner, num_steiner + num_terminal):
        vertices.append(q)
        parent[q] = q
        rank[q] = 0
    selected_edges = []
    global gr1
    global gr2
    gr1 = {}
    gr2 = {}
    for edge in existing_edges:
        Add_edge(edges_1[edge], edges_2[edge])
    visited = {}  # This is the same size az vertices and tells if a vertex is visited
    for vertex in vertices:
        visited[vertex] = False
    connected = Is_Connected(visited)
    vistshode = []
    notvisited = []
    for y in visited.keys():
        if visited[y]:
            vistshode.append(y)
        else:
            notvisited.append(y)

    # if the graph is not connected it
    # returns numbers based on the number of different
    # parts consisting the graph
    if connected == 2:
        return 120000
    elif connected == 3:
        return 150000
    elif connected == 4:
        return 160000
    elif connected >= 5:
        return 1000000

    # This loop does the MST part
    for edge in edges_dict_sorted_mst.keys():
        vertex_1 = FindParent(parent, edges_1[edge])
        vertex_2 = FindParent(parent, edges_2[edge])
        # Parents of the source and destination nodes are not in the same subset
        # Add the edge to the spanning tree
        if vertex_1 != vertex_2:
            selected_edges.append(edge)
            if rank[vertex_1] < rank[vertex_2]:
                parent[vertex_1] = vertex_2
                rank[vertex_1] += 1
            else:
                parent[vertex_2] = vertex_1
                rank[vertex_2] += 1
    cost = 0
    for edge in selected_edges:
        cost += edges_dict_sorted_mst[edge]
    return cost

# initial encodings of the problem
encodings = []
encod = []
for i in range(num_steiner):
    encod.append(1)
encodings.append(encod)

# drawing the initial version were all edges exist
# drawing the graph
# encode = encod
# G = nx.Graph()
# edges_dict_sorted_mst = edges_dict_sorted.copy()
# existing_edges = []
# for edge in edges_dict_sorted.keys():
#     existing_edges.append(edge)
#     for vert in range(num_steiner):
#         if encode[vert] == 0:
#             tmp_to_be_removed = []
#             for edge in existing_edges:
#                 if edges_1[edge] == vert or edges_2[edge] == vert:
#                     edges_dict_sorted_mst.pop(edge)
#                     tmp_to_be_removed.append(edge)
#             for i in tmp_to_be_removed:
#                 existing_edges.remove(i)
#
# for edge in existing_edges:
#     point1 = [vertices_x[edges_1[edge]], vertices_y[edges_1[edge]]]
#     point2 = [vertices_x[edges_2[edge]], vertices_y[edges_2[edge]]]
#     xs = [point1[0], point2[0]]
#     ys = [point1[1], point2[1]]
#     plt.plot(xs, ys, color='blue', linewidth=0.5)
# for v in range(240, 320):
#     plt.scatter(vertices_x[v], vertices_y[v], c='red')
#
# plt.savefig("Initial.png")




# if you change these indexes to zero the graph
# will be disconnected
not_changable_indexes = []
con = []
for e in range(len(encod)):
    encod[e]=0
    tmp = mst(encod)
    con.append(tmp)
    if tmp>115000:
        not_changable_indexes.append(e)
    encod[e]=1
for k in range(num_initial_generation - 1):
    encodee = []
    tmp = random.sample(range(num_steiner), 110)  # not bad with 30 #also good wih 50
    encodee = encod.copy()
    for n in tmp:
        if n in not_changable_indexes:
            continue
        else:
            encodee[n] = 0
    encodings.append(encodee)



# iterations
for it in range(num_iterations):
    fitness = []
    children = []
    if it > 0:
        # by this part of the code we always maintain
        # the best coding we have ever reached and
        # by doing the recombination between two best
        # encodings we try to make them even better
        # also since the recombination probability
        # for this specific part of the code is 95 percent
        # and we do 110 iterations I am confident that even if
        # the recombinations do not improve the fitness of the
        # encodings, the encodings themselves will transfer to the
        # children array and also for this specific part I have not added
        # any mutation part.
        for itt in range(110):
            parents = random.sample(bests, 2)
            parent1 = parents[0][:]
            parent2 = parents[1][:]
            temp_child_1 = parent1[:]
            temp_child_2 = parent2[:]
            if random.random() < 0.95:
                single_point = random.randint(0, 235)  # ?????
                for i in range(single_point, single_point+5): # 20 was fine
                    if i in not_changable_indexes:
                        continue
                    else:
                        temp_child_1[i] = parent2[i]
                        temp_child_2[i] = parent1[i]
            children.append(temp_child_1)
            children.append(temp_child_2)
            fitness.append(100 / mst(temp_child_1))
            fitness.append(100 / mst(temp_child_2))

    # This is the main part of any iteration
    for itt in range(
            5 * num_initial_generation):
        # choosing the parents
        parent1 = random.sample(encodings, 1)[0]
        parent2 = random.sample(encodings, 1)[0]
        temp_child_1 = parent1[:]
        temp_child_2 = parent2[:]
        # recombination
        if random.random() < recom_prob:
            single_point = random.randint(0, 120)  # ?????
            for i in range(single_point, single_point+120): # 20 was fine
                if i in not_changable_indexes:
                    continue
                else:
                    temp_child_1[i] = parent2[i]
                    temp_child_2[i] = parent1[i]
        # mutation
        # This part is for mutation
        tmp1 = random.sample(range(num_steiner), 2)
        tmp2 = random.sample(range(num_steiner), 2)
        for n in tmp1:
            if n in not_changable_indexes:
                continue
            else:
                if temp_child_1[n] == 0:
                    temp_child_1[n] = 1
                else:
                    temp_child_1[n] = 0
        for n in tmp2:
            if n in not_changable_indexes:
                continue
            else:
                if temp_child_2[n] == 0:
                    temp_child_2[n] = 1
                else:
                    temp_child_2[n] = 0
        children.append(temp_child_1)
        children.append(temp_child_2)
        fitness.append(100 / mst(temp_child_1))
        fitness.append(100 / mst(temp_child_2))

    # choosing the survivors
    arr = np.array(fitness)
    encodings_key_indexes = arr.argsort()[-num_initial_generation:][::-1]
    encodings = []
    fitness1 = []
    fitt_probs = []
    for index in encodings_key_indexes:
        encodings.append(children[index])
        fitness1.append(fitness[index])
    bests = encodings[0:2][:]
    print(it)
    print("average: ", statistics.mean(fitness))
    print("Max: " , max(fitness))
    print("Min: " , min(fitness))



# print the value of mst of the shortest possible tree
fitness = {}
# calculate the fitnesses
for encodings_index in range(num_initial_generation):
    # list is an unhashable type
    # fitness[encodings[i]] = 1 / mst(encodings[i])
    fitness[encodings_index] = 100 / mst(encodings[encodings_index])
sum_fitness = 0
for i in fitness.values():
    sum_fitness += i
for key in fitness.keys():
    fitness[key] = fitness[key] / sum_fitness
final_res = []
for i in fitness.values():
    final_res.append(i)
index, value = max(enumerate(final_res), key=operator.itemgetter(1))
print("mst: ", mst(encodings[index]))
print("encoding: ", encodings[index])

# drawing the graph
# encode = encodings[index]
# G = nx.Graph()
# edges_dict_sorted_mst = edges_dict_sorted.copy()
# existing_edges = []
# for edge in edges_dict_sorted.keys():
#     existing_edges.append(edge)
#     for vert in range(num_steiner):
#         if encode[vert] == 0:
#             tmp_to_be_removed = []
#             for edge in existing_edges:
#                 if edges_1[edge] == vert or edges_2[edge] == vert:
#                     edges_dict_sorted_mst.pop(edge)
#                     tmp_to_be_removed.append(edge)
#             for i in tmp_to_be_removed:
#                 existing_edges.remove(i)
#
# for edge in existing_edges:
#     point1 = [vertices_x[edges_1[edge]], vertices_y[edges_1[edge]]]
#     point2 = [vertices_x[edges_2[edge]], vertices_y[edges_2[edge]]]
#     xs = [point1[0], point2[0]]
#     ys = [point1[1], point2[1]]
#     plt.plot(xs, ys, color='blue', linewidth=0.5)
# for v in range(240, 320):
#     plt.scatter(vertices_x[v], vertices_y[v], c = 'red')

# plt.savefig("Finall.png")