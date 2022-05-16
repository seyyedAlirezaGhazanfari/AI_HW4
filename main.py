import itertools
import re
from collections import defaultdict


class Factor:
    def __init__(self, parents, cpt, current_var):
        self.elements = []
        self.elements.extend(parents)
        self.elements.append(key)
        self.probs = cpt
        self.current_var = current_var
        self.permutation_memo = {}
        self.table = defaultdict(dict)
        self.build_factor()

    def gen_permutations(self, length):
        assert (length >= 0)
        if length in self.permutation_memo:
            return self.permutation_memo[length]
        else:
            l = [True, False]
            perms = list(itertools.product(l, repeat=length))
            assert (len(perms) == pow(2, length))
            self.permutation_memo[length] = perms
            return perms

    def build_factor(self, evidences):
        perms = self.gen_permutations(len(self.elements))
        entries = dict()
        temp = dict()
        for perm in perms:
            conflict = False
            for pair in zip(self.elements, perm):
                if pair[0] in evidences.keys() and evidences[pair[0]] != pair[1]:
                    conflict = True
                    break
                temp[pair[0]] = pair[1]
            if conflict:
                continue
            else:
                key = tuple(temp[elem] for elem in temp.keys())
                entries[key] = self.get_prob(temp)

    def get_prob(self, temp: dict):
        if len(self.elements) == 1:
            prob = self.net[Y]['prob'] if e[Y] else 1 - self.net[Y]['prob']

            # Y has at least 1 parent
        else:
            # get the value of parents of Y
            parents = tuple(e[p] for p in self.net[Y]['parents'])

            # query for prob of Y = y
            prob = self.net[Y]['condprob'][parents] if e[Y] else 1 - self.net[Y]['condprob'][parents]
        return prob



    def marginalize(self, marginalized_elem):
        for row in range(len(self.table.keys())):
            for column in range(len(self.table[row].keys())):
                pass

    def find_row_idx(self, ):
        pass

    @staticmethod
    def join(factor1, factor2):
        pass


def find_all_paths(x: int, y: int, undirected_graph: dict, visited: set):
    if x == y:
        return [[y]]
    if x in visited:
        return list()
    options = undirected_graph[x]
    temp_options = list()
    for option in options:
        if option not in visited:
            temp_options.append(option)
    options = temp_options
    if not options:
        return list()
    paths = list()
    visited.add(x)
    for option in options:
        paths_res = find_all_paths(option, y, undirected_graph, visited)
        if paths_res:
            for path in paths_res:
                path.append(x)
                paths.append(path)
    return paths


def is_path_blocked(path, evidences, graph):
    if len(path) < 3:
        return False
    for i in range(0, len(path)):
        if i > len(path) - 3:
            return False
        node1 = path[i]
        node2 = path[i + 1]
        node3 = path[i + 2]
        # A -> C -> B
        if node2 in graph[node1] and node3 in graph[node2]:
            if node2 in evidences:
                return True
        elif node1 in graph[node2] and node2 in graph[node3]:
            # A <- C <- B
            if node2 in evidences:
                return True
        elif node1 in graph[node2] and node3 in graph[node2]:
            # A <- C -> B
            if node2 in evidences:
                return True
        elif node2 in graph[node1] and node2 in graph[node3]:
            # A -> C <- B
            if node2 not in evidences:
                return True
    return False


def create_undirected_graph(graph: dict):
    undirected_graph = dict()
    for node in graph.keys():
        temp_list = graph[node]
        for item in graph.keys():
            if node in graph[item] and item not in graph[node]:
                temp_list.append(item)
        undirected_graph[node] = temp_list
    return undirected_graph


def reverse_paths(paths):
    res = []
    for path in paths:
        path.reverse()
        res.append(path)
    return res


def create_evidences_pos(evidences):
    temp = list()
    for pos, val in evidences:
        temp.append(pos)
    return temp


def separation(graph, x, y, evidences):
    visited = set()
    undirected_graph = create_undirected_graph(graph)
    paths = find_all_paths(x, y, undirected_graph=undirected_graph, visited=visited)
    paths = reverse_paths(paths)
    evidences_pos = create_evidences_pos(evidences)
    for path in paths:
        if not is_path_blocked(path, evidences_pos, graph):
            print("dependent")
            return
    print("independent")


def join_factors(hidden_var, cpts, graph):
    for child in graph[hidden_var]:
        # cpts[hidden_var] = multiply(cpts[child], cpts[hidden_var], hidden_var)
        pass


def eliminate_variable(hidden_var, cpts):
    pass


def variable_elimination(hiddens: list, graph, cpts):
    if not hiddens:
        return cpts
    to_eliminate = hiddens.pop(0)
    join_factors(to_eliminate, cpts, graph)
    eliminate_variable(to_eliminate, cpts)
    return variable_elimination(hiddens, graph, cpts)


def get_input():
    n = int(input())
    graph = dict()
    graph_cpts = dict()
    for node_id in range(1, n + 1):
        input_txt = input()
        neighbors = list(map(int, input_txt.split(" "))) if input_txt else list()
        input_txt = input()
        cpt = list(map(float, input_txt.split(" "))) if input_txt else list()
        graph[node_id] = neighbors
        graph_cpts[node_id] = cpt
    input_txt = input()
    evidences = re.findall("(\d)->(\d)", input_txt) if input_txt else list()
    x, y = list(map(int, input().split(" ")))
    separation(graph, x, y, evidences)


def create_cpt_tables(cpt: list, node_id: int):
    pass


if __name__ == '__main__':
    get_input()
