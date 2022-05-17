import re

import itertools
from collections import defaultdict


class Net:
    def __init__(self, graph, undirected_graph, cpt, evidences):
        self.permutations_memo = {}
        self.net = defaultdict(dict)
        self.graph = graph
        self.evidences = evidences
        self.undirected_graph = undirected_graph
        self.cpt = cpt
        self.create_net()

    def create_net(self):
        for node in self.graph.keys():
            self.net[node]['parents'] = self.get_parents(node, self.graph)
            self.net[node]['children'] = self.get_children(node, self.undirected_graph, self.graph)
            self.net[node]['prob'] = self.get_prob(node, self.graph, self.cpt)
            if self.net[node]['prob'] != -1:
                self.net[node]['condprob'] = self.get_cond_prob(node, self.graph, self.cpt)
            else:
                self.net[node]['condprob'] = []

    @staticmethod
    def get_parents(var, graph):
        return graph[var]

    @staticmethod
    def get_children(var, undirected_graph, graph):
        all_neighbors = set(undirected_graph[var])
        parents = set(graph[var])
        return list(
            all_neighbors.difference(parents)
        )

    @staticmethod
    def get_prob(var, graph, cpt):
        if not graph[var]:
            return cpt[var][0]
        return -1

    @staticmethod
    def get_cond_prob(var, graph, cpt):
        var_probs = cpt[var]
        length = len(graph[var])
        l = [True, False]
        perms = list(itertools.product(l, repeat=length))
        assert (len(perms) == pow(2, length))
        result = list()
        for perm_idx in range(len(perms)):
            result.append((perms[perm_idx], var_probs[perm_idx]))
        return result

    def query_given(self, y, e):
        if self.net[y]['prob'] != -1:
            prob = self.net[y]['prob'] if e[y] else 1 - self.net[y]['prob']
        else:
            parents = tuple(e[p] for p in self.net[y]['parents'])
            prob = self.net[y]['condprob'][parents] if e[y] else 1 - self.net[y]['condprob'][parents]
        return prob

    def gen_permutations(self, length):
        assert (length >= 0)
        if length in self.permutations_memo:
            return self.permutations_memo[length]
        else:
            l = [True, False]
            perms = list(itertools.product(l, repeat=length))
            assert (len(perms) == pow(2, length))
            self.permutations_memo[length] = perms
            return perms

    def build_factor(self, var, factor_vars, evidences: dict):
        params = [var, ]
        params.extend(factor_vars)
        perms = self.gen_permutations(len(params))
        entries = dict()
        e = dict()
        for perm in perms:
            conflict = False
            for pair in zip(params, perm):
                if pair[0] in evidences.keys() and evidences[pair[0]] != pair[1]:
                    conflict = True
                    break
                e[pair[0]] = pair[1]
            if conflict:
                continue
            else:
                key = tuple(e[elem] for elem in e.keys())
                entries[key] = self.query_given(var, e)
        return params, entries

    def join(self, common_var, factor1, factor2):
        vars1 = factor1[0]
        vars2 = factor2[0]
        new_vars = list()
        new_vars.extend(vars2)
        new_vars.extend(vars1)
        new_vars = set(new_vars)
        new_factor = dict()
        perms = self.gen_permutations(len(new_vars))
        entries = {}
        for perm in perms:
            for pair in zip(new_vars, perm):
                entries[pair[0]] = pair[1]
        key = tuple(entries[v] for v in new_vars)
        key1 = tuple(entries[v] for v in factor1[0])
        key2 = tuple(entries[v] for v in factor2[0])
        prob = factor1[1][key1] * factor2[1][key2]
        new_factor[key] = prob

        return new_vars, new_factor

    def marginalization(self, var, factor):
        vars = factor[0]
        for i, variable in enumerate(vars):
            if variable == var:
                new_variables = list(set(vars).difference([variable, ]))
                new_table = {}
                for entry in factor[1]:
                    entry = list(entry)
                    new_key = tuple(new_variables)
                    entry[i] = True
                    prob1 = factor[1][tuple(entry)]
                    entry[i] = False
                    prob2 = factor[1][tuple(entry)]
                    prob = prob1 + prob2
                    new_table[new_key] = prob
                    factor = (new_variables, new_table)
                    if len(new_variables) == 0:
                        del factor
        return factor

    def variable_elimination(self, X, evidences):
        eliminated = set()
        factors = list()
        while len(eliminated) < len(self.net):
            variables = filter(lambda v: v not in eliminated, list(self.net.keys()))
            variables = filter(lambda v: all(c in eliminated for c in self.net[v]['children']), variables)
            factor_vars = {}
            for v in variables:
                factor_vars[v] = [p for p in self.net[v]['parents'] if p not in evidences]
                if v not in evidences:
                    factor_vars[v].append(v)
            var = sorted(factor_vars.keys(), key=(lambda x: (len(factor_vars[x]), x)))[0]
            if len(factor_vars[var]) > 0:
                factors.append(self.build_factor(var, factor_vars, evidences))
            if var != X and var not in evidences:
                temp_factors = []
                for factor in factors:
                    factor_res = self.marginalization(var, factor)
                    temp_factors.append(factor_res)
                factors = temp_factors
            eliminated.add(var)
            for factor in factors:
                asg = {}
                perms = list(self.gen_permutations(len(factor[0])))
                perms.sort()
                for perm in perms:
                    for pair in zip(factor[0], perm):
                        asg[pair[0]] = pair[1]
                    key = tuple(asg[v] for v in factor[0])
        if len(factors) >= 2:
            result = factors[0]
            for factor in factors[1:]:
                result = self.join(var, result, factor)
        else:
            result = factors[0]
        return self.normalize((result[1][(False,)], result[1][(True,)]))

    @staticmethod
    def normalize(dist):
        return tuple(x * 1 / sum(dist) for x in dist)


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


def separation(graph, x, y, evidences, undirected_graph):
    visited = set()
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
    undirected_graph = create_undirected_graph(graph)
    separation(graph, x, y, evidences, undirected_graph)
    evidences = create_evidences(evidences)
    net = Net(graph, undirected_graph, graph_cpts, evidences)
    res = net.variable_elimination(x, evidences)
    print(res)
    res = net.variable_elimination(y, evidences)
    print(res)


def create_evidences(evidences):
    result = dict()
    for var, val in evidences:
        result[var] = val
    return result


if __name__ == '__main__':
    get_input()
