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
            self.net[node]['prob'] = -1
            self.net[node]['condprob'] = []
            if self.net[node]['parents']:
                self.net[node]['condprob'] = self.get_cond_prob(node, self.graph, self.cpt)
            else:
                self.net[node]['prob'] = self.get_prob(node, self.graph, self.cpt)

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
        result = dict()
        for perm_idx in range(len(perms)):
            result[perms[perm_idx]] = var_probs[perm_idx]
        return result

    @staticmethod
    def reachable(bn, x, e, end_node):
        L = set(e)
        ancestors = set()
        while L:
            y = L.pop()
            if y not in ancestors:
                L = L.union(bn[y]['parents'])
            ancestors = ancestors.union({y})
        visited, reachable = set(), set()
        up, down = True, False
        L = {(x, up)}
        while L:
            y, d = L.pop()
            if (y, d) not in visited:
                if y not in e:
                    reachable = reachable.union({y})
                visited = visited.union({(y, d)})
                if d == up and y not in e:
                    parents = set(bn[y]['parents'])
                    children = set(bn[y]['children'])
                    for elem in parents:
                        L = L.union({(elem, up)})
                    for elem in children:
                        L = L.union({(elem, down)})
                elif d == down:
                    if y not in e:
                        children = set(bn[y]['children'])
                        for elem in children:
                            L = L.union({(elem, down)})
                    if y in ancestors:
                        parents = set(bn[y]['parents'])
                        for elem in parents:
                            L = L.union({(elem, up)})
        if end_node in reachable:
            print("dependent")
        else:
            print("independent")

    @staticmethod
    def marginalization(var, factor):
        for j, v in enumerate(factor[0]):
            if v == var:
                new_vars = list(factor[0])[:j] + list(factor[0])[j + 1:]
                new_entries = {}
                for entry in factor[1]:
                    entry = list(entry)
                    new_key = tuple(entry[:j] + entry[j + 1:])
                    entry[j] = True
                    prob1 = factor[1][tuple(entry)]
                    entry[j] = False
                    prob2 = factor[1][tuple(entry)]
                    prob = prob1 + prob2
                    new_entries[new_key] = prob
                factor = (new_vars, new_entries)
                if len(new_vars) == 0:
                    del factor
        return factor

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
        params = []
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

    def join(self, factor1, factor2, evidences):
        vars1 = factor1[0]
        vars2 = factor2[0]
        new_vars = list()
        new_vars.extend(vars2)
        new_vars.extend(vars1)
        new_vars = set(new_vars)
        new_factor = dict()
        perms = self.gen_permutations(len(new_vars))
        for perm in perms:
            entries = {}
            conflict = False
            for pair in zip(new_vars, perm):
                if pair[0] in evidences.keys() and pair[1] != evidences[pair[0]]:
                    conflict = True
                    break
                entries[pair[0]] = pair[1]
            if conflict:
                continue
            key = tuple(entries[v] for v in new_vars)
            key1 = tuple(entries[v] for v in factor1[0])
            key2 = tuple(entries[v] for v in factor2[0])
            prob = factor1[1][key1] * factor2[1][key2]
            new_factor[key] = prob
        return new_vars, new_factor

    def topological_sort(self):
        variables = list(self.net.keys())
        variables.sort()
        s = set()
        l = []
        while len(s) < len(variables):
            for v in variables:
                if v not in s and all(x in s for x in self.net[v]['parents']):
                    s.add(v)
                    l.append(v)
        return l

    def ordering(self):
        return self.net.keys()

    def eliminate(self, var, factors, evidences):
        base_factor = None
        list_indx = []
        for factor in factors:
            vars = factor[0]
            if var in vars:
                if base_factor:
                    base_factor = self.join(base_factor, factor, evidences)
                else:
                    base_factor = factor
                list_indx.append(factor)
        for element in list_indx:
            factors.remove(element)
        base_factor = self.marginalization(var, base_factor)
        factors.append(base_factor)

    def variable_elimination(self, X, e):
        factor_vars = {}
        for v in self.net.keys():
            factor_vars[v] = self.net[v]['parents'].copy()
            factor_vars[v].append(v)
        factors = []
        for var in self.net.keys():
            factors.append(self.build_factor(var, factor_vars[var], e))
        for var in self.ordering():
            if var not in e and var != X:
                self.eliminate(var, factors, e)
        entries = {}
        list_entry = []
        list_elem = []
        for elem in e.keys():
            list_elem.append(elem)
        list_elem.append(X)
        list_elem = set(list_elem)
        temp_factor = None
        for factor in factors:
            if temp_factor:
                temp_factor = self.join(temp_factor, factor, e)
            else:
                temp_factor = factor

        temp = self.normalize(temp_factor[1])
        factor_set = set(temp_factor[0])
        if factor_set == list_elem:
            for element in temp_factor[0]:
                if element in e:
                    list_entry.append(e[element])
                else:
                    list_entry.append(True)
        res = "{:.2f}".format(temp[tuple(list_entry)])
        print(res)

    @staticmethod
    def normalize(dist):
        temp = dist.copy()
        normal = sum(dist.values())
        for entry, val in zip(dist.keys(), dist.values()):
            temp[entry] = val / normal
        return temp


def create_undirected_graph(graph: dict):
    undirected_graph = dict()
    for node in graph.keys():
        temp_list = graph[node].copy()
        for item in graph.keys():
            if node in graph[item] and item not in graph[node]:
                temp_list.append(item)
        undirected_graph[node] = temp_list
    return undirected_graph


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
    evidences = create_evidences(evidences)
    net = Net(graph, undirected_graph, graph_cpts, evidences)
    Net.reachable(net.net, x, evidences.keys(), y)
    net.variable_elimination(x, evidences)
    net.variable_elimination(y, evidences)


def create_evidences(evidences):
    result = dict()
    for var, val in evidences:
        if val == '1':
            value = True
        else:
            value = False
        result[int(var)] = value
    return result


if __name__ == '__main__':
    get_input()
