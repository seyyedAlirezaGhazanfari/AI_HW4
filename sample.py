import itertools
import os
import re
from collections import defaultdict
import json
import numpy as np
import shutil
import matplotlib.pyplot as plt
SIZE = 10000

class Sampling:
    @staticmethod
    def get_prior_result(requests: dict, evidences: dict, bn: dict, n: int):
        result = defaultdict(int)
        variables = None
        for i in range(n):
            sample = Sampling.prior_sampling(bn)
            result[tuple(sample.values())] += 1
            variables = tuple(sample.keys())
        result = normalize(result)
        conclusion = 0
        for sample in result.keys():
            sample = tuple(sample)
            temp = dict()
            for i, var in enumerate(variables):
                temp[var] = sample[i]
            if all(item in temp.items() for item in requests.items()) and all(
                    item in temp.items() for item in evidences.items()):
                conclusion += result[sample]
        return conclusion

    @staticmethod
    def prior_sampling(bn: dict):
        variables = topological_sort(bn)
        result_total = dict()
        for var in variables:
            val = Sampling.conditional_sampler(var, {x: result_total[x] for x in bn[var]['parents']}, bn)
            result_total[var] = val
        return result_total

    @staticmethod
    def rejection_sampling(requests: dict, bn: dict, evidences: dict, n: int):
        result = defaultdict(int)
        for i in range(n):
            sample = Sampling.prior_sampling(bn)
            if all(item in sample.items() for item in evidences.items()):
                request = tuple(sample[x] for x in requests.keys())
                result[request] += 1
        return normalize(result)[tuple(requests.values())]

    @staticmethod
    def likelihood_sampling(requests, evidences, bn, n):
        res = defaultdict(int)
        for i in range(n):
            x, w = Sampling.weighted_sampling(bn, evidences)
            request = tuple(x[a] for a in requests.keys())
            res[request] += w
        return normalize(res)[tuple(requests.values())]

    @staticmethod
    def weighted_sampling(bn, evidences):
        w = 1
        sample = {}
        for var in topological_sort(bn):
            temp = [var]
            temp.extend(bn[var]['parents'])
            factor = build_factor(var, temp, bn)
            if var in evidences.keys():
                entity = dict()
                for variable in factor[0]:
                    if variable in sample.keys():
                        entity[variable] = sample[variable]
                    else:
                        entity[variable] = evidences[variable]
                w *= factor[1][tuple(entity.values())]
            else:
                s = dict()
                for parent in bn[var]['parents']:
                    if parent in evidences.keys():
                        s[parent] = evidences[parent]
                    else:
                        s[parent] = sample[parent]
                x = Sampling.conditional_sampler(var, s, bn)
                sample[var] = x
        return sample, w

    @staticmethod
    def conditional_sampler(var, parents, bn):
        factor_vars = [var]
        factor_vars.extend(parents.keys())
        factor = Sampling.make_factor(var, factor_vars, parents, bn)
        sample = np.random.choice([1, 0], p=list(factor[1].values()))
        return sample

    @staticmethod
    def get_parents(var, e, non_e, bn):
        parents = bn[var]['parents']
        res = dict()
        for parent in parents:
            if parent in e.keys():
                res[parent] = e[parent]
            elif parent in non_e.keys():
                res[parent] = non_e[parent]
        return res

    @staticmethod
    def gibbs_sampling(requests: dict, evidences: dict, bn: dict, n: int):
        res = defaultdict(int)
        non_ev = dict()
        for node in bn.keys():
            if node not in evidences.keys():
                non_ev[node] = bool(np.random.randint(0, 1, 1))
        for j in range(n):
            for zi in non_ev.keys():
                parents = Sampling.get_parents(zi, evidences, non_ev, bn)
                non_ev[zi] = Sampling.conditional_sampler(zi, parents, bn)
                request = tuple(non_ev[a] for a in requests.keys())
                res[request] += 1
        return normalize(res)[tuple(requests.values())]

    @staticmethod
    def make_factor(var, factor_vars, evidences: dict, bn: dict):
        params = []
        params.extend(factor_vars)
        perms = gen_permutations(len(params))
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
                entries[key] = query_given(var, e, bn)
        return params, entries


def topological_sort(net):
    variables = list(net.keys())
    variables.sort()
    s = set()
    l = []
    while len(s) < len(variables):
        for v in variables:
            if v not in s and all(x in s for x in net[v]['parents']):
                s.add(v)
                l.append(v)
    return l


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


def query_given(y, e, net):
    if net[y]['prob'] != -1:
        prob = net[y]['prob'] if e[y] else 1 - net[y]['prob']
    else:
        parents = tuple(e[p] for p in net[y]['parents'])
        prob = net[y]['cpt'][parents] if e[y] else 1 - net[y]['cpt'][parents]
    return prob


def gen_permutations(length):
    assert (length >= 0)
    l = [True, False]
    perms = list(itertools.product(l, repeat=length))
    assert (len(perms) == pow(2, length))
    return perms


def build_factor(var, factor_vars, net):
    params = []
    params.extend(factor_vars)
    perms = gen_permutations(len(params))
    entries = dict()
    e = dict()
    for perm in perms:
        for pair in zip(params, perm):
            e[pair[0]] = pair[1]
        key = tuple(e[elem] for elem in e.keys())
        entries[key] = query_given(var, e, net)
    return params, entries


def create_parent_net(net, name, parents):
    net[name]['parents'] = parents
    net[name]['prob'] = -1
    net[name]['children'] = []


def create_children_net(net):
    for node in net.keys():
        for parent in net[node]['parents']:
            net[parent]['children'].append(node)


def normalize(dist):
    temp = dist.copy()
    normal = sum(dist.values())
    for entry, val in zip(dist.keys(), dist.values()):
        temp[entry] = val / normal
    return temp


def join(factor1, factor2):
    vars1 = factor1[0]
    vars2 = factor2[0]
    new_vars = list()
    new_vars.extend(vars2)
    new_vars.extend(vars1)
    new_vars = set(new_vars)
    new_factor = dict()
    perms = gen_permutations(len(new_vars))
    for perm in perms:
        entries = {}
        for pair in zip(new_vars, perm):
            entries[pair[0]] = pair[1]
        key = tuple(entries[v] for v in new_vars)
        key1 = tuple(entries[v] for v in factor1[0])
        key2 = tuple(entries[v] for v in factor2[0])
        prob = factor1[1][key1] * factor2[1][key2]
        new_factor[key] = prob
    return new_vars, new_factor


def calculate_p(requests, evidences, net):
    hidden_vars = (set(net.keys()).difference(requests.keys())).difference(evidences.keys())
    factors = []
    for var in net.keys():
        temp = [var]
        temp.extend(net[var]['parents'])
        factors.append(build_factor(var, temp, net))
    base_factor = None
    for factor in factors:
        if base_factor is None:
            base_factor = factor
        else:
            base_factor = join(base_factor, factor)
    for hidden in hidden_vars:
        base_factor = marginalization(hidden, base_factor)
    result_indx = []
    for i, var in enumerate(base_factor[0]):
        if var in requests.keys():
            result_indx.append(requests[var])
        if var in evidences.keys():
            result_indx.append(evidences[var])
            delete_list = []
            for perm in base_factor[1].keys():
                if perm[i] != evidences[var]:
                    delete_list.append(perm)
            for perm in delete_list:
                base_factor[1].pop(perm)

    temp = normalize(base_factor[1])
    return temp[tuple(result_indx)]


def input_parser(input_file):
    n = int(input_file.readline().strip().split("\n")[0])
    net = defaultdict(dict)
    for _ in range(n):
        name = input_file.readline().strip().split("\n")[0]
        next_line = input_file.readline().strip().split("\n")[0]
        is_not_parent = re.match('\d+\.\d+', next_line)
        if is_not_parent:
            p = float(next_line)
            net[name]['prob'] = p
            net[name]['cpt'] = []
            net[name]['parents'] = []
            net[name]['children'] = []
        else:
            parents = list(next_line.strip().split(" "))
            l = [False, False]
            create_parent_net(net, name, parents)
            entity = {}
            for _ in range(2 ** len(parents)):
                input_line = list(input_file.readline().strip().split("\n")[0].split(" "))
                perms = tuple(map(bool, list(map(int, input_line[:-1]))))
                p = float(input_line[-1])
                entity[perms] = p
            net[name]['cpt'] = entity
    create_children_net(net)
    return net


def parse_input():
    dirs = os.listdir("./inputs/")
    for dir in dirs:
        input_file = open("./inputs/" + dir + "/" + "input.txt", "r")
        input_q = open("./inputs/" + dir + "/" + "q_input.txt", "r")
        net = input_parser(input_file)
        bn = net
        list_of_queries = json.load(input_q)
        i = 1
        result_p, res_prior, res_reject, res_like, res_gibbs = dict(), dict(), dict(), dict(), dict()
        for query in list_of_queries:
            requests = query[0]
            evidences = query[1]
            result_p[i] = calculate_p(requests, evidences, net)
            res_prior[i] = Sampling.get_prior_result(requests, evidences, bn, SIZE)
            res_reject[i] = Sampling.rejection_sampling(requests, bn, evidences, SIZE)
            res_like[i] = Sampling.likelihood_sampling(requests, evidences, bn, SIZE)
            res_gibbs[i] = Sampling.gibbs_sampling(requests, evidences, bn, SIZE)
            i += 1
        file = open("./output/" + dir + ".txt", "w")
        i = 1
        query_nums, q_prior, q_reject, q_like, q_gibbs = [], [], [], [], []
        for p, prior, reject, like, gibbs in zip(
                result_p.values(),
                res_prior.values(),
                res_reject.values(),
                res_like.values(),
                res_gibbs.values()):
            query_nums.append(i)
            q_prior.append(abs(p - prior))
            q_reject.append(abs(p - reject))
            q_like.append(abs(p - like))
            q_gibbs.append(abs(p - gibbs))
            file.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                p,
                abs(p - prior),
                abs(p - reject),
                abs(p - like),
                abs(p - gibbs)
            )
            )
            i += 1
        file.close()
        plt.plot(query_nums, q_prior, color="red", label="prior")
        plt.plot(query_nums, q_reject, color="blue", label="reject")
        plt.plot(query_nums, q_like, color="black", label="likelihood")
        plt.plot(query_nums, q_gibbs, color="pink", label="gibbs")
        plt.xlabel("#Q")
        plt.ylabel("MAE")
        plt.legend(loc="upper left")
        plt.savefig("./output/"+dir+".png")
        plt.close()


if __name__ == '__main__':
    try:
        os.mkdir("./output")
    except FileExistsError:
        shutil.rmtree("./output")
        os.mkdir("./output")
    parse_input()
