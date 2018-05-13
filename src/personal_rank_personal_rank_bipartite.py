#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: personal_rank_bipartite.py
Author: leowan(leowan)
Date: 2018/05/13 14:03:44
"""

import networkx as nx
import matplotlib.pyplot as plt

def load_graph(fn):
    """
        Load adjacent format
    """
    G = nx.read_adjlist(fn, delimiter=',')
    return G


def initialize_seed(G, seed_node):
    """
        Seeding
    """
    valdict = {}
    valdict.update((n, 0) for n in G.nodes)
    valdict[seed_node] = 1
    return valdict


def bipartite_layout(G):
    """
        Layout graph
    """
    l, r = nx.bipartite.sets(G)
    pos = {}
    pos.update((node, (1, index)) for index, node in enumerate(l))
    pos.update((node, (2, index)) for index, node in enumerate(r))
    return pos


def candidate_nodes(G, node):
    """
        Candidate nodes
    """
    _, r = nx.bipartite.sets(G)
    return set(r).difference(set(G.adj[node]))


def recommend(fn, export_graph=False):
    """
        Recommendation
    """
    fd = open('out_pr_recom.csv', 'w')
    G = load_graph(fn)
    layout =  bipartite_layout(G)
    users, items = nx.bipartite.sets(G)
    pr = nx.pagerank(G, alpha=0.85, \
        weight=None, personalization=None, max_iter=100)
    for user in users:
        recoms = sorted([(n, round(pr[n], 3)) for n in candidate_nodes(G, user)], \
            key=lambda x:x[1], reverse=True)
        fd.write('{},'.format(user))
        fd.write('|'.join([':'.join([str(v) for v in recom]) for recom in recoms]))
        fd.write('\n')
    fd.close()
    if export_graph is True:
        plt.figure()
        plt.title(fn)
        nx.draw(G, pos=layout, \
            node_size=[x * 1e4 for x in pr.values()], with_labels=True)
        plt.savefig('out_pagerank_figure.jpg')

if __name__ == "__main__":
    recommend("../data/simple_user_item.csv", export_graph=True)
        