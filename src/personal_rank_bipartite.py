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

import time
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


def candidate_nodes(G, user, items):
    """
        Candidate nodes
    """
    candidates = set(items).difference(set(G.adj[user]))
    return candidates


def load_user_items(fn):
    """
        User & Items
    """
    users = set()
    items = set()
    for line in open(fn, 'r'):
        elems = line.rstrip().split(',')
        if len(elems) > 0:
            users.add(elems[0])
        if len(elems) > 1:
            items |= set(elems[1:])
    return users, items


def recommend_strong(fn, export_graph=True):
    """
        Recommendation (strong subgraph)
    """
    fd = open('out_pagerank_recom.csv', 'w')
    G_raw = load_graph(fn)
    users, items = load_user_items(fn)
    recom_dict = {}
    for user in users:
        nodes = nx.node_connected_component(G_raw, user)
        G = G_raw.subgraph(nodes)
        itemset = set(G.nodes).difference(users)
        pr = nx.pagerank_scipy(G, alpha=0.85, \
            weight=None, personalization=None, max_iter=100)
        threshed_recoms = []
        cnodes = candidate_nodes(G, user, itemset)
        for node in cnodes:
            if pr[node] > 1e-4:
                threshed_recoms.append((node, round(pr[node], 3)))
        recoms = sorted(threshed_recoms, key=lambda x:x[1], reverse=True)
        fd.write('{},'.format(user))
        recoms_out = []
        for recom in recoms:
            if recom[1] > 0:
                recoms_out.append(recom)
        recoms_out = recoms_out[:50]
        fd.write('|'.join([':'.join([str(v) for v in recom]) for recom in recoms_out]))
        fd.write('\n')
        recom_dict[user] = recoms
    fd.close()
    if export_graph is True:
        plt.figure()
        plt.title(fn)
        pr = nx.pagerank_scipy(G, alpha=0.85, \
            weight=None, personalization=None, max_iter=100)
        nc = []
        for n in G.nodes:
            if n in users:
                nc.append('r')
            else:
                nc.append('b')
        nx.draw_networkx(G, pos=nx.spring_layout(G), \
            node_size=[x * 1e3 for x in pr.values()], node_color=nc, \
            with_labels=True)
        plt.savefig('out_figure_pagerank.jpg')
    return recom_dict


def recommend(fn, export_graph=True):
    """
        Recommendation
    """
    G = load_graph(fn)
    all_users, all_items = load_user_items(fn)
    for user in all_users:
        G.nodes[user]['type'] = 'user'
    for item in all_items:
        G.nodes[item]['type'] = 'item'
    comp_iter = nx.connected_components(G)
    comp_cnt = len(list(comp_iter))
    print('connected components: {}'.format(comp_cnt))

    comp_iter = nx.connected_components(G)
    fd = open('out_pagerank_recom.csv', 'w')
    iter_idx = 0
    for node_set in comp_iter:
        G2 = G.subgraph(node_set)
        time_s = time.time()
        pr = nx.pagerank_scipy(G2, alpha=0.85, tol=1e-06, \
            personalization=None, weight=None, max_iter=100)
        time_e = time.time()
        iter_idx += 1
        print('{}/{} subgraph ({} nodes): {} sec elapsed'.format( \
            iter_idx, comp_cnt, len(G2.nodes), time_e - time_s))

        users = []
        items = []
        for node in G2.nodes:
            if G2.nodes[node]['type'] == 'user':
                users.append(node)
            else:
                items.append(node)
        print('{} users, {} items'.format(len(users), len(items)))
        
        user_idx = 0
        for user in users:
            user_idx += 1
            if user_idx % 1000 == 0:
                print("calc user {}".format(user_idx))
            threshed_recoms = []
            cnodes = candidate_nodes(G, user, items)
            for node in cnodes:
                if pr[node] > 1e-4:
                    threshed_recoms.append((node, pr[node]))
            recoms = sorted(threshed_recoms, key=lambda x:x[1], reverse=True)
            fd.write('{},'.format(user))
            recoms_out = []
            for recom in recoms:
                if recom[1] > 0:
                    recoms_out.append(recom)
            recoms_out = recoms_out[:50]
            fd.write('|'.join([':'.join([str(v) for v in recom]) for recom in recoms_out]))
            fd.write('\n')
        print('completed make recomm for subgraph {}/{}'.format(iter_idx, comp_cnt))
        if export_graph is True:
            plt.figure()
            plt.title(fn)
            nc = []
            for n in G2.nodes:
                if n in users:
                    nc.append('r')
                else:
                    nc.append('b')
            node_values = []
            for node in G2.nodes:
                node_values.append(pr[node])
            nx.draw_networkx(G2, pos=nx.spring_layout(G2), \
                node_size=[x * 1e3 for x in node_values], node_color=nc, \
                with_labels=True)
            plt.savefig('out_figure_subgraph_{}.jpg'.format(iter_idx))
    fd.close()
    return


if __name__ == "__main__":
    recommend("../data/simple_user_item.csv", export_graph=True)

