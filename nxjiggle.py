import os
import numpy as np
import pandas as pd
import glob as glob
import matplotlib.pyplot as plt
# import scipy.stats
# import seaborn as sns
# from sklearn import metrics as skm
import math
from collections import Counter
import networkx as nx
import matplotlib.lines as mlines
from itertools import combinations
import matplotlib

def create_bboxes(pos, text_scale = 14):
    node_bboxes = {}
    for node in list(pos.keys()):
        xy = pos[node]
        x = xy[0]
        y = xy[1]
        bbox_length = len(node) * text_scale / 1000
        bbox_height = 2 * text_scale / 1000
        bbox = np.array([(x-bbox_length/2,y-bbox_height/2),bbox_length, bbox_height])
        node_bboxes[node] = bbox
    return node_bboxes

def visualize(subG, pos, node_bboxes):
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)

    nodelabels = dict([(x,x) for x in subG.nodes(data=False)])

    nx.draw_networkx_labels(subG, pos, labels=nodelabels, ax=ax)

    for node in list(node_bboxes.keys()):
        bbox = node_bboxes[node]
        rect = matplotlib.patches.Rectangle(bbox[0],
                                            bbox[1],
                                            bbox[2],
                                            edgecolor = 'black',
                                            fill=False)
        ax.add_patch(rect)

    plt.show()

def visualize_bboxes(node_bboxes):
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)

    for node in list(node_bboxes.keys()):
        bbox = node_bboxes[node]
        rect = matplotlib.patches.Rectangle(bbox[0],
                                            bbox[1],
                                            bbox[2],
                                            edgecolor = 'black',
                                            fill=False)
        ax.add_patch(rect)

    plt.show()


def calculate_overlap(node1, node2, node_bboxes):

    bbox1 = node_bboxes[node1]
    bbox2 = node_bboxes[node2]

    xmin1 = bbox1[0][0]
    ymin1 = bbox1[0][1]
    xmax1 = xmin1 + bbox1[1]
    ymax1 = ymin1 + bbox1[2]

    xmin2 = bbox2[0][0]
    ymin2 = bbox2[0][1]
    xmax2 = xmin2 + bbox2[1]
    ymax2 = ymin2 + bbox2[2]

    dx = min(xmax1,xmax2) - max(xmin1,xmin2)
    dy = min(ymax1,ymax2) - max(ymin1,ymin2)

    if (dx < 0) | (dy < 0):
        return 0
    else:
        return dx*dy

def calculate_total_overlap(node_bboxes):
    all_pairs = [x for x in combinations(list(node_bboxes.keys()),2)]
    total_overlap = 0
    node_bboxes = node_bboxes ##
    for pair in all_pairs:
        overlap = calculate_overlap(pair[0], pair[1], node_bboxes)
        total_overlap += overlap
    return total_overlap

def calculate_node_overlap(node_bboxes):
    # node_bboxes = node_bboxes ##
    node_list = list(node_bboxes.keys())
    node_overlap_dict = {}
    for node in node_list:
        node_overlap = 0
        for othernode in node_list:
            if node != othernode:
                node_overlap += calculate_overlap(node, othernode, node_bboxes)
        node_overlap_dict[node] = node_overlap
    return node_overlap_dict

def jiggle(node_bboxes, node_overlap_dict, scale):
    node_list = list(node_bboxes.keys())
    new_node_bboxes = {}
    for node in node_list:
        bbox = node_bboxes[node]
        overlap = node_overlap_dict[node]
        dx = (np.random.randn()*overlap + np.random.randn()*0.00001) * scale
        dy = (np.random.randn()*overlap + np.random.randn()*0.00001) * scale
        print(dx,dy)
        new_bbox = np.array([(bbox[0][0]+dx, bbox[0][1]+dy), bbox[1], bbox[2]])
        new_node_bboxes[node] = new_bbox
    return new_node_bboxes

def bbox2pos(bbox):
    return (bbox[0]+bbox[1]/2, bbox[1]+bbox[2]/2)

def bboxes2pos(node_bboxes):
    new_pos = {}
    for node in list(node_bboxes.keys()):
        xy = bbox2pos(node_bboxes[node])
        new_pos[node] = xy
    return new_pos

def fit_jiggle(node_bboxes, num_iter = 500):
    best_node_bboxes = node_bboxes
    for i in range(num_iter):

        current_node_overlap = calculate_total_overlap(best_node_bboxes) # current total overlap

        node_overlap_dict = calculate_node_overlap(best_node_bboxes) # current overlap per node
        new_node_bboxes = jiggle(best_node_bboxes, node_overlap_dict, scale=100000) # make new bboxes
        if new_node_bboxes['ADORA1'][0] == best_node_bboxes['ADORA1'][0]:
            print('No change')
            break

        new_node_overlap = calculate_total_overlap(new_node_bboxes) # new total overlap

        print(f'Iteration {i}, current overlap {current_node_overlap}, new overlap {new_node_overlap}')

        if new_node_overlap < current_node_overlap:
            best_node_bboxes = new_node_bboxes
            print(f'Saving better version')

        if new_node_overlap <= 0:
            break

    return best_node_bboxes


def fit_jiggle(node_bboxes, num_iter = 500):
    best_node_bboxes = node_bboxes
    for i in range(num_iter):

        current_node_overlap = calculate_total_overlap(best_node_bboxes) # current total overlap

        node_overlap_dict = calculate_node_overlap(best_node_bboxes) # current overlap per node
        new_node_bboxes = jiggle(best_node_bboxes, node_overlap_dict, scale=100000) # make new bboxes
        # if new_node_bboxes['ADORA1'][0] == best_node_bboxes['ADORA1'][0]:
        #     print('No change')
        #     break

        new_node_overlap = calculate_total_overlap(new_node_bboxes) # new total overlap

        print(f'Iteration {i}, current overlap {current_node_overlap}, new overlap {new_node_overlap}')

        if new_node_overlap < current_node_overlap:
            best_node_bboxes = new_node_bboxes
            print(f'Saving better version')

        if new_node_overlap <= 0:
            break

    return best_node_bboxes


def main_jiggle(G, pos, text_scale = 14, num_iter = 500):

    # Make bboxes
    node_bboxes = create_bboxes(pos, text_scale = 14)

    # Check that they look correct
    visualize(G, pos, node_bboxes)

    # Fit a jiggle
    new_bboxes = fit_jiggle(node_bboxes, num_iter = 500)

    # Inspect new bboxes
    visualize(G, pos, node_bboxes)

    # Convert back to node positions
    new_pos = bboxes2pos(new_bboxes)

    return new_pos