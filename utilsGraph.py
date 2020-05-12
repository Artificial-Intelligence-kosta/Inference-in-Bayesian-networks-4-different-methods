# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:25:09 2020
@author: Kosta
"""
from igraph import *
from igraph.drawing.text import TextDrawer
import string
import math as m
import cairo
def letter_to_num(pairs):
    pairs_of_num = []
    for pair in pairs:
        p = (ord(pair[0])-65, ord(pair[1])-65)
        pairs_of_num.append(p)
    return pairs_of_num
    
def createGraph(num_of_nodes,names,connections):
    """
    Create igraph graph object
    """
    graph = Graph().as_directed()
    graph.add_vertices(num_of_nodes)
    graph.add_edges(letter_to_num(connections))
    # set attributes to vertexes
    node_labels = list(string.ascii_uppercase)[0:num_of_nodes]
    graph.vs['label'] = node_labels
    graph.vs['name'] = names
    return graph
def plotGraph(graph,mylayout,save_dir):
    """
    Plots and saves the graph
    """
    visual_style = {}
    # Set bbox and margin
    visual_style["bbox"] = (3000,3000)
    visual_style["margin"] = 300
    # Set vertex colours
    visual_style["vertex_color"] = 'pink'
    visual_style["vertex_size"] = 200
    visual_style["vertex_label_size"] = 100
    visual_style["vertex_label"] = graph.vs["label"]
    # edges
    visual_style["edge_curved"] = False
    visual_style["edge_width"] = 10
    visual_style["edge_arrow_size"] = 7
    # Set the layout
    if mylayout is not None:
        visual_style["layout"] = mylayout
    else:
        visual_style["layout"] = graph.layout('tree')
        mylayout = graph.layout('tree')
    plt = plot(graph, save_dir, **visual_style)
    # Make the plot draw itself on the Cairo surface
    plt.redraw()
    # Grab the surface, construct a drawing context and a TextDrawer
    ctx = cairo.Context(plt.surface)
    ctx.set_font_size(60)
    for i,coords in enumerate(mylayout):
        drawer = TextDrawer(ctx, graph.vs["name"][i], halign=TextDrawer.LEFT)
        x = (coords[0]*800+300+120) # bbox and margin influence this
        y = (coords[1]*600+300)
        drawer.draw_at(x, y, width=600)

     # Save the plot
    plt.save()
    print("Graph is saved to {}".format(save_dir))

    
    
