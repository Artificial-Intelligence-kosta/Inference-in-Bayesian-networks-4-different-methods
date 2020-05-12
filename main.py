# -*- coding: utf-8 -*-
"""
Created on Sun May  3 18:00:50 2020
@author: Kosta
"""
from utilsGraph import *
from utilsNetwork import *
from Methods import *
import time
import numpy as np


def Simulation(method,Nr):
    """
    Performs Monte Carlo simulation to estimate the probability for the given method.
    
    Parameters
    -----
    method: object from one of the classes in the 'Methods.py' script
    Nrs: number of iterations for Monte Carlo simulation
    
    Returns
    -----
    avg and std for probability
    """
    estimated_probs = []
    avg_Neff = 0
    for i in range(Nr):
        print("MC iteration: {}".format(i+1))
        # run 1 MC iteration
        estimated_prob, Neff = method.run()
        estimated_probs.append(estimated_prob)
        avg_Neff += Neff/Nr
    # calculate scores
    avg_prob = sum(estimated_probs)/len(estimated_probs)
    std_prob = m.sqrt(sum([(x-avg_prob)**2 for x in estimated_probs])/len(estimated_probs))

    return avg_prob, std_prob, avg_Neff


if __name__ == "__main__":
        # Comment this line of code if you do not have 'cairo' and 'igraph' installed
        # Also comment 'from utilsGraph import *' 
        # you will also need to exclude Gibbs sampling method from the list of methods
        # because it uses graph to find succecors
        #=====================================================================
        # create the graph and save it as image
        graph_dir = "graph.png"
        names = ['learning','knowledge','skills','grades','interview','stipend',
                 'job','hobby','happiness'
                 ]
        connections = [('A','B'),('A','C'),('B','D'),('C','D'),('B','E'),('C','E'),
                       ('D','F'),('E','G'),('E','H'),('G','I'),('B','I')
                       ]
        # you can think of custom layout as rows and columns where you want to put nodes
        # if you do not know how to organize nodes put None, it will create tree layout (but text will be distorted)
        custom_layout = [(1,0), (0,1), (2,1), (0,2), (2,2), (0,3), (2,3), (3,3), (2,4)] 
        num_of_nodes = 9
        graph=createGraph(num_of_nodes, names, connections)
        plotGraph(graph,custom_layout,graph_dir)
        #=====================================================================
        
        # INITIALIZATION
        # Create tables of conditional probabilities
        A = createTable([],[],[0.2],'a-')
        B = createTable(['A'], [(0,1)], [0.9, 0.3], print_flag='b-')
        C = createTable(['A'], [(0,1)], [0.8, 0.2], print_flag='c-')
        F = createTable(['D'], [(0,1)], [0.8, 0.3], print_flag='f-')
        I = createTable(['B','G'], [(0,1),(0,1)], [0.8,0.6,0.6,0.1], print_flag='i-')
        G = createTable(['E'], [(0,1)], [0.7, 0.1], print_flag='g-')
        H = createTable(['E'], [(0,1)], [0.2, 0.9], print_flag='h-')
        D = createTable(['B','C'], [(0,1),(0,1)], [0.2,0.5,0.7,0.8], print_flag='d-')
        E = createTable(['B','C'], [(0,1),(0,1)], [0.9,0.7,0.7,0.2], print_flag='e-') 
        network = {'A':A,'B':B,'C':C,'D':D,'E':E,'F':F,'G':G,'H':H,'I':I}
        interogative = ['I']
        hidden = ['A','B','C','D','E','F','G']
        observed = ['H']
        observed_values = [0]
        sampling_order = ['A','B','C','D','E','F','G','H','I']

        # cut irrelevant nodes
        irrelevant = cutIrrelevant(network, hidden, interogative, observed)
        for node in irrelevant:
            sampling_order.remove(node)
        # create table for scores
        scores = pd.DataFrame(columns=["Method", "AVG prob","STD prob",
                                       "AVG Sample size","Execution Time",
                              ])
        # define methods and run simulation
        method1 = EliminationMethod(network,observed,observed_values,interogative,hidden)
        method2 = SamplingWithRejection(88,sampling_order,network,observed,observed_values,interogative,hidden)
        method3 = WeightedLikelihood(60,sampling_order,network,observed,observed_values,interogative,hidden)
        method4 = GibbsSampling(100,sampling_order,network,observed,observed_values,interogative,hidden,irrelevant,graph)
        methods = [method1,method2,method3,method4]
        Nr = 100
        for method in methods:
            print('---'*10)
            print('Executing {} method'.format(method.name))
            tic = time.time()
            if (method.name == 'Elimination Method'):
                negative_prob = method1.run()
                avg_prob, std_prob, avg_sample_size = negative_prob, np.nan, np.nan
                print("Exact probability: {:.5f}".format(avg_prob))  
            else:
                avg_prob, std_prob, avg_sample_size = Simulation(method,Nr)
                print("Average probability: {:.5f}".format(avg_prob))
            toc = time.time()
            print("Average sample size: {:.3f}".format(avg_sample_size))
            print("Time: {:.3f}s".format(toc-tic))
            print(std_prob)
            # save the data
            scores = scores.append(
                {"Method": method.name,
                 "AVG prob": avg_prob,
                 "STD prob": std_prob,
                 "AVG Sample size": avg_sample_size,
                 "Execution Time": toc-tic}, 
                ignore_index = True
                )
        # save the scores
        scores.to_csv(os.path.join(os.getcwd(), "scores.csv"))
        
        
        
        
        