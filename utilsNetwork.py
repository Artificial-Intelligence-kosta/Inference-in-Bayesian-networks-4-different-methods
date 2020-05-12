# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:24:04 2020
@author: Kosta
"""
import pandas as pd
from numpy import random 

def createTable(var_names, var_values, probs, print_flag=None):
    """
    Creates a pandas DataFrame with conditional probabilities.
    
    Parameters
    ----------
    var_names : list of variable names that have effect on variable for which you make table.
    var_values : list of tuples, where each tuple corresponds to the variable in var_names.
                 Elements in tuple represent the order of encoding of values for variable.
                 Index of the tuple in the var_values list is the level of the binary encoded variable.
    probs : list of probabilities. They will be placed in the DataFrame in the same order.
            Length of prabs should be 2^len(var_names)
    print_flag : string that represents the variable for which the table is created.
                 Default is None which does not print anything.
                 
    Returns
    -------
    table : pandas DataFrame object
    """
    if len(var_names) != len(var_values):
        raise ValueError("'var_names' and 'var_values' should have same length")
    if 2**len(var_names) != len(probs):
        raise ValueError("length of probs should be 2^len(var_names)")
    
    table = pd.DataFrame(columns=var_names+['probs'])
    table['probs'] = probs
    # create values for variables (1 for plus, 0 for minus)
    n = len(var_names)
    for level,values in enumerate(var_values):
        j = 0 # iterates over rows
        value_ind = 0 
        while(j != 2**n):
            delta_j = 2**(n-level-1)
            table.loc[j:j+delta_j,var_names[level]] = values[value_ind]
            value_ind = 1-value_ind # swtich between zeros and ones
            j+=delta_j
    if print_flag is not None:
        print('---'*7)
        print("Table for P({}|".format(print_flag),end='')
        print(",".join(str(e) for e in var_names)+'):')
        print(table.to_string(index=False))
    return table

def observeInNetwork(network,nodes,values):
    """
    Observes the network (probability tables) in order to remove rows from tables 
    where the observed variables are not consistent with observed values.
    
    Parameters
    ----------
    network: dictionary where values are DataFrame tables
    nodes: list of names of observed nodes
    values : List of correspodnding observed values 

    Returns
    -------
    observed_network
    """
    observed_network = {}
    for name,table in network.items():
        for node,value in zip(nodes,values):
            if node in table.columns: # if observed variable exist in table
                table = table[table[node] == value] # keep rows where observed variable has value value
        observed_network[name]=table
    return observed_network

    
            
def cutIrrelevant(network, hidden, interogative, observed):
    """
    Cuts the irrelevant nodes in the network, by removing leaf nodes if they are in hidden nodes.
    If the node is removed, the function is again called recursively.
    
    Parameters
    ----------
    network: dictionary where values are DataFrame tables
    hidden: list of names of hidden nodes
    interrogative : List with name of interrogative node
    observed: List of names of observed nodes
    Returns
    -------
    irrelevant: list of irrelevant nodes
    """
    irrelevant = []
    all_nodes = hidden + interogative + observed 
    parents=[]
    for node in all_nodes:
        parents += list(network[node].columns)[:-1]
    for node in all_nodes:
        if (node not in parents) and (node in hidden):
            hidden.remove(node)
            del network[node]
            print('Node {} is irrelevant, it is removed.'.format(node))
            irrelevant.append(node)
            irrelevant += cutIrrelevant(network, hidden, interogative, observed)
    return irrelevant

def sampleNode(prob):
    """
    Samples the 0 with given probability, otherwise returns 1.
    
    Parameters
    -----
    prob: probability for negative value
    
    Returns
    -----
    Samples value (0 or 1)
    """
    p  = random.random()
    if prob > p:
        return 0 # because table is conditional probability for negative outcome 
    else:
        return 1
def normalize(outcome1,outcome2):
    """
    Normalizes outcomes so that they sum up to 1.
    """
    alpha = outcome1 + outcome2
    return outcome1/alpha, outcome2/alpha

def getTableProduct(table1,table2,observed,observed_values):
    """
    Get the product of 2 tables, by multiplying where variable values overlap.
    Also, delete rows which are not consisent with observed values, and after that
    delete the columns with observed variables.
    
    Parameters
    -----
    table1: pandas DataFrame
    table2: pandas DataFrame
    observed: list of names of observed variables
    observed_values: list of corresponding observed values
    
    Returns
    -----
    product: pandas DataFrame
    
    """
    if (table1 is None):
        return table2
    all_columns = list((set(table1.columns) | set(table2.columns)) - set(["probs"]))
    shared_columns = list((set(table1.columns) & set(table2.columns)) - set(["probs"]))
    product = pd.DataFrame(columns=all_columns)
    probs = []
    k = 0
    for i in range(len(table1)):
        for j in range(len(table2)):
            if sum(table1[shared_columns].iloc[i] == table2[shared_columns].iloc[j]) == len(shared_columns):
                prob = table1["probs"].iloc[i] * table2["probs"].iloc[j]
                probs.append(prob)
                values = []
                for name in all_columns:
                    if name in table1.columns:
                        values.append(table1[name].iloc[i])
                    else:
                        values.append(table2[name].iloc[j])
                product.loc[k] = values
                k += 1
    product["probs"] = probs
    # the product table is now created but it may contain some variables that are observed, so we need to handle that
    for var in list(product.columns)[:-1]:
        if var in observed: # if variable is observed variable
            product = product[product[var] == observed_values[observed.index(var)]]
            del product[var]
    return product


