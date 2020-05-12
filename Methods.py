# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:48:07 2020

@author: Kosta
"""
from numpy import random 
import pandas as pd
import numpy as np
from utilsNetwork import sampleNode, normalize, getTableProduct
from copy import deepcopy
import matplotlib.pyplot as plt
from itertools import permutations

class EliminationMethod(object):
    def __init__(self,network,observed,observed_values,interogative,hidden):
        self.name = "Elimination Method"
        self.network = deepcopy(network)
        self.observed = observed.copy()
        self.observed_values = observed_values
        self.interogative = interogative
        self.hidden = hidden.copy()
        
    def run(self):
        self._expandTables()
        factors = list(self.network.values())
        for i in range(len(self.hidden)):
            optimal_node, price =  self._getOptimalNode(factors)
            factor = self._associateFactors(optimal_node,factors)
            marginalized_factor = self._marginalize_factor(factor,optimal_node)
            factors.append(marginalized_factor)
        negative_prob,positive_prob = factors[0].loc[:,'probs']
        negative_prob, positive_prob = normalize(negative_prob, positive_prob)
        return negative_prob
        
    def _expandTables(self):
        """
        Expands the tables in the network, by adding the column for variables 
        for which the tables are defined. As a consequence, the rows are doubled.
        """
        for node,table in self.network.items():
            # insert new column (name = node)
            table.insert(0, node, 0)
            # concatenate table with itself
            table = pd.concat([table]*2,ignore_index=True)
            # change the probs and the value of the node column in the other half
            table.loc[len(table)/2:,'probs'] = 1 - table.loc[len(table)/2:,'probs']
            table.loc[len(table)/2:,node] = 1
            self.network[node] = table
    def _getOptimalNode(self,factors):
        """
        Searches for the next node to be eliminated for the given factors. The node
        has minimal price. Price is defined as number of unique variables in
        factors that have the node as a variable. 

        Parameters
        ----------
        factors : list of factors (DataFrame tables)
        
        Returns
        -------
        minimal_node : node with minimal price
        minimal_price : price of the returned node
        """
        minimal_price = len(self.hidden) + 1 + 1
        minimal_node = None
        for hidden in self.hidden:
            variables = set()
            for factor in factors:
                if hidden in factor.columns:
                    variables = variables | set(list(factor.columns)[:-1])
            # observed variables should not be considered
            variables = variables - set(self.observed)
            price = len(variables)
            # check if the price is minimal
            if price < minimal_price:
                minimal_price = price
                minimal_node = hidden
        self.hidden.remove(minimal_node)
        return minimal_node, minimal_price
        
    def _associateFactors(self,node,factors):
        """
        Multiplys all factors, from the list of factors, that have the node as
        variable. Associated factors are deleted from the list of factors.

        Parameters
        ----------
        node: name of the node to be eliminated
        factors : list of factors

        Returns
        -------
        product: product of associated factors (pandas DataFrame object) 
        """
        product = None
        delete_idx = []
        for i,factor in enumerate(factors):
            if (node in factor.columns): # if node is variable in factor
                product = getTableProduct(product, factor, self.observed, self.observed_values) 
                # append index of the associated factor to the delete_indicies
                delete_idx.append(i)
        # delete factors that are associated
        for i in reversed(delete_idx):
            del factors[i]
        return product
    def _marginalize_factor(self,factor,node):
        """
        Marginalizes out a factor by summing over all possible value combinations
        for that factor.
        
        Parameters
        -----
        factor: pandas DataFrame table
        node: name of the node along which the marginalization is performed
        
        Returnes
        -----
        marginalized_factor
        """
        
        # Drop the factor that we want to marginalize
        factor = factor.drop([node], axis=1)
        # Create a new table to store marginalized values
        marginalized_factor = pd.DataFrame(columns=list(factor.columns))
        n = len(factor)
        while n != 0:
            # extract first row of values to be matched
            matched_values = factor.iloc[[0],:-1].to_dict('list')
            # search for rows where values are same as matched_values
            probs = factor.loc[factor[matched_values.keys()].isin(matched_values).all(1),'probs'] 
            # sum probabilities for matched rows
            prob = sum(probs)
            # write matched_values to the marginalized_factor
            for name, value in matched_values.items():
                matched_values[name]=value[0]
            # write probabilities to the marginalized factor
            matched_values['probs'] = prob
            marginalized_factor = marginalized_factor.append(matched_values,ignore_index=True)
            # drop processed rows and reset index
            factor=factor.drop(list(probs.index),axis=0)
            factor.reset_index()
            n = n - len(probs)
        return marginalized_factor
    
        
        
#=========================================================================================
class SamplingWithRejection(object):
    def __init__(self,Nr,sampling_order,network,observed,observed_values,interogative,hidden):
        self.name = "Sampling With Rejection"
        self.sampling_order = sampling_order
        self.network = network
        self.observed = observed
        self.observed_values = observed_values
        self.interogative = interogative
        self.hidden = hidden
        self.Nr = Nr
    def run(self):
        Neff = 0
        reject_count = 0
        num_of_positive_samples = 0
        for i in range(self.Nr):
            sampled_nodes = self._getOneSample()
            if sampled_nodes is not None: # if sample is not rejected
                num_of_positive_samples += sampled_nodes[self.interogative[0]]
                Neff += 1
            else:
                reject_count += 1
        return (Neff-num_of_positive_samples)/Neff, Neff
    
    def _getOneSample(self):
        """
        Sample node one by one in self.sampling_order and return sampled values.
        If the sampled value, for the observed node, is not consistent with the 
        observed value, the function returns None.

        Returns
        -------
        sampled_nodes : dictionary where keys are names of the nodes and values
                        are sampled values
        """
        sampled_nodes = {} # dictionary which contains name of sampled node and sampled value
        for node in self.sampling_order:
            table = self.network[node]
            parents = list(table.columns)[:-1]
            if len(parents) == 0: # if there are no parents
                prob = table.iloc[0,0]
            else:
                matched_values = {} # maps parents names to their sampled values
                for parent in parents:  
                    matched_values[parent] = [sampled_nodes[parent]]
                prob = table.loc[table[parents].isin(matched_values).all(1)]
                prob = prob.loc[list(prob.index)[0],'probs']
            # sample value (0 or 1) with given prob
            sampled_value = sampleNode(prob)
            # REJECTION part
            if self._isRejected(node,sampled_value):
                return None
            # if not rejected add value to sampled values
            sampled_nodes[node] = sampled_value
        return sampled_nodes
    
    def _isRejected(self,node,sampled_value):
        """
        Check if node is in observed variables. If it is and sampled_values is 
        not consistent with observation, it returns False. Otherwise returns True.
        
        Parameters:
        node: name of the node to be checked
        sampled_value: sampled value for the node
        """

        if node in self.observed:
            if sampled_value != self.observed_values[self.observed.index(node)]:
                return True
        return False

#================================================================================================          
class WeightedLikelihood(object):
    def __init__(self,Nr,sampling_order,network,observed,observed_values,interogative,hidden):
        self.name = "Weighted Likelihood"
        self.sampling_order = sampling_order
        self.network = network
        self.observed = observed
        self.observed_values = observed_values
        self.interogative = interogative
        self.hidden = hidden
        self.Nr = Nr
        
    def run(self):
        samples = self._generateSamples()
        weights = self._getWeights()
        negative_outcome = self._determineOutcome(samples,weights,0)
        positive_outcome = self._determineOutcome(samples,weights,1)
        negative_prob, positive_prob = normalize(negative_outcome,positive_outcome)
        return negative_prob, self.Nr
    def _generateSamples(self):
        """
        Generates Nr sapmles for each node that is not in observed nodes.

        Parameters
        ----------
        Nr : number of samples 

        Returns
        -------
        samples : pandas DataFrame where each row is one sample

        """
        samples = pd.DataFrame(columns=self.hidden+self.interogative)
        for i in range(self.Nr):
            sampled_nodes = {} # dictionary which contains name of sampled node and sampled value
            for node in self.sampling_order:
                if node in self.observed: # do not sample observed variables
                    continue 
                table = self.network[node]
                parents = list(table.columns)[:-1]
                if len(parents) == 0: # if there are no parents
                    prob = table.iloc[0,0]
                else:
                    matched_values = {} # maps parents names to their sampled values
                    for parent in parents:
                        if parent in self.observed: # if the parent is observed variable, we need observed value but it does not exist in sampled nodes
                            matched_values[parent] = [self.observed_values[self.observed.index(parent)]]
                        else:
                            matched_values[parent] = [sampled_nodes[parent]]
                    prob = table.loc[table[parents].isin(matched_values).all(1)]
                    prob = prob.loc[list(prob.index)[0],'probs']
                # sample value (0 or 1) with given prob
                sampled_value = sampleNode(prob)
                sampled_nodes[node] = sampled_value
            samples = samples.append(sampled_nodes, ignore_index=True)
        return samples
        
    def _getWeights(self):
        """
        Mulitplys all tables for observed variables .It returns one probability 
        table where observed variables are deleted, and rows which are not consistent
        with observed values are also deleted.
        Returns
        -----
        weights: pandas DataFrane table
        """
        weights = self.network[self.observed[0]]
        if len(self.observed) == 1:
            return weights
        for node in self.observed[1:]:
            weights = getTableProduct(weights,self.network[node], self.observed, self.observed_values)
        return weights

    def _determineOutcome(self,samples,weights,interogative_value):
        """
        Multiply weights with corresponing number of occurances in samples.

        Parameters
        ----------
        samples : pandas Data Frame
        weights : pandas Data Frame
        interogative_value : int (0 or 1), value to be matched 

        Returns
        -------
        outcome
        """
        outcome = 0
        for i in range(len(weights)):
            matched_values = {}
            matched_values = {self.interogative[0]:[interogative_value]}
            skip_weight = False
            for column in list(weights.columns)[0:-1]:
                # check if the intergotaive value in weights is consistent with interogative_value
                if (column in self.interogative) and (weights[column].iloc[i] != interogative_value):
                    skip_weight = True
                    break;
                else: # if it is consistent, add values of variables to the matched_values
                    matched_values[column] = [weights[column].iloc[i]]
            if not skip_weight: # if the interogative value is consistent
                weight = weights['probs'].iloc[i] # get weight
                # match matched_values with samples, and count the number of matched rows
                occurances = len(samples.loc[samples[matched_values.keys()].isin(matched_values).all(1)])
                outcome += weight*occurances
        return outcome
#==========================================================================================================
class GibbsSampling(object):
    def __init__(self,Nr,sampling_order,network,observed,observed_values,interogative,hidden,irrelevant,graph):
        self.name = "Gibbs Sampling"
        self.sampling_order = sampling_order
        self.network = network
        self.observed = observed
        self.observed_values = observed_values
        self.interogative = interogative
        self.hidden = hidden
        self.irrelevant = irrelevant
        self.graph = graph
        self.Nr = Nr

    def run(self):
        # random initialization
        sample = {}
        all_samples = []
        for var in self.interogative+self.hidden:
            sample[var] = random.randint(0,2)
        for i,var in enumerate(self.observed):
            sample[var] = self.observed_values[i]
        # get samples
        for i in range(self.Nr):
            all_samples.append(sample.copy())
            sample = self._sample(sample)
        n = len(all_samples)
        # burnout
        self._burnout(all_samples)
        print("Number of burned samples = {}".format(n-len(all_samples)))
        # calculate prob
        positive_prob = 0
        for sample in all_samples:
            positive_prob += sample[self.interogative[0]]/len(all_samples)
        return 1-positive_prob, self.Nr
    def _burnout(self,all_samples):
        """
        Throws out a couple of first samples if they are not consistent with the 
        joint distribution. Joint distribution is evaluated with histogram of 
        all_samples, where each sample is transformed from binary to integer.
        (for example 0000101 is represented as 5). All samples that have minimal
        probability (different from 0) in joint distribution are considered for
        burnout. Sample are thrown away only if they are first samples.

        Parameters
        ----------
        all_samples : list of dictionaries (records)

        Returns
        -------
        None.
        """
        # trasnform binary number to integer
        samples = np.empty((len(all_samples),1),dtype=np.uint16)
        for i in range(len(all_samples)):
            value_s=""
            for value in all_samples[i].values():
                value_s += str(value) 
            samples[i,0] = int(value_s,2)    
        # generate histogram            
        H, edges, _ = plt.hist(samples,bins=range(2**len(value_s)),density=True) # 3*10**(-5)
        # extract what values to drop
        edges = edges[:-1]
        values_to_drop = edges[H==min(H[H>0])]
        # find indicies for the values_to_drop, and drop them if they are first
        num_of_deleted = 0
        for value in values_to_drop:
            indicies = list(np.where(samples==value)[0])
            indicies.sort()
            for ind in indicies:
                ind = ind - num_of_deleted
                if ind == 0:
                    del all_samples[ind]
                    num_of_deleted += 1
                else:
                    break
    def _sample(self,samples):
        """
        Samples all node that are not in observed

        Parameters
        ----------
        samples : previous samples 

        Returns
        -------
        samples : new samples
        """
        for node in self.sampling_order:
            if node in self.observed: # do not sample observed variables
                continue
            else:
                children = self._getChildren(node)
                negative_prob = self._calculateProb(node,0,children,samples)
                positive_prob = self._calculateProb(node,1,children,samples)
                negative_prob,positive_prob = normalize(negative_prob,positive_prob)
                sample = sampleNode(negative_prob)
                samples[node] = sample
        return samples
    def _getChildren(self,node):
        """
        Get children for the current node. 
        """
        children_idx = self.graph.successors(self.graph.vs['label'].index(node))
        children = self.graph.vs[children_idx]['label']
        children = list(set(children) - set(self.irrelevant))
        return children
    def _calculateProb(self, node, node_value, children, samples):
        """
        Extracts the probability from the tables in the Markov blanket for the node,
        by fixing node_value and samples.

        Parameters
        ----------
        node : name of the node 
        node_value : if 1 get positive prob, if 0 get negative prob
        children : list of children for the current node
        samples : dict with sampled values
        Returns
        -------
        product: probability 
        """
        product = 1
        for var in children+[node]:
            table = self.network[var].copy()
            if (var == node):
                var_value = node_value
            else:
                var_value = samples[var]
            if (var_value == 1):
                table.loc[:,'probs'] = 1-table.loc[:,'probs']
            # match current and previous samples and node_value with product table
            matched_values = {}
            for column in list(table.columns)[:-1]:
                if column == node:
                    matched_values[column] = [node_value]
                else:
                    matched_values[column] = [samples[column]]
            # get probability in product table determined by matched values
            if (len(matched_values) != 0):
                prob = table.loc[table[matched_values.keys()].isin(matched_values).all(1)]
            else:
                prob = table
            prob = prob.loc[list(prob.index)[0],'probs']
            # mmultiply with other probs
            product *= prob
        return product
        
        
    
        
        
        
        
        