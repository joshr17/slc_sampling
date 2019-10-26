#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:07:51 2019

@author: joshuarobinson
"""

import numpy as np
import math
import utils
import random
from numpy.linalg import inv
from scipy.linalg import eigh

# MCDPP sampler
def compute_PSRF(chain_averages, within_variance, chain_len):
    PSRF_num = chain_averages.shape[0]
    B = np.var(chain_averages, axis = 0)
    W = np.average(within_variance, axis = 0)
    
    V = ( (chain_len-1) / chain_len) * sum(W) + ( (PSRF_num+1)/ (PSRF_num) ) * sum(B)
    PSRF = V / sum(W) 

    return PSRF




def compute_multivariate_PSRF(chains, step_num, PSRF_num, mix_step, N):
    chain_averages = (mix_step / step_num ) * chains.mean(axis = 1, keepdims = True)
   
    centered_chains = chains - chain_averages
    N = chains.shape[2]
    
    W = np.zeros((N,N))
    for j in range(PSRF_num):
        for l in range(step_num):
            W = W + np.outer(centered_chains[j,l], centered_chains[j,l])
    

    W = (1/PSRF_num)*(1 / (step_num -1))* W    
    
    chain_avg_1 = (mix_step / step_num ) * chains.mean(axis = 1)
    centered_chain_avg_1 = chain_avg_1 - chain_avg_1.mean(axis = 0, keepdims = True)

    
    Bn = np.zeros((N,N))
    for j in range(PSRF_num):
        Bn = Bn + np.outer(centered_chain_avg_1[j], centered_chain_avg_1[j])

    Bn = (1 / (PSRF_num - 1))* Bn


    WBn = np.matmul( inv(W), Bn)

    lmbda = eigh(WBn, eigvals = (N, N) )
    
    PSRF = ((step_num - 1) / step_num) + ( (PSRF_num + 1)/ PSRF_num) * lmbda 
    return PSRF


def dpp(L, alpha):
    def function(subset):
        return np.power(np.linalg.det(np.copy(L[np.ix_(subset, subset)])), alpha)   
    return function


def sym_homog_dpp(L, alpha):
    N = L.shape[0]
    def function(subset):
        if not type(subset) == np.ndarray:
            subset = np.array(subset)
            
        intersected_subset = subset[subset < N]
        return np.power(np.linalg.det(np.copy(L[np.ix_(intersected_subset, intersected_subset)])), alpha)
    
    return function


#implementing the base exchange walk for a homogeneous distribution
def base_walk(distribution, mix_step, N, k, init_sample=None, flag_gpu=False, track_PSRF = False, \
              PSRF_freq = None, PSRF_tol = None, PSRF_num = None):
    

    sample = init_sample
    tic_len = mix_step // 5

    if not track_PSRF:
        if sample is None:
            sample = np.random.permutation(N)[:k]
            sample = np.sort(sample)

        chain = []
        
        for i in range(mix_step):
            if (i+1) % tic_len == 0:
                print('{}-th iteration.'.format(i+1))
       
            drop_index= random.randint(0,k-1)
            sample = np.delete(sample, drop_index)
            sample_compl = np.setxor1d(np.array(range(N)), sample)
            
            proposal_probabilities = np.zeros(N-k+1)
    
            for index, add in enumerate(sample_compl):
                proposed_move = np.append(sample, add)
                proposal_probabilities[index] = distribution(proposed_move)
                
            proposal_probabilities = proposal_probabilities/ sum(proposal_probabilities)
            
            chosen_index = utils.weighted_random(proposal_probabilities)
            sample = np.append(sample, sample_compl[chosen_index])
            
            chain.append(sample)
            
        return chain

    elif track_PSRF:
        if sample is None:
            sample = np.zeros((PSRF_num,k))
            for j in range(PSRF_num):
                sample[j] = np.random.permutation(N)[:k]
                sample[j] = np.sort(sample[j])
                
        chains = np.zeros((PSRF_num, mix_step,k))      
        
        for i in range(mix_step):
            if (i+1) % tic_len == 0:
                print('{}-th iteration.'.format(i+1))
             
            for j in range(PSRF_num):

                drop_index= random.randint(0,k-1)
                sample_drop = np.delete(sample[j], drop_index)
                sample_compl = np.setxor1d(np.array(range(N+k)), sample_drop)
                
                proposal_probabilities = np.zeros(N+1)
        
                for index, add in enumerate(sample_compl):
                    proposed_move = np.append(sample_drop, add)
                    proposed_move = list(map(int, proposed_move))
                    proposal_probabilities[index] = distribution(proposed_move)
                    
                proposal_probabilities = proposal_probabilities/ sum(proposal_probabilities)
                
                chosen_index = utils.weighted_random(proposal_probabilities)
                sample[j] = np.append(sample_drop, sample_compl[chosen_index])
                
                
                chains[j,i,:] = sample[j]


            if (i+1) % PSRF_freq == 0:
                chains_binary = np.zeros((PSRF_num, mix_step, N))
                within_variance = np.zeros((PSRF_num, N))
                chain_averages = np.zeros((PSRF_num, N))

                for j in range(PSRF_num):
                    
                    chain = utils.set_to_binary(chains[j],N)  
              
                    chains_binary[j] = chain
    
                    chain_averages[j] = (mix_step / i ) * np.average(chain, axis = 0)
                    within_variance[j] = (mix_step / i ) * np.var(chain, axis = 0)
                    
   
                PSRF = compute_PSRF(chain_averages, within_variance, PSRF_num, N)
              
                if PSRF < PSRF_tol :
                    return i
                
        return mix_step            


#implementing Metropolis-Hastings with the base exchange walk as proposal and 
#nonhomogeneous stationary distribution
def base_walk_proposal_scaled(distribution, mix_step, N, k, init_sample=None, flag_gpu=False, track_PSRF = False, \
              PSRF_freq = None, PSRF_tol = None, PSRF_num = None):
    

    sample = init_sample
    tic_len = mix_step // 5

    if not track_PSRF:
        if sample is None:
            sample = np.random.permutation(N+k)[:k]
            sample = np.sort(sample)

        chain = []
        
        for i in range(mix_step):
            if (i+1) % tic_len == 0:
                print('{}-th iteration.'.format(i+1))
       
            drop_index= random.randint(0,k-1)
            sample_drop = np.delete(sample, drop_index)
            sample_compl = np.setxor1d(np.array(range(N+k)), sample)
            
            current_len = sample[sample < N].shape[0]
            
            proposal_probabilities = np.zeros(N+1)

            for index, add in enumerate(sample_compl):
                proposed_move = np.append(sample, add)
                proposed_len = proposed_move[proposed_move < N].shape[0]

                prob = distribution(proposed_move)
                        
                if proposed_len == current_len:
                   proposal_probabilities[index] = prob
                
                elif proposed_len == current_len + 1:
                   proposal_probabilities[index] = proposed_len * prob
                    
                elif proposed_len == current_len - 1:
                   proposal_probabilities[index] = proposed_len * prob 
            
     
            proposal_probabilities = proposal_probabilities/ sum(proposal_probabilities)
                   
            proposed_index = utils.weighted_random(proposal_probabilities)
            proposed_sample = np.append(sample_drop, sample_compl[proposed_index])
            proposed_sample = np.array(list(map(int, proposed_sample)))

            proposed_len = proposed_sample[proposed_sample < N].shape[0]

            if current_len < proposed_len:
                acceptance_prob = 1/(k - current_len)
        
                ran = np.random.uniform()
                if ran < acceptance_prob:
                    sample = proposed_sample
  
            else:
                sample = proposed_sample   
                
            chain.append(sample)
            
        return chain

    elif track_PSRF:
        if sample is None:
            sample = np.zeros((PSRF_num,k))
            for j in range(PSRF_num):
                sample[j] = np.random.permutation(N+k)[:k]
                sample[j] = np.sort(sample[j])
                
        chains = np.zeros((PSRF_num, mix_step,k))      
        

        PSRFs = []
        mix_time = mix_step
        first = True
        for i in range(mix_step):
            if (i+1) % tic_len == 0:
                print('{}-th iteration.'.format(i+1))


            for j in range(PSRF_num):

                drop_index= random.randint(0,k-1)
                sample_drop = np.delete(sample[j], drop_index)
                sample_compl = np.setxor1d(np.array(range(N+k)), sample_drop)
         
                current_sample = sample[j]
                current_len = current_sample[current_sample < N].shape[0]

                proposal_probabilities = np.zeros(N+1)

                for index, add in enumerate(sample_compl):
                    proposed_move = np.append(sample_drop, add)
                    proposed_move = np.array(list(map(int, proposed_move)))
                    proposed_len = proposed_move[proposed_move < N].shape[0]
                   
                    prob = distribution(proposed_move)
                    
                    if proposed_len == current_len:
                       proposal_probabilities[index] = prob
                    
                    elif proposed_len == current_len + 1:
                       proposal_probabilities[index] = ( math.exp(1) / k ) *(k - current_len ) * prob
                        
                    elif proposed_len == current_len - 1:
                       proposal_probabilities[index] =  ( ( k / math.exp(1) ) * prob ) / (k - proposed_len) 
               
                proposal_probabilities = proposal_probabilities/ sum(proposal_probabilities)
                proposed_index = utils.weighted_random(proposal_probabilities)
                proposed_sample = np.append(sample_drop, sample_compl[proposed_index])
                
                current_len = current_sample[current_sample < N].shape[0]
                
                proposed_sample = np.array(proposed_sample)
                proposed_len = proposed_sample[proposed_sample < N].shape[0]

                if current_len < proposed_len:

                    acceptance_prob = min( 1, ( k / math.exp(1) ) /(k - current_len) )
                    ran = np.random.uniform()
                    if ran < acceptance_prob:
                        sample[j] = proposed_sample
                        
                elif current_len > proposed_len:
        
                    acceptance_prob = min( 1, (  math.exp(1) / k) * (k - current_len + 1) )
                    ran = np.random.uniform()
                    if ran < acceptance_prob:
                        sample[j] = proposed_sample

                        
                else:
                    sample[j] = proposed_sample
                    
                chains[j,i,:] = sample[j]
             

            if (i+1) % PSRF_freq == 0:
                chains_binary = np.zeros((PSRF_num, mix_step, N+k))
                within_variance = np.zeros((PSRF_num, N+k))
                chain_averages = np.zeros((PSRF_num, N+k))

                for j in range(PSRF_num):
                    
                    chain = utils.set_to_binary(chains[j],N+k)  
              
                    chains_binary[j] = chain
         
                    chain_averages[j] = (mix_step / i ) * np.average(chain, axis = 0)
                    within_variance[j] = (mix_step / i ) * np.var(chain, axis = 0)
                    
                PSRF = compute_PSRF(chain_averages, within_variance, i)
            
                PSRFs.append(PSRF)
       
                if PSRF < PSRF_tol and first:
                    print(PSRF)
                    mix_time = i
                    first = False


        return chains, np.array(PSRFs), mix_time  
 
