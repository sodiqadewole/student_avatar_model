# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 20:27:59 2018

@author: SODIQ-PC
"""

import numpy as np
from actions_space import action_space

def transition(state, action, states, level):
    '''The function contains the model of the environment
    it takes the current state and action and return the most likely next state'''
    Actions = action_space[level]
    
    num_states = len(states)
    num_actions = len(Actions)
    
    transition_mat = np.zeros([num_states, num_actions, num_states])
    transition_mat[0,0,] = np.round(np.random.dirichlet(np.ones(4), size=1), decimals=2).tolist()[0]
    transition_mat[0,1,] = np.round(np.random.dirichlet(np.ones(4), size=1), decimals=2).tolist()[0]
    transition_mat[0,2,] = np.round(np.random.dirichlet(np.ones(4), size=1), decimals=2).tolist()[0]
    transition_mat[0,3,] = np.round(np.random.dirichlet(np.ones(4), size=1), decimals=2).tolist()[0]
    transition_mat[1,0,] = np.round(np.random.dirichlet(np.ones(4), size=1), decimals=2).tolist()[0]
    transition_mat[1,1,] = np.round(np.random.dirichlet(np.ones(4), size=1), decimals=2).tolist()[0]
    transition_mat[1,2,] = np.round(np.random.dirichlet(np.ones(4), size=1), decimals=2).tolist()[0]
    transition_mat[1,3,] = np.round(np.random.dirichlet(np.ones(4), size=1), decimals=2).tolist()[0]
    transition_mat[2,0,] = np.round(np.random.dirichlet(np.ones(4), size=1), decimals=2).tolist()[0]
    transition_mat[2,1,] = np.round(np.random.dirichlet(np.ones(4), size=1), decimals=2).tolist()[0]
    transition_mat[2,2,] = np.round(np.random.dirichlet(np.ones(4), size=1), decimals=2).tolist()[0]
    transition_mat[2,3,] = np.round(np.random.dirichlet(np.ones(4), size=1), decimals=2).tolist()[0]
    transition_mat[3,0,] = np.round(np.random.dirichlet(np.ones(4), size=1), decimals=2).tolist()[0]
    transition_mat[3,1,] = np.round(np.random.dirichlet(np.ones(4), size=1), decimals=2).tolist()[0]
    transition_mat[3,2,] = np.round(np.random.dirichlet(np.ones(4), size=1), decimals=2).tolist()[0]
    transition_mat[3,3,] = np.round(np.random.dirichlet(np.ones(4), size=1), decimals=2).tolist()[0]
    
    next_state = np.argmax(transition_mat[state,action,])
    #reward = Rewards[state][action]
    
    return next_state
    