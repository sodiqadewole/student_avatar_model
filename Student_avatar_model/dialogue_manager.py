# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 11:22:13 2018

@author: SODIQ-PC
"""
from MDP import mdp
import numpy as np
#import sys

dialogue_level = [1,2,3,4] 

def game_manager(score = input()):
    
    dialogue_level = 1
    state_space = [0, 1, 2, 3]
    terminal_dialogue = 4
    
    while dialogue_level <= terminal_dialogue:
        if score < 3:
            while score < 3:
                state = np.random.choice(state_space[:3])
                return response_manager(dialogue_level, state)
        dialogue_level += 1
       

def response_manager(dialogue_level, state):
    '''the state manager returns the optimal response for each dialogue 
    level and based on the state of the avatar agent
    
    Input = Q_table
    Output = action of the avatar in different states'''
    
    Q, actions = dialogue_manager(dialogue_level)
    
    if state == 0:
        response = actions[np.argmax(Q[state])]
    elif state == 1:
        response = actions[np.argmax(Q[state])]
    elif state == 2:
        response = actions[np.argmax(Q[state])]
    elif state == 3:
        response = actions[np.argmax(Q[state])]
    
    return response
    

def dialogue_manager(dialogue_level):
    
    '''In every dialogue level, agent learn new Q-function and develops a models
    the avatar's policy in different states of that level'''
    
    if dialogue_level == 1:
        
        state_space = [0,1,2,3]
        Rewards = {}
        Rewards[state_space[0]] = [1, 2, -3, 10, 0, -1]
        Rewards[state_space[1]] = [-2, 1, 0, 2, 0, -1]
        Rewards[state_space[2]] = [2, 1, 0, -1, 0, -1]
        Rewards[state_space[3]] = [0, 13, 4, -1, 0, -1]
        
        Q, actions = mdp(dialogue_level, state_space, Rewards)
    
    elif dialogue_level == 2:
        
        state_space = [0,1,2,3]
        Rewards = {}
        Rewards[state_space[0]] = [1, 2, -3, 10, 1, 2, 3, 0, -1]
        Rewards[state_space[1]] = [-2, 1, 0, 2, 1, 2, 3, 0, -1]
        Rewards[state_space[2]] = [2, 1, 0, -1, 1, 2, 3, 0, -1]
        Rewards[state_space[3]] = [0, 13, 4, -1, 1, 2, 3, 0, -1]
        
        Q, actions = mdp(dialogue_level, state_space, Rewards)
        
    elif dialogue_level == 3:
        
        state_space = [0,1,2,3]
        Rewards = {}
        Rewards[state_space[0]] = [1, 2, -3, 10, 1, 2, 3, 0, -1]
        Rewards[state_space[1]] = [-2, 1, 0, 2, 1, 2, 3, 0, -1]
        Rewards[state_space[2]] = [2, 1, 0, -1, 1, 2, 3, 0, -1]
        Rewards[state_space[3]] = [0, 13, 4, -1, 1, 2, 3, 0, -1]
        
        Q, actions = mdp(dialogue_level, state_space, Rewards)
        
    elif dialogue_level == 4:
        
        state_space = [0,1,2,3]
        Rewards = {}
        Rewards[state_space[0]] = [1, 2, -3, 10, 1, 2, 3, 0, -1]
        Rewards[state_space[1]] = [-2, 1, 0, 2, 1, 2, 3, 0, -1]
        Rewards[state_space[2]] = [2, 1, 0, -1, 1, 2, 3, 0, -1]
        Rewards[state_space[3]] = [0, 13, 4, -1, 1, 2, 3, 0, -1]
        
        Q, actions = mdp(dialogue_level, state_space, Rewards)

    return Q, actions

if __name__ == '__main__':
    
    game_manager()
    
    
        
    