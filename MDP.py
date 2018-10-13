# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 09:18:14 2018

@author: SODIQ-PC
"""
#%%
import numpy as np
import random
from actions_space import action_space
from env_model import transition
#import sys

random.seed(30)

#%%

def choose_action(Q, state, num_actions, actions, eps, prob):
    '''the function allows agent to explore randomly and exploit 
    based on action that maximizes the state value'''
    
    #Chose action by e-greedy
    nongreedy = eps/num_actions
    greedy = 1 - eps + (eps/num_actions)
    prob = nongreedy * prob
    prob[np.argmax(Q[state])] = greedy
    
    return np.random.choice(list(actions.keys()), p = prob)

#%%

def mdp(level, states, Rewards):
    '''function helps implement the q_learning algorithm and construct the q_table'''
    #states = [0,1,2,3]
    Q = dict()
    num_episodes = 20
    min_alpha = 0.02
    alphas = np.linspace(1.0, min_alpha, num_episodes)
    actions = dict(enumerate(action_space[level]))
    num_actions = len(actions)
    #num_state = len(states)
    gamma = 0.8
    eps = 0.3
    prob = np.ones(num_actions)
    
    for ep in range(num_episodes):
        
        state = np.random.choice(states)
        total_reward = 0
        alpha = alphas[ep]
    
        i = 0
        while i < 1000:
            
            if state not in Q:
                Q[state] = np.zeros(num_actions)
            else:
                action = choose_action(Q, state, num_actions, actions, eps, prob)
                next_state = transition(state, action, states, level)
                reward = Rewards[state][action]
                if next_state not in Q:
                    Q[next_state] = np.zeros(num_actions)
                total_reward += reward
                Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
                state = next_state
            i += 1
            
        
        #print("Episode {}: total reward -> {}".format(ep+1, total_reward))
    
    #best_response = actions[np.argmax(Q[0])]
    #print(best_response)
    
    return Q, actions

#%%    
'''
if __name__ == '__main__':
    
    mdp(level = dialogue_level, Rewards) #level = int(sys.argv[1]))'''