#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 19:10:10 2023

@author: ben
"""

import random
from typing import List


class Environment:
    """
    A class representing an environment for an agent to 
    interact with.
    """
    def __init__(self):
        """
        Initializes the Environment class with a steps_left 
        attribute set to 10.
        """
        self.steps_left = 10

    def observation(self) -> List[float]:
        """
        Returns the current observation of the environment 
        as a list of floats.

        Returns:
        --------
        List[float]
            A list of floats representing the current 
            observation of the environment.
        """
        return [0.0, 0.0, 0.0]

    def actions(self) -> List[int]:
        """
        Returns the list of available actions that an agent 
        can perform in the environment as a list of integers.

        Returns:
        --------
        List[int]
            A list of integers representing the available 
            actions in the environment.
        """
        return [0, 1]

    def is_done(self) -> bool:
        """
        Returns a boolean indicating whether or not the 
        interaction between the agent and the environment 
        is finished.

        Returns:
        --------
        bool
            A boolean indicating whether or not the interaction 
            between the agent and the environment is finished.
        """
        return self.steps_left == 0

    def action(self, action: int) -> float:
        """
        Takes an integer representing the action chosen by 
        the agent and returns a float representing the reward 
        received by the agent. The number of steps left is 
        decreased by one after an action is taken. If there 
        are no more steps left, an Exception is raised.

        Parameters:
        -----------
        action: int
            An integer representing the action chosen by 
            the agent.

        Returns:
        --------
        float
            A float representing the reward received by 
            the agent.

        Raises:
        -------
        Exception
            If there are no more steps left.
        """
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        return random.random()


class Agent:
    """
    A class representing an agent that interacts with an 
    environment and keeps track of its total reward.

    Attributes:
    -----------
    total_reward: float
        The total reward accumulated by the agent during 
        its interaction with the environment.

    Methods:
    --------
    __init__():
        Initializes the Agent class with a total_reward 
        attribute set to 0.0.
    
    step(env: Environment):
        Takes an instance of an Environment class and 
        performs one step of interaction with it. 
        The agent gets the current observation from 
        the environment, available actions, 
        chooses an action at random, and receives a 
        reward. The reward is added to the total_reward 
        attribute.
    """
    def __init__(self):
        self.total_reward = 0.0

    def step(self, env: Environment):
        current_obs = env.observation()
        actions = env.actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward


if __name__ == "__main__":
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print("Total reward got: %.4f" % agent.total_reward) 
        
        
        