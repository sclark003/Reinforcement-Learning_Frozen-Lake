import numpy as np
from ModelBasedPolicy import *
from SmallFrozenLake import *
    
lake = np.array([['&',' ',' ',' ',' ',' ',' ',' '],
                    [' ',' ',' ',' ',' ',' ',' ',' '],
                    [' ',' ',' ','#',' ',' ',' ',' '],
                    [' ',' ',' ',' ',' ','#',' ',' '],
                    [' ',' ',' ','#',' ',' ',' ',' '],
                    [' ','#','#',' ',' ',' ','#',' '],
                    [' ','#',' ',' ','#',' ','#',' '],
                    [' ',' ',' ','#',' ',' ',' ','$']])
    
env = FrozenLake(lake,0.1,64)
#biggame.play()
#biggame.multiplePlay(5)

gamma = 0.9
theta = 0.001
max_iterations = 100
    

# Policy Iteration
policy,value,it = policy_iteration(env, gamma, theta, max_iterations, policy=None)
print("1) Policy Iteration: Number of Iterations:",it)
print(" ")
env.render(policy,value)
print(" ")
    
#Value Iteration
policy,value,it = value_iteration(env, gamma, theta, max_iterations, value=None)
print("2) Value Iteration: Number of Iterations:",it)
print(" ")
env.render(policy,value)
