import numpy as np
import random
from itertools import product
import contextlib
from EnvironmentModel import *
from ModelBasedPolicy import *
from TabularModelFreePolicy import *
from TabularModelFreePolicy import *
from NonTabularModelFreeLearning import *


# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        Environment.__init__(self, lake.size + 1, 4, max_steps, None, seed)
         
        self.lake = np.array(lake)
        self.lake_flat = lake.reshape(-1)
        
        self.slip = slip  # probability of slipping
        
        n_states = self.lake.size + 1
        n_actions = 4
        
        self.pi = np.zeros(n_states, dtype=float)
        self.pi[np.where(self.lake_flat == '&')[0]] = 1.0  # intialise starting square by pi to 0 at this square (zero at all others)
        
        self.absorbing_state = n_states - 1
    
        # Up, down, left, right
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
        # Indices to states (coordinates), states (coordinates) to indices 
        itos = list(product(range(self.lake.shape[0]), range(self.lake.shape[1])))
        stoi = {s: i for (i, s) in enumerate(itos)}
            
        # Precomputed transition probabilities
        self.p_mat = np.zeros((n_states, n_states, n_actions))

        hole = []
        goal = 0
        for i in range(len(self.lake_flat)):
            if self.lake_flat[i] == '#':   # find where there are holes in the lake
                hole.append(i)
            if self.lake_flat[i] == '$':   # find where the goal is in the lake
                goal = i
        
        for state_index, state in enumerate(itos):
            for action_index, action in enumerate(actions):
                if stoi.get(state) in hole or stoi.get(state) == goal: # if next state s absorbing state (i.e if state is hole or goal)
                    next_state_index = self.absorbing_state
                else:
                    next_state = (state[0] + action[0], state[1] + action[1])
                    # If next_state is not valid, default to current state index
                    next_state_index = stoi.get(next_state, state_index)
                
                # include slipping probability in transition matrix when next state = current state
                if stoi.get(next_state) == stoi.get(state):
                    self.p_mat[next_state_index, state_index, action_index] = (self.slip/n_actions)

                # if next state is absorbing state, probability is 1
                if stoi.get(state) in hole or stoi.get(state) == goal:
                    self.p_mat[next_state_index, state_index, action_index] = 1

                # include slipping probability in transition matrix
                else:
                    self.p_mat[next_state_index, state_index, action_index] += (1-self.slip)  # save probabilities of tansitioning between states
                    for i in range(n_actions):
                        self.p_mat[next_state_index, state_index, i] += (self.slip/n_actions) # include probability of sliping and taking a random action
 
        # every action at absorbing state always goes to absorbing state       
        self.p_mat[self.absorbing_state][self.absorbing_state] = (1)

        # Transition probabilities for non-slipping environment (for comparison)
        self.p_mat_nonslip = np.zeros((n_states, n_states, len(actions)))

        for state_index, state in enumerate(itos):
            for action_index, action in enumerate(actions):
                if stoi.get(state) in hole or stoi.get(state) == goal: # if next state s absorbing state (i.ei if state is hole or goal)
                    next_state_index = self.absorbing_state
                else:
                    next_state = (state[0] + action[0], state[1] + action[1])
                    # If next_state is not valid, default to current state index
                    next_state_index = stoi.get(next_state, state_index)
                self.p_mat_nonslip[next_state_index, state_index, action_index] = 1.0  # save probabilities of tansitioning between states
        self.p_mat_nonslip[self.absorbing_state][self.absorbing_state] = 1   # every action at absorbing state always goes to absorbing state


    def draw(self, state, action):
        p = [self.p(ns, state, action, self.p_mat) for ns in range(self.n_states)]
        # find next state according to probabilities
        next_state = self.random_state.choice(self.n_states, p=p)
        # find reward of transitioning to next state
        reward = self.r(next_state, state, action)
        
        return next_state, reward

    def nonslip_draw(self, state, action):        
        # find what the non-slipping next state would be
        p = [self.p(ns, state, action, self.p_mat_nonslip) for ns in range(self.n_states)]
        nonslip_next_state = self.random_state.choice(self.n_states, p=p)

        return nonslip_next_state

    def step(self, action):
        # end game if run out of moves
        state, reward, done = Environment.step(self, action)
        # end game if reached absorbing state
        done = (state == self.absorbing_state) or done
        
        return state, reward, done

    def p(self, next_state, state, action, mat = None):
        if mat is None:
            mat = self.p_mat
        # find probability from pre-caluated transition probabilities matrix
        probability = mat[next_state, state, action]
        return probability
        
    def r(self, next_state, state, action):
        # only recieve a reward when transitioning from goal state to absorbing state
        if state == self.absorbing_state - 1 and next_state == self.absorbing_state:
            reward = 1
        else:
            reward = 0
   
        return reward
        
    def render(self, policy=None, value=None):

        if policy is None:
            lake = np.array(self.lake_flat)    
            if self.state < self.absorbing_state:
                lake[self.state] = '@'    
            print(lake.reshape(self.lake.shape))

        else:
            actions = ['↑', '↓', '←', '→']
            print('Lake:')
            print(self.lake)
                        
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))

        print("")

    def selectAction(self):
        keys = ["w","s","a","d"]   # up, down, left, right
        a = False
            
        while not a:
            key = input("Input action:")
            a= self.validMove(key, keys)                
        action = keys.index(key)
            
        return action
    
    def validMove(self, key, keys): 
        # if input action is invalid, return false
        if key not in keys:
            print("Invalid action! Try again")
            return False
        else:
            return True
        
    def randomAction(self): 
        return random.choice([0,1,2,3])
    
    def play(self,):
        state = self.reset()
        self.render()        
        done = False
        move_no = 0
        actions = ['↑', '↓', '←', '→']
        
        while not done:
            move = self.max_steps - move_no
            print("Number of moves remaining:",move)
            # select action to make move
            action = self.selectAction()
            print("Move:", actions[action],"\n")
            nonslip_next_state = self.nonslip_draw(state, action)
            state, r, done = self.step(action)
            
            # find out if you've slipped        
            if state != nonslip_next_state:
                print("Whoops, you slipped!")
            
            self.render()           
            
            if move_no == self.max_steps-1:
                print("Run out of Moves")

            if done == True:
                print("Score:",r)
                return r
            
            move_no +=1
        
    # function to play game multple times
    def multiplePlay(self, iterations):
        score = 0
        for i in range(iterations):
            print("\n\n___________________________________")
            print("Game", i+1)
            print("Current Score:",score)
            r = self.play()
            score += r
        print("Final Score:", score)
    



def main():
    
    seed = 0
    
    # Small lake
    lake = np.array([['&',' ',' ',' ',],
            [' ','#',' ','#'],
            [' ',' ',' ','#'],
            ['#',' ',' ','$']])
    
    env = FrozenLake(lake,slip=0.1,max_steps=16,seed = seed)

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
    print("___________________________________")
    

if __name__ == "__main__":
    main()
