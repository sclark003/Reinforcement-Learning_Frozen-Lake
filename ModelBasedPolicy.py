import numpy as np        

# function to evaluate current policy
def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)                # loop value array
    
    for i in range(int(max_iterations)):
        delta = 0
        for state in range(env.n_states):                         # iterate through states
            v = 0  
            states = np.linspace(0,env.n_states-1,env.n_states)
            for action in range(env.n_actions):                   # loop through possible actions
                
                # policy_s_a = probability of taking action from state according to current policy
                # Here, 1 for action in policy[state], 0 for other actions
                if action == policy[state]:
                    policy_s_a = 1
                else:
                    policy_s_a = 0

                p = np.array([env.p(ns, state, action) for ns in range(env.n_states)])       # find transition probabilities
                trans_probability = p[p>0]                                                   # select probabilities over zero
                next_states = np.array(states[np.where(p>0)], dtype = int)                   # find possible next states according to transition probabilities
                
                # calculate probability*(reward + gamma*value[next_state]) for each possible
                # next state, and sum together
                sum_s = 0
                for i in range(len(next_states)):                                   # for each possible next state
                    ns = next_states[i]                                             # find possible next state
                    reward = env.r(ns,state, action)                                # reward of possible next_state
                    sum_s += trans_probability[i]*(reward + (gamma*value[ns]))      # find sum for each possible next state from current state and action

                v += policy_s_a * sum_s                     # value for state = sum * probability of taking action from state according to current policy
             
            delta = max(delta, np.abs(v- value[state]))     # calculate change
            value[state] = v                                # update value array
                    
        if delta < theta:                # if change is less than set threshold, end iterations
            return value
                             
    return value                         # if max iterations have been done, end iterations



# function to improve current policy by taking value of each state and finding policy
# with value better than or equal to old value at each state
def policy_improvement(env, value, gamma, policy=None):        
    if policy is None:
        policy = np.zeros(env.n_states, dtype = int)

    policy_stable = True
    for state in range(env.n_states):                                # Improve the policy at each state, except absorbing
        chosen_a = policy[state]                                     # Choose action according to policy                                
        best_a = np.argmax(actionValues(env, state, gamma, value))   # Find the best action by one-step-look-ahead   
       
        if chosen_a != best_a:                                       # If policy action is not best, replace with best action
            policy_stable = False    
        policy[state] = (best_a)                                     # Update policy

    return policy, policy_stable                                     # Return improved policy                                          




# function to iteratively evaluate policy and improve it
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype = int)
                                                                                   # Number of loops
    policy_stable = True
    for i in range(int(max_iterations)):                                           # Repeat until convergence or max_iterations   
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)       # Evaluate current policy
        policy,policy_stable = policy_improvement(env, value, gamma,policy)        # improve current policy
        
        if policy_stable:
            return policy, value,i+1           

    return policy, value, i+1




   
# function to iteratively find best actions according to value, and produce policy from this
def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
                                                                             # Number of loops
    for i in range(max_iterations):                                              # Repeat until convergence or max_iteration
        delta = 0
        for state in range(env.n_states):                                        # Loop through states                     
            best_action_value = np.max(actionValues(env, state, gamma, value))   # Find the best action by one-step-look-ahead
            delta = max(delta, np.abs(best_action_value - value[state]))         # Calculate change in value
            value[state] = best_action_value                                     # Update value for current state         
        if delta < theta:                                                        # if change is less than set threshold, end iterations
            break  
        
    policy  = np.zeros(env.n_states, dtype = int)
    for state in range(env.n_states):                                       # Create policy using optimal value function                        
        best_a = np.argmax(actionValues(env, state, gamma, value))          # Find the best action by one-step-look-ahead
        policy[state] = best_a                                              # Always take the best action
        
    return policy, value,i+1





  
def actionValues(env, state, gamma, value):
    action_values = np.zeros(env.n_actions)
    states = np.linspace(0,env.n_states-1,env.n_states)
    for action in range(env.n_actions):                                              # loop through possible actions
        p = np.array([env.p(ns, state, action) for ns in range(env.n_states)])       # find transition probabilities
        probability = p[p>0]                                                         # select probabilities over zero
        next_states = np.array(states[np.where(p>0)], dtype = int)                   # find possible next states according to transition probabilities

        for i in range(len(next_states)):
            reward = env.r(next_states[i],state, action)
            action_values[action]+= probability[i]*(reward + (gamma*value[next_states[i]]))  # find values for each possible next state from current state and action
        
    noise = np.random.rand(1,env.n_actions)/10000                                  
    action_values = (action_values+noise)                                            # add noise to help break ties randomly
    return action_values                                                             # return action values for possible actions
