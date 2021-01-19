Frozen Lake Environment and Policy algorithms in Python, created for AI in Games module.

Frozen lake environment based on the Frozen lake from the Gym Library: https://gym.openai.com/envs/FrozenLake-v0/

There are two different sizes of the Frozen lake:

![Small Frozen Lake](https://user-images.githubusercontent.com/60785645/105031167-d0add680-5a4c-11eb-9ba0-fc73968f1259.png)

![Big Frozen Lake](https://user-images.githubusercontent.com/60785645/105031109-b96ee900-5a4c-11eb-89ee-410c22a8fdc5.png)

Each square in the grid refers to a state. There is also an additional absorbing state.
Four types of tile:
- Grey = Start tile
- Frozen lake = Light blue
- Hole = Dark blue, agent goes to absorbing state
- Goal = White, agent goes to absorbing state

If an action taken would cause the agent to leave the grid, the agent will just stay in the same sqaure.
Each action taken at the absorbing state, leads to the absorbing state.
There is a probability of 0.1 that the agent 'slips' on the frozen lake and ignores the chosen action.

The agent recieves a reward 1 upon taking an action at the goal. In every other case, the reard is zero.
