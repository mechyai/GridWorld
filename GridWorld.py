import GridMap as gm


# ----------------------------------- GridWorld Environment Creation -----------------------------------
map = gm.GridMap(4, 3)
# set -1 reward for each step
map.set_const_reward(-1)
border_reward_distr = (0, 0)
# set non-interactive border wall with 0 reward
map.set_border('W', border_reward_distr)
# set all states path states, initialize all states to defualt
path_reward_distr = (0, 0)
map.set_state([], 'P', path_reward_distr)
print('Init Map')
# set wall at (2,2)
map.set_state([(2, 2)], 'W', (0, 0))
print('Set 1 Wall')
# set goal at (4,3)
map.set_state([(4, 3)], 'G', (10, 0))
print('Set 1 Goal')
# set ditch at (4,2)
map.set_state([(4, 2)], 'D', (-10, 0))
print('Set 1 Ditch')
# set start state at (1,1)
map.set_state([(1, 1)], 'S', (0, 0))
print('Set 1 Start')

# ----------------------------------- Q2(a) -----------------------------------
action_set = [(1, 0), (-1, 0), (0, 1), (0, -1)]
gamma = 1  # discount factor
# initialize values
map.initialize_value(0)
# get first state
init_state = map.initialize_state()
print(f'Start State Location {init_state.loc}')
threshold = 0.00001
delta = threshold + 1  # to enter while loop always
i = 0
while delta > threshold:
    delta = 0
    for path_state in map.path_states:
        v = map.get_value(path_state)
        temp_val = 0
        for action in action_set:
            # print(f'State Location {map_state.loc}')
            new_state = map.return_state(path_state, action)
            state_return = map.get_reward(new_state) + gamma * map.get_value(new_state)
            temp_val += 1 / len(action_set) * state_return
        map.set_value(path_state, temp_val)
        delta = max(delta, abs(v - temp_val))
    i += 1
    print(f'Policy Evaluation Iteration [{i}]')
    map.print_data('value')

# ----------------------------------- Q2(b) -----------------------------------
action_set = [(1, 0), (-1, 0), (0, 1), (0, -1)]
gamma = 1  # discount factor
# initialize values
map.initialize_value(0)
# get first state
init_state = map.initialize_state()
print(f'Start State Location {init_state.loc}')
threshold = 0.01
delta = threshold + 1  # to enter while loop always
i = 0
while delta > threshold:
    delta = 0
    for path_state in reversed(map.path_states):  # reverse order iteration
        v = map.get_value(path_state)
        temp_val = 0
        for action in action_set:
            # print(f'State Location {map_state.loc}')
            new_state = map.return_state(path_state, action)
            state_return = map.get_reward(new_state) + gamma * map.get_value(new_state)
            temp_val += 1 / len(action_set) * state_return
        map.set_value(path_state, temp_val)
        delta = max(delta, abs(v - temp_val))
    i += 1
    print(f'Policy Evaluation Iteration [{i}]')
    map.print_data('value')

