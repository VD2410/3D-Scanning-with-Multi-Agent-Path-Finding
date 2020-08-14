import heapq
import numpy as np

def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent, mymap):
    ##############################
    # Task 1.2/1.3: Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.

    # determine the max timestamp
    max_timestamp = -1
    for v_constraint in constraints:
        v_ts = v_constraint['timestep']
        v_agent = v_constraint['agent']
        if v_ts > max_timestamp and v_agent == agent:
            max_timestamp = v_ts

    if max_timestamp == -1:
        return None

    mymap = np.asarray(mymap)
    vertex_constraint_table = -1 * np.ones((max_timestamp+1, mymap.shape[0], mymap.shape[1]))
    edge_constraint_table = -1 * np.ones((max_timestamp+1, mymap.shape[0], mymap.shape[1], mymap.shape[0], mymap.shape[1]))

    for v_constraint in constraints:
        v_ts = v_constraint['timestep']
        v_agent = v_constraint['agent']
        v_positive_flag = False if 'positive' not in v_constraint else True

        if agent == v_agent and len(v_constraint['loc']) == 1:
            v_loc = v_constraint['loc'][0]
            vertex_constraint_table[v_ts, v_loc[0], v_loc[1]] = 0 if v_positive_flag is False else 1

        elif agent == v_agent and len(v_constraint['loc']) == 2:
            v_from_loc = v_constraint['loc'][0]
            v_to_loc = v_constraint['loc'][1]
            edge_constraint_table[v_ts, v_from_loc[0], v_from_loc[1], v_to_loc[0], v_to_loc[1]] = 0 if v_positive_flag is False else 1

    return {'vertex': vertex_constraint_table, 'edge': edge_constraint_table}


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # Task 1.2/1.3: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.

    # blocked = False

    # if the next time is not in constraint table, return un-blocked
    if constraint_table['vertex'].shape[0] <= next_time:
        return False
    else:
        # check vertex
        vertex_const_flag = constraint_table['vertex'][next_time, next_loc[0], next_loc[1]]
        edge_const_flag = constraint_table['edge'][next_time, curr_loc[0], curr_loc[1], next_loc[0], next_loc[1]]
        return (vertex_const_flag == 0) or (edge_const_flag == 0)

def is_positive_constrained(curr_loc, next_loc, next_time, constraint_table):

    # if the next time is not in constraint table, return un-blocked
    if constraint_table['vertex'].shape[0] <= next_time:
        return False
    else:
        # check vertex
        vertex_const_flag = constraint_table['vertex'][next_time, next_loc[0], next_loc[1]]
        edge_const_flag = constraint_table['edge'][next_time, curr_loc[0], curr_loc[1], next_loc[0], next_loc[1]]
        return (vertex_const_flag == 1) or (edge_const_flag == 1)

def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node['ts'], node))


def pop_node(open_list):
    _, _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints, max_time_steps=-1):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
        max_timestamp - maximum of timestamp for failure cases
    """

    ##############################
    # Task 1.1: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.

    rows = len(my_map)
    cols = len(my_map[0])

    # build the constraint table
    constraint_table = build_constraint_table(constraints, agent, my_map)

    open_list = []
    closed_list = dict()

    earliest_goal_timestep = 0
    for v_constraint in constraints:
        if len(v_constraint['loc']) == 1 and 'additional' not in v_constraint:
            v_time_step = v_constraint['timestep']
            if v_time_step > earliest_goal_timestep:
                earliest_goal_timestep = v_time_step

    h_value = h_values[start_loc]
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'ts': 0, 'parent': None}
    push_node(open_list, root)
    closed_list[(root['loc'])] = (root, 0)

    while len(open_list) > 0:
        curr = pop_node(open_list)

        # Task 2.4: Terminate if the maximum time step reached
        if max_time_steps > 0 and curr['ts'] >= max_time_steps:
            return None

        #############################
        # Task 1.4: Adjust the goal test condition to handle goal constraints
        if curr['loc'] == goal_loc:
            if (constraint_table is not None and curr['ts'] >= earliest_goal_timestep):
                return get_path(curr)
            elif constraint_table is None:
                return get_path(curr)

        ''' Add positive constraint if exists --------------------------------------------------------------------------
        '''
        positive_const_locs = []

        # check if the positive constraint exists in neighbors
        for dir in range(4):
            child_loc = move(curr['loc'], dir)

            is_neg_constrained = False
            is_pos_constrained = False

            if (child_loc[0] >= rows or child_loc[0] < 0) or (child_loc[1] >= cols or child_loc[1] < 0):
                continue

            if my_map[child_loc[0]][child_loc[1]]:
                continue    # ignore the location where it was blocked

            if constraint_table is not None:
                if is_constrained(curr['loc'], child_loc, curr['ts'] + 1, constraint_table):
                    is_neg_constrained = True

                if is_positive_constrained(curr['loc'], child_loc, curr['ts'] + 1, constraint_table):
                    is_pos_constrained = True

                if is_pos_constrained:
                    positive_const_locs.append(child_loc)

        if constraint_table is not None:

            is_neg_constrained = False
            is_pos_constrained = False

            if is_constrained(curr['loc'], curr['loc'], curr['ts'] + 1, constraint_table):
                is_neg_constrained = True

            if is_positive_constrained(curr['loc'], curr['loc'], curr['ts'] + 1, constraint_table):
                is_pos_constrained = True

            if is_pos_constrained:
                positive_const_locs.append(curr['loc'])

        # add positive constraints child
        if len(positive_const_locs) > 1:
            raise Exception('Can not have two positive constraint for next step')
        elif len(positive_const_locs) == 1:
            pos_loc = positive_const_locs[0]
            child = {'loc': pos_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[pos_loc],
                    'ts': curr['ts'] + 1,
                    'parent': curr}

            closed_tag = (child['loc'], curr['ts'] + 1)
            if closed_tag in closed_list:
                existing_node, ts = closed_list[closed_tag]
                # existing_node
                if compare_nodes(child, existing_node):
                    closed_list[closed_tag] = (child, child['ts'])
                    push_node(open_list, child)
            else:
                closed_list[closed_tag] = (child, child['ts'])
                push_node(open_list, child)

            continue    # continue as we have enforced the positive constraint

        ''' Add negative constraints -----------------------------------------------------------------------------------
        '''
        # add the 4-neighbors
        for dir in range(4):
            child_loc = move(curr['loc'], dir)

            if (child_loc[0] >= rows or child_loc[0] < 0) or (child_loc[1] >= cols or child_loc[1] < 0):
                continue

            if my_map[child_loc[0]][child_loc[1]]:
                continue    # ignore the location where it was blocked

            if constraint_table is not None:
                if is_constrained(curr['loc'], child_loc, curr['ts'] + 1, constraint_table):
                    continue

            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'ts': curr['ts'] + 1,
                    'parent': curr}

            closed_tag = (child['loc'], curr['ts'] + 1)
            if closed_tag in closed_list:
                existing_node, ts = closed_list[closed_tag]
                if compare_nodes(child, existing_node):
                    closed_list[closed_tag] = (child, child['ts'])
                    push_node(open_list, child)
            else:
                closed_list[closed_tag] = (child, child['ts'])
                push_node(open_list, child)

        # add current position with different time stamp
        child = {'loc': curr['loc'],
                 'g_val': curr['g_val'] + 1,
                 'h_val': h_values[curr['loc']],
                 'ts': curr['ts'] + 1,
                 'parent': curr}

        if constraint_table is not None and is_constrained(curr['loc'], curr['loc'], curr['ts'] + 1, constraint_table):
            continue

        closed_tag = (child['loc'], curr['ts'] + 1)
        if closed_tag in closed_list:
            existing_node, ts = closed_list[closed_tag]
            # existing_node
            if compare_nodes(child, existing_node):
                closed_list[closed_tag] = (child, child['ts'])
                push_node(open_list, child)
        else:
            closed_list[closed_tag] = (child, child['ts'])
            push_node(open_list, child)

    return None  # Failed to find solutions
