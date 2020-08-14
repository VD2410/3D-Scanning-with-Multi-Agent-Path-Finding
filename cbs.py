import time as timer
import heapq
import random, copy
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost

def loc_equal(loc1, loc2):
    """ Check if two location is equal
    """
    return loc1[0] == loc2[0] and loc1[1] == loc2[1]

def detect_collision(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.

    collisions = {}
    collisions['vertex'] = []
    collisions['edge'] = []

    # vertex collision
    max_time = max(len(path1), len(path2))
    for t in range(max_time):
        loc_p1 = get_location(path1, t)
        loc_p2 = get_location(path2, t)
        if loc_equal(loc_p1, loc_p2):
            collisions['vertex'].append((loc_p1, t))

    # edge collision
    for t in range(max_time - 1):
        pre_loc_p1 = get_location(path1, t)
        cur_loc_p1 = get_location(path1, t+1)

        pre_loc_p2 = get_location(path2, t)
        cur_loc_p2 = get_location(path2, t+1)

        # check the edge swap
        if loc_equal(pre_loc_p1, cur_loc_p2) and loc_equal(pre_loc_p2, cur_loc_p1):
            collisions['edge'].append((cur_loc_p1, cur_loc_p2, t + 1))

    return collisions

def detect_collisions(paths):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.

    collisions = []

    for i in range(len(paths)):
        for j in range(i, len(paths)):
            if i == j:
                continue
            else:
                collision = detect_collision(paths[i], paths[j])

                # add vertex collision
                for v_collision_event in collision['vertex']:
                    collisions.append({
                        'a1': i,
                        'a2': j,
                        'loc': [v_collision_event[0]],
                        'timestep': v_collision_event[1]
                    })

                # add edge collision
                for e_collision_event in collision['edge']:
                    collisions.append({
                        'a1': i,
                        'a2': j,
                        'loc': [e_collision_event[0], e_collision_event[1]],
                        'timestep': e_collision_event[2]
                    })

    return collisions

def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep

    if not isinstance(collision, list):
        collision = [collision]

    constraints = []
    for collision_event in collision[:1]:
        collision_agent_a = collision_event['a1']
        collision_agent_b = collision_event['a2']

        collision_loc = collision_event['loc']
        collision_ts = collision_event['timestep']

        # neg vertex constraints
        if len(collision_loc) == 1:
            constraints.append({
                'agent': collision_agent_a,
                'loc': [(collision_loc[0][0], collision_loc[0][1])],
                'timestep': collision_ts,
            })
            constraints.append({
                'agent': collision_agent_b,
                'loc': [(collision_loc[0][0], collision_loc[0][1])],
                'timestep': collision_ts,
            })

        # neg edge constraints
        if len(collision_loc) == 2:
            from_ = collision_loc[0]
            to_ = collision_loc[1]

            constraints.append({
                'agent': collision_agent_a,
                'loc': [(to_[0], to_[1]), (from_[0], from_[1])],
                'timestep': collision_ts,
            })
            constraints.append({
                'agent': collision_agent_b,
                'loc': [(from_[0], from_[1]), (to_[0], to_[1])],
                'timestep': collision_ts,
            })

    return constraints


def disjoint_splitting(collision: object) -> object:
    ##############################
    # Task 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep
    #           Choose the agent randomly

    if not isinstance(collision, list):
        collision = [collision]

    constraints = []
    for collision_event in collision[:1]:
        collision_agent_a = collision_event['a1']
        collision_agent_b = collision_event['a2']

        collision_loc = collision_event['loc']
        collision_ts = collision_event['timestep']
        random_choose_flag = random.randint(0, 1)

        if len(collision_loc) == 1: # vertex constraints

            constraints.append({
                'agent': collision_agent_a,
                'loc': [(collision_loc[0][0], collision_loc[0][1])],
                'timestep': collision_ts,
            })
            if random_choose_flag == 0: # mark the positive constraint if meets the random flag
                constraints[-1]['positive'] = True

            constraints.append({
                'agent': collision_agent_b,
                'loc': [(collision_loc[0][0], collision_loc[0][1])],
                'timestep': collision_ts,
            })
            if random_choose_flag == 1:
                constraints[-1]['positive'] = True

        if len(collision_loc) == 2: # edge constraints
            from_ = collision_loc[0]
            to_ = collision_loc[1]

            constraints.append({
                'agent': collision_agent_a,
                'loc': [(to_[0], to_[1]), (from_[0], from_[1])],
                'timestep': collision_ts,
            })
            if random_choose_flag == 0:
                constraints[-1]['positive'] = True

            constraints.append({
                'agent': collision_agent_b,
                'loc': [(from_[0], from_[1]), (to_[0], to_[1])],
                'timestep': collision_ts,
            })
            if random_choose_flag == 1:
                constraints[-1]['positive'] = True

    return constraints


def paths_violate_constraint(constraint, paths):
    assert constraint['positive'] is True
    rst = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == curr:
                rst.append(i)
        else:  # edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                rst.append(i)
    return rst


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def add_constraint(self, constraint, exist_constraint_set:list):
        """
        Add the constraint, if the constraint exsits, discard it.
        """

        def loc_equal(loc1, loc2):
            if len(loc1) == len(loc2):
                if len(loc1) == 1:
                    return loc1[0][0] == loc2[0][0] and loc1[0][1] == loc2[0][1]
                elif len(loc1) == 2:
                    return (loc1[0][0] == loc2[0][0] and loc1[0][1] == loc2[0][1]) and \
                           (loc1[1][0] == loc2[1][0] and loc1[1][1] == loc2[1][1])
            else:
                return False

        exist_flag = False

        # find if exist:
        for exist_constraint in exist_constraint_set:
            agent_id = exist_constraint['agent']
            loc = exist_constraint['loc']
            ts = exist_constraint['timestep']
            p_flag = True if 'positive' in exist_constraint else False
            cur_flag = True if 'positive' in constraint else False

            if agent_id == constraint['agent'] and loc_equal(loc, constraint['loc']) and \
                     ts == constraint['timestep'] and p_flag == cur_flag:
                exist_flag = True
                break

        if exist_flag == False:
            exist_constraint_set.append(constraint)

    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        # Task 3.1: Testing
        print(root['collisions'])

        # Task 3.2: Testing
        for collision in root['collisions']:
            print(standard_splitting(collision))

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        while len(self.open_list) > 0:
            P = self.pop_node()
            if len(P['collisions']) < 1:
                self.print_results(P)
                return P['paths']

            # select first collision
            collision = P['collisions'][0]
            constraints = standard_splitting(collision) if disjoint is False else disjoint_splitting(collision)

            for c_idx, constraint in enumerate(constraints):

                Q = {'cost': 0,
                     'constraints': [],
                     'paths': [],
                     'collisions': []}
                Q['constraints'] = copy.copy(P['constraints'])
                self.add_constraint(constraint, Q['constraints'])
                Q['paths'] = copy.copy(P['paths'])

                agent = constraint['agent']
                new_path = a_star(self.my_map, self.starts[agent], self.goals[agent], self.heuristics[agent], agent, Q['constraints'])

                # find the violated agent given the new path with positive constraints (valid when disjoint == True)
                if disjoint and 'positive' in constraint:
                    violate_agents = paths_violate_constraint(constraint, Q['paths'])
                    for c_neg_idx, neg_constraint in enumerate(constraints):
                        if 'positive' not in neg_constraint:
                            self.add_constraint(neg_constraint, Q['constraints'])
                else:
                    violate_agents = []

                # update other agent path given new negative constraints (valid when disjoint == True)
                violate_paths = {}
                num_new_paths = 0
                for violate_agent in violate_agents:
                    new_vio_path = a_star(self.my_map, self.starts[violate_agent], self.goals[violate_agent], self.heuristics[violate_agent], violate_agent, Q['constraints'])
                    violate_paths[violate_agent] = new_vio_path

                    if new_vio_path is not None and len(new_vio_path) > 0:
                        num_new_paths += 1

                # add new child node if new path of current agent and new paths from other agents exsit.
                if len(new_path) > 0 and num_new_paths == len(violate_agents):

                    # update pathes
                    Q['paths'][agent] = new_path
                    for violate_agent in violate_agents:
                        Q['paths'][violate_agent] = violate_paths[violate_agent]

                    # update collison and costs
                    new_collison = detect_collisions(Q['paths'])
                    Q['collisions'] = new_collison
                    Q['cost'] = get_sum_of_cost(Q['paths'])

                    self.push_node(Q)

        self.print_results(root)
        return root['paths']


    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
