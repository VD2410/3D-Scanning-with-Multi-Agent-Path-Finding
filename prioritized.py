import time as timer
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost


class PrioritizedPlanningSolver(object):
    """A planner that plans for each robot sequentially."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.CPU_time = 0

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()
        result = []
        constraints = [
            # {'agent': 1,
            #  'loc': [(1, 4)],
            #  'timestep': 2},
            # {'agent': 1,
            #  'loc': [(1, 2)],
            #  'timestep': 2},
            # {'agent': 1,
            #  'loc': [(1, 3)],
            #  'timestep': 2}
        ]

        # Task 2.4: determine the maximum time step
        rows = len(self.my_map)
        cols = len(self.my_map[0])
        max_time_step = rows * cols

        for i in range(self.num_of_agents):  # Find path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, constraints, max_time_step)

            if path is None:
                raise BaseException('No solutions')
            result.append(path)

            # Task 2.4: update the maximum time step search space
            max_time_step += len(path)

            ##############################
            # Task 2: Add constraints here
            #         Useful variables:
            #            * path contains the solution path of the current (i'th) agent, e.g., [(1,1),(1,2),(1,3)]
            #            * self.num_of_agents has the number of total agents
            #            * constraints: array of constraints to consider for future A* searches


            ##############################

            # Task 2.1 build vertex constraint for other agents
            for time, loc in enumerate(path):
                for j in range(self.num_of_agents):
                    if j != i:
                        constraints.append({
                            'agent': j,
                            'loc': [(loc[0], loc[1])],
                            'timestep': time
                        })

            # Task 2.3: additional vertex constraint
            last_timestamp = len(path)
            last_loc = path[-1]
            for t in range(last_timestamp, last_timestamp + max_time_step):
                for j in range(self.num_of_agents):
                    if j != i:
                        constraints.append({
                            'agent': j,
                            'loc': [(last_loc[0], last_loc[1])],
                            'timestep': t,
                            'additional': True, # additional flag
                        })

            # Task 2.2 build the edge constraint
            for time in range(1, len(path)):
                cur_time = time
                pre_time = time - 1

                pre_loc = path[pre_time]
                cur_loc = path[cur_time]

                for j in range(self.num_of_agents):
                    if j != i:
                        constraints.append({
                            'agent': j,
                            'loc': [(pre_loc[0], pre_loc[1]), (cur_loc[0], cur_loc[1])],
                            'timestep': time
                        })
                        constraints.append({
                            'agent': j,
                            'loc': [(cur_loc[0], cur_loc[1]), (pre_loc[0], pre_loc[1])],
                            'timestep': time
                        })


        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        print(result)
        return result
