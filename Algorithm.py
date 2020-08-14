import numpy as np
from utils import *
import mlrose
from cbs import CBSSolver
import datetime
class Algorithm:

    def __init__(self, my_map):

        self.my_map = my_map
        self.my_map_np = np.asarray(self.my_map)

        # analysis the map
        self.bound_cells = mark_boundaries(self.my_map_np)
        print('Find Minimal Set Cover')

        start = datetime.datetime.now()

        self.key_points, self.key_pt_dirs, self.key_pt_cover_bd_idx = find_min_set_cover(self.bound_cells)
        self.state_dist, self.state_space = gen_solution_space(self.bound_cells, self.key_points, self.key_pt_dirs, 0.1, 50)

        print("Time for Minimal Set Cover is ",datetime.datetime.now() - start)

        # run the TSP to initialize
        print('Run TSP on finding vertices cover')

        start = datetime.datetime.now()


        fitness_function = mlrose.TravellingSales(distances=self.state_dist)
        problem_fit = mlrose.TSPOpt(length=len(self.state_space), fitness_fn=fitness_function, maximize=False)

        # Solve the problem using the genetic algorithm
        best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state=5)
        self.best_state = best_state

        # extract states
        idx_2_state_spaces = {}
        for key, value in self.state_space.items():
            idx_2_state_spaces[value] = key
        state_dict = [idx_2_state_spaces[i].split("-") for i in self.best_state]
        self.state_dict = [(self.key_points[int(s[0])], s[1]) for s in state_dict]

        if np.isinf(best_fitness):
            print('Error when finding TSP')

        print("Time for TSP is ",datetime.datetime.now() - start)

    def run(self, start_pts):
        agent_paths = []        # N
        agent_state_idx = []    # N - 1
        n_agents = len(start_pts)

        # find the initial goal
        init_picked_flag = dict()
        occupied_flag = dict()
        for agent_start_loc in start_pts:

            dists = [manhattan_distance(agent_start_loc, state[0]) for state in self.state_dict]
            dists = np.asarray(dists)

            idx = np.argsort(dists)
            j = 0
            picked_id = idx[j]
            loc = self.state_dict[picked_id][0]
            loc_key = '%d-%d' % (loc[0], loc[1])

            while loc_key in init_picked_flag:
                j += 1
                picked_id = idx[j]
                loc = self.state_dict[picked_id][0]
                loc_key = '%d-%d' % (loc[0], loc[1])

                if picked_id > idx.shape[0]:
                    raise Exception('Can not found any start states.')

            init_picked_flag[loc_key] = True
            path = [(agent_start_loc, self.state_dict[picked_id][1])]
            path.append(self.state_dict[picked_id])

            agent_paths.append(path)
            agent_state_idx.append([picked_id])
            occupied_flag[picked_id] = True

        # continue search the remaining vertices
        n_picked_states = len(occupied_flag)

        def exists_in_path(paths, loc):
            loc_key = '%d-%d' % (loc[0], loc[1])
            for path in paths:
                path_last_loc = path[-1][0]
                path_loc_key = '%d-%d' % (path_last_loc[0], path_last_loc[1])
                if loc_key == path_loc_key:
                    return True
            return False

        iter = 0
        while n_picked_states <= len(self.state_dict):

            agent_id = iter % n_agents
            other_agent_ids = []
            for i in range(n_agents):
                if i != agent_id:
                    other_agent_ids.append(i)

            # agent_path = agent_paths[agent_id]

            cur_state_idx = agent_state_idx[agent_id][-1]
            next_state_idx = cur_state_idx + 1
            if next_state_idx >= len(self.state_dict):
                next_state_idx = 0
            next_loc = self.state_dict[next_state_idx][0]

            try_times = 0
            found_flag = True
            while next_state_idx in occupied_flag:
                next_state_idx += 1
                if next_state_idx >= len(self.state_dict):
                    next_state_idx = 0
                next_loc = self.state_dict[next_state_idx][0]

                try_times += 1
                if try_times >= len(self.state_dict):
                    found_flag = False
                    break

            # check if the loc is exists
            other_agent_paths = [agent_paths[i] for i in other_agent_ids]
            while exists_in_path(other_agent_paths, next_loc):
                next_state_idx += 1
                if next_state_idx >= len(self.state_dict):
                    next_state_idx = 0
                next_loc = self.state_dict[next_state_idx][0]

                try_times += 1
                if try_times >= len(self.best_state):
                    found_flag = False
                    break

            if found_flag is False:
                break
            else:
                occupied_flag[next_state_idx] = True
                agent_state_idx[agent_id].append(next_state_idx)
                agent_paths[agent_id].append(self.state_dict[next_state_idx])
                n_picked_states += 1

            iter += 1

        if n_picked_states < len(self.state_dict):
            raise Exception('Not all states are collected')

        # Run CBS
        max_path_length = 0
        for path in agent_paths:
            path_length = len(path)
            if path_length > max_path_length:
                max_path_length = path_length

        output_agent_pathes = []
        output_agent_idx = dict()
        start_time = datetime.datetime.now()
        for path_id in range(0, max_path_length-1):

            agend_ids = []
            start_pos = []
            start_dir = []
            goal_pos = []
            goal_dir = []
            for agent_id in range(len(start_pts)):
                agent_path = agent_paths[agent_id]

                start = agent_path[path_id] if path_id < len(agent_path) else None
                end = agent_path[path_id+1] if path_id + 1 < len(agent_path) else None

                if start is not None and end is not None:
                    start_pos.append(start[0])
                    start_dir.append(start[1])
                    goal_pos.append(end[0])
                    goal_dir.append(end[1])
                    agend_ids.append(agent_id)

            # run CBS
            

            cbs = CBSSolver(self.my_map, start_pos, goal_pos)
            solutions = cbs.find_solution(disjoint=True)

            

            max_sol_length = 0
            for solution in solutions:
                if len(solution) > max_sol_length:
                    max_sol_length = len(solution)

            current_last_item = []
            for agent_id in range(len(start_pts)):
                if agent_id not in output_agent_idx:
                    output_agent_idx[agent_id] = True
                    output_agent_pathes.append([])

                if agent_id in agend_ids:

                    # locate the solution given agent id
                    sol_idx = 0
                    for s_idx in range(0, len(agend_ids)):
                        if agent_id == agend_ids[s_idx]:
                            sol_idx = s_idx
                            break

                    current_last_item.append(solutions[sol_idx][-1])
                else:
                    current_last_item.append(output_agent_pathes[agent_id][-1])

            for agent_id in range(len(start_pts)):
                if agent_id in agend_ids:
                    # locate the solution given agent id
                    sol_idx = 0
                    for s_idx in range(0, len(agend_ids)):
                        if agent_id == agend_ids[s_idx]:
                            sol_idx = s_idx
                            break
                    sol = solutions[sol_idx]
                    s_dir = start_dir[sol_idx]
                    g_dir = goal_dir[sol_idx]

                    for sol_item in range(0, max_sol_length):
                        if sol_item < len(sol) - 1:
                            output_agent_pathes[agent_id].append((sol[sol_item], s_dir))
                        elif sol_item == len(sol) - 1:
                            output_agent_pathes[agent_id].append((sol[sol_item], g_dir))
                        else:
                            output_agent_pathes[agent_id].append((sol[-1], g_dir))
                else:
                    for sol_item in range(0, max_sol_length):
                        last_item = output_agent_pathes[agent_id][-1]
                        output_agent_pathes[agent_id].append(last_item)

        print("Time for CBS Solver is ", datetime.datetime.now()-start_time)

        return output_agent_pathes