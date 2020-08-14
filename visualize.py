#!/usr/bin/env python3
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import mlrose
import math
from utils import *

import datetime

Colors = ['green', 'blue', 'orange']


class Animation:
    def __init__(self, my_map, starts, goals, paths):



        # print(paths)
        self.my_map = np.flip(np.transpose(my_map), 1)
        self.starts = []
        for start in starts:
            self.starts.append((start[1], len(self.my_map[0]) - 1 - start[0]))
        self.goals = []
        self.cycle = 0
        for goal in goals:
            self.goals.append((goal[1], len(self.my_map[0]) - 1 - goal[0]))
        self.paths = []
        if paths:
            for path in paths:
                self.paths.append([])
                for loc in path:

                    # print("loc ",loc)
                    self.paths[-1].append((loc[0][1], len(self.my_map[0]) - 1 - loc[0][0]))

        aspect = len(self.my_map) / len(self.my_map[0])

        # mark the boundary
        self.bound_cells = mark_boundaries(self.my_map)
        self.min_set, dir, _ = find_min_set_cover(self.bound_cells)
        state_dist, state_space = gen_solution_space(self.bound_cells, self.min_set, dir, 0.5, 50)

        # solve
        # Initialize fitness function object using dist_list
        fitness_function = mlrose.TravellingSales(distances=state_dist)
        problem_fit = mlrose.TSPOpt(length=len(state_space), fitness_fn=fitness_function, maximize=False)

        # Solve the problem using the genetic algorithm
        best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state=5)

        print("The best state found is: ", best_state)
        print("The fitness at the best state is: ", best_fitness)


        self.fig = plt.figure(frameon=False, figsize=(2*aspect, 2))
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
        # self.ax.set_frame_on(False)

        self.patches = []
        self.artists = []
        self.agents = dict()
        self.Frustrum = dict()
        self.agent_names = dict()
        # create boundary patch

        # print("Bounded cells",self.bound_cells[7][3])

        # print("Map",self.my_map[7][3])

        x_min = -0.5
        y_min = -0.5
        x_max = len(self.my_map) - 0.5
        y_max = len(self.my_map[0]) - 0.5
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        self.patches.append(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, facecolor='none', edgecolor='gray'))
        for i in range(len(self.my_map)):
            for j in range(len(self.my_map[0])):
                
                if self.my_map[i][j]:
                    self.patches.append(Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor='gray', edgecolor='gray'))

        # draw solution space
        space_loc_keys = {}
        for solution in self.min_set:
            loc = solution
            loc_key = '%d,%d' % (loc[0], loc[1])
            if loc_key in space_loc_keys:
                continue
            else:
                i, j = loc[0], loc[1]
                self.patches.append(Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor='black', edgecolor='gray'))
                space_loc_keys[loc_key] = True

        # create agents:
        self.T = 0
        # cycle = 1
        # draw goals first
        # for i, goal in enumerate(self.goals):
        #     self.patches.append(Rectangle((goal[0] - 0.25, goal[1] - 0.25), 0.5, 0.5, facecolor=Colors[i % len(Colors)],
        #                                   edgecolor='black', alpha=0.5))


        



        for i in range(len(self.paths)):
            name = str(i)
            self.agents[i] = Circle((starts[i][0], starts[i][1]), 0.3, facecolor=Colors[i % len(Colors)],
                                    edgecolor='black')
            
            self.agents[i].original_face_color = Colors[i % len(Colors)]
            self.patches.append(self.agents[i])
            self.T = max(self.T, len(paths[i]) - 1)
            self.agent_names[i] = self.ax.text(starts[i][0], starts[i][1] + 0.25, name)
            self.agent_names[i].set_horizontalalignment('center')
            self.agent_names[i].set_verticalalignment('center')
            self.artists.append(self.agent_names[i])




        for i in range(0,int(self.T+1)*10*len(self.agents)):

            self.Frustrum[8*i] = Circle((-1, -1), 0.3, facecolor='red',
                                    edgecolor='black')

            self.patches.append(self.Frustrum[8*i])

            self.Frustrum[(8*i)+1] = Circle((-1, -1), 0.3, facecolor='red',
                                    edgecolor='black')

            self.patches.append(self.Frustrum[(8*i)+1])

            self.Frustrum[(8*i)+2] = Circle((-1, -1), 0.3, facecolor='red',
                                    edgecolor='black')

            self.patches.append(self.Frustrum[(8*i)+2])

            self.Frustrum[(8*i)+3] = Circle((-1, -1), 0.3, facecolor='red',
                                    edgecolor='black')

            self.patches.append(self.Frustrum[(8*i)+3])

            self.Frustrum[(8*i)+4] = Circle((-1, -1), 0.3, facecolor='red',
                                    edgecolor='black')

            self.patches.append(self.Frustrum[(8*i)+4])

            self.Frustrum[(8*i)+5] = Circle((-1, -1), 0.3, facecolor='red',
                                    edgecolor='black')

            self.patches.append(self.Frustrum[(8*i)+5])  

            self.Frustrum[(8*i)+6] = Circle((-1, -1), 0.3, facecolor='red',
                                    edgecolor='black')

            self.patches.append(self.Frustrum[(8*i)+6])  

            self.Frustrum[(8*i)+7] = Circle((-1, -1), 0.3, facecolor='red',
                                    edgecolor='black')

            self.patches.append(self.Frustrum[(8*i)+7])                        




        self.animation = animation.FuncAnimation(self.fig, self.animate_func,
                                                 init_func=self.init_func,
                                                 frames=int(self.T + 1) * 10,
                                                 interval=100,
                                                 blit=True)

    def save(self, file_name, speed):
        self.animation.save(
            file_name,
            fps=10 * speed,
            dpi=200,
            savefig_kwargs={"pad_inches": 0, "bbox_inches": "tight"})

    @staticmethod
    def show():
        plt.show()

    def init_func(self):
        for p in self.patches:
            self.ax.add_patch(p)
        for a in self.artists:
            self.ax.add_artist(a)
        return self.patches + self.artists

    def animate_func(self, t):

        # self.patches.append(Rectangle(xy=(-0.5, -0.5), width=1, height=1, angle=0, facecolor='red'))

        # print(t)
        flag = dict()

        start = datetime.datetime.now()

        # print(self.paths[0][len(self.paths[0])-1])


        for k in range(len(self.paths)):

            pos = self.get_state(t / 10, self.paths[k])
            if pos[0] == self.paths[k][len(self.paths[k])-1][0] and pos[1] == self.paths[k][len(self.paths[k])-1][1]:
                flag[k] = True
            else:
                flag[k] = False
            self.agents[k].center = (pos[0], pos[1])
            # self.Frustrum[k].center = (pos[0], pos[1])
            # print(pos[0])
            self.agent_names[k].set_position((pos[0], pos[1] + 0.5))

            i = 1;

            # print((self.bound_cells[3][3]), (self.my_map[4][4]))


            while(math.floor(pos[0]-i) >= 0):

                if self.bound_cells[math.floor(pos[0]-i)][math.floor(pos[1])] == -1:

                    j = 0

                    while(self.Frustrum[(8*k*t+j)].center[0] != -1):
                        j = j+1


                    self.Frustrum[(8*k*t+j)].center = (math.floor(pos[0]-i),math.floor(pos[1]))

                    # print(i, math.floor(pos[0]-i), self.bound_cells[math.floor(pos[0]-i)][math.floor(pos[1])])

                    i = 1000

                i = i + 1

            i = 1

            while(math.floor(pos[1]-i) >= 0):

                if self.bound_cells[math.floor(pos[0])][math.floor(pos[1]-i)] == -1:

                    j = 0

                    while(self.Frustrum[((8*k*t)+j+1)].center[0] != -1):

                        j = j+1




                    self.Frustrum[(8*k*t)+j+1].center = (math.floor(pos[0]),math.floor(pos[1]-i))

                    i = 1000

                i = i+1

            i = 1


            while math.floor(pos[0]+i) < len(self.bound_cells):
                
                if self.bound_cells[math.floor(pos[0]+i)][math.floor(pos[1])] == -1:


                    j = 0

                    while(self.Frustrum[((8*k*t)+j+2)].center[0] != -1):

                        j = j+1


                    self.Frustrum[(8*k*t)+j+2].center = (math.floor(pos[0]+i),math.floor(pos[1]))

                    i = 1000

                i = i + 1

            i = 1

            while math.floor(pos[1]+i) < len(self.bound_cells[0][:]):

                if self.bound_cells[math.floor(pos[0])][math.floor(pos[1]+i)] == -1:            

                    j = 0

                    while(self.Frustrum[((8*k*t)+j+3)].center[0] != -1):

                        j = j+1


                    self.Frustrum[(8*k*t)+j+3].center = (math.floor(pos[0]),math.floor(pos[1]+i))

                    i = 1000

                i = i + 1

            i = 1


            while math.floor(pos[1]+i) < len(self.bound_cells[0][:]) and math.floor(pos[0]+i) < len(self.bound_cells):

                # print("Pos0",math.floor(pos[0]+i), i, "Pos1", math.floor(pos[1]+i))

                if self.bound_cells[math.floor(pos[0]+i)][math.floor(pos[1]+i)] == -1:


                    j = 0

                    while(self.Frustrum[((8*k*t)+j+4)].center[0] != -1):

                        j = j+1


                    # print("diagonal",self.bound_cells[math.floor(pos[0]+i)][math.floor(pos[1]+i)] == -1,  math.floor(pos[0]+i), math.floor(pos[1]-i))

                    self.Frustrum[(8*k*t)+j+4].center = (math.floor(pos[0]+i),math.floor(pos[1]+i))

                    i = 1000

                elif math.floor(pos[0]+i+1)<len(self.bound_cells):

                    if self.bound_cells[math.floor(pos[0]+i+1)][math.floor(pos[1]+i)] == -1:


                        j = 0

                        while(self.Frustrum[((8*k*t)+j+4)].center[0] != -1):

                            j = j+1


                        self.Frustrum[(8*k*t)+j+4].center = (math.floor(pos[0]+i+1),math.floor(pos[1]+i))


                        i = 1000

                i = i + 1

            i = 1



            while math.floor(pos[1]-i) >= 0 and math.floor(pos[0]-i) >= 0:

                if self.bound_cells[math.floor(pos[0]-i)][math.floor(pos[1]-i)] == -1:



                    j = 0

                    while(self.Frustrum[((8*k*t)+j+5)].center[0] != -1):

                        j = j+1



                    self.Frustrum[(8*k*t)+j+5].center = (math.floor(pos[0]-i),math.floor(pos[1]-i))

                    i = 1000

                elif math.floor(pos[0]-i-1) >=0:

                    if self.bound_cells[math.floor(pos[0]-i-1)][math.floor(pos[1]-i)] == -1:


                        j = 0

                        while(self.Frustrum[((8*k*t)+j+5)].center[0] != -1):

                            j = j+1

                        self.Frustrum[(8*k*t)+j+5].center = (math.floor(pos[0]-i-1),math.floor(pos[1]-i))

                    
                        i = 1000
                
                i = i + 1

            i = 1

            while math.floor(pos[1]+i) < len(self.bound_cells[0][:]) and math.floor(pos[0]-i) >= 0:

                if self.bound_cells[math.floor(pos[0]-i)][math.floor(pos[1]+i)] == -1:            


                    j = 0

                    while(self.Frustrum[((8*k*t)+j+6)].center[0] != -1):

                        j = j+1

                    self.Frustrum[(8*k*t)+j+6].center = (math.floor(pos[0]-i),math.floor(pos[1]+i))

                    i = 1000

                elif math.floor(pos[0]-i-1) >=0:

                    if self.bound_cells[math.floor(pos[0]-i-1)][math.floor(pos[1]+i)] == -1:


                        j = 0

                        while(self.Frustrum[((8*k*t)+j+5)].center[0] != -1):

                            j = j+1

                        self.Frustrum[(8*k*t)+j+5].center = (math.floor(pos[0]-i-1),math.floor(pos[1]+i))

                        while(self.Frustrum[((8*k*t)+j+5)].center[0] != -1):

                            j = j+1

                        if math.floor(pos[1]+i+1) < len(self.bound_cells[0][:]):

                            if self.bound_cells[math.floor(pos[0]-i)][math.floor(pos[1]+i+1)] == -1:

                                self.Frustrum[(8*k*t)+j+5].center = (math.floor(pos[0]-i),math.floor(pos[1]+i+1))

                    
                        i = 1000

                i = i + 1

            i = 1


            while math.floor(pos[1]-i) >= 0 and math.floor(pos[0]+i) < len(self.bound_cells):


                if self.bound_cells[math.floor(pos[0]+i)][math.floor(pos[1]-i)] == -1:            


                    j = 0

                    while(self.Frustrum[((8*k*t)+j+7)].center[0] != -1):

                        j = j+1

                    self.Frustrum[(8*k*t)+j+7].center = (math.floor(pos[0]+i),math.floor(pos[1]-i))

                    i = 1000

                elif math.floor(pos[0]+i+1) < len(self.bound_cells):

                    if self.bound_cells[math.floor(pos[0]+i+1)][math.floor(pos[1]-i)] == -1:


                        j = 0

                        while(self.Frustrum[((8*k*t)+j+5)].center[0] != -1):

                            j = j+1

                        self.Frustrum[(8*k*t)+j+5].center = (math.floor(pos[0]+i+1),math.floor(pos[1]-i))

                        while(self.Frustrum[((8*k*t)+j+5)].center[0] != -1):

                            j = j+1

                        if math.floor(pos[1]-i-1) >=0:

                            if self.bound_cells[math.floor(pos[0]+i)][math.floor(pos[1]-i-1)] == -1:

                                self.Frustrum[(8*k*t)+j+5].center = (math.floor(pos[0]+i),math.floor(pos[1]-i-1))

                    
                        i = 1000

                i = i+1


            # print(self.Frustrum[10],self.Frustrum[11])                         

        # reset all colors
        for _, agent in self.agents.items():
            agent.set_facecolor(agent.original_face_color)

        # check drive-drive collisions
        agents_array = [agent for _, agent in self.agents.items()]
        for i in range(0, len(agents_array)):
            for j in range(i + 1, len(agents_array)):
                d1 = agents_array[i]
                d2 = agents_array[j]
                pos1 = np.array(d1.center)
                pos2 = np.array(d2.center)

                # print(self.bound_cells[int(pos1[0])-1][int(pos1[1])])

                # self.patches.append(Rectangle(xy=(-1, -1), width=1, height=1, angle=0, facecolor='red'))

                # if self.bound_cells[int(pos1[0])-1][int(pos1[1])] == -1:
                #     self.patches.append(Rectangle((int(pos1[0])-1 - 1,int(pos1[1]) - 1), 1, 1, facecolor='red', edgecolor='gray'))

                # if self.bound_cells[int(pos2[0])-1][int(pos2[1])] == -1:
                #     self.patches.append(Rectangle((i - 1, j - 1), 1, 1, facecolor='red', edgecolor='gray'))


                if np.linalg.norm(pos1 - pos2) < 0.7:
                    d1.set_facecolor('red')
                    d2.set_facecolor('red')
                    print("COLLISION! (agent-agent) ({}, {}) at time {}".format(i, j, t/10))

        # print("Time for one step is ",datetime.datetime.now() - start)

        complete = 0


        for x in range(0,len(flag)):
            if flag[x]:
                # print(x)
                complete = complete+1

        # print(flag, complete, len(self.agents))

        if self.cycle - complete == len(self.agents):
            print("The Scanning is complete")
            exit(0)

        self.cycle = complete

        return self.patches + self.artists

    @staticmethod
    def get_state(t, path):
        if int(t) <= 0:
            return np.array(path[0])
        elif int(t) >= len(path):
            return np.array(path[-1])
        else:
            pos_last = np.array(path[int(t) - 1])
            pos_next = np.array(path[int(t)])
            pos = (pos_next - pos_last) * (t - int(t)) + pos_last
            return pos
