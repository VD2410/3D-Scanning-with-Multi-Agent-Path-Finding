import numpy as np
import SetCoverPy.setcover as setcover

def mark_boundaries(map:np.ndarray):
    ''' mark the boundary
    '''
    H, W = map.shape
    bound_cells = np.zeros_like(map, dtype=np.int)

    for h in range(0, H):
        for w in range(0, W):
            bound_cells[h, w] = 1 if map[h, w] == True else 0

            if map[h, w] == True:

                up_cell = True if h > 1 and map[h-1, w] == False else False
                btm_cell = True if h < H-1 and map[h+1, w] == False else False
                rht_cell = True if w < W-1 and map[h, w+1] == False else False
                left_cell = True if w > 1 and map[h, w-1] == False else False

                if (up_cell or btm_cell or rht_cell or left_cell):
                    bound_cells[h, w] = -1

    return bound_cells

def distance_search(kernel, direction):
    dist_levels = int(kernel / 2)
    distances = []
    for i in range(1, dist_levels+1):
        tmp = np.arange(-1*i, i*1+1).astype(np.int)

        loc_offset = []
        if direction == 'top':
            for j in range(tmp.shape[0]):
                loc_offset.append((-i, tmp[j]))
        elif direction == 'btm':
            for j in range(tmp.shape[0]):
                loc_offset.append((i, tmp[j]))
        elif direction == 'left':
            for j in range(tmp.shape[0]):
                loc_offset.append((tmp[j], -i))
        elif direction == 'right':
            for j in range(tmp.shape[0]):
                loc_offset.append((tmp[j], i))
        distances.append(loc_offset)

    return distances

def find_min_set_cover(boundary_map, kernel=5):

    H, W = boundary_map.shape
    radius = int(kernel / 2)

    # count the boundary cells
    n_boundaries = np.sum(boundary_map == -1)
    boundary2idx = dict()
    idx2bounary = dict()
    boundary_cell_idx = 0
    for h in range(0, H):
        for w in range(0, W):
            if boundary_map[h, w] == -1:
                key = '%d-%d' % (h, w)
                boundary2idx[key] = boundary_cell_idx
                idx2bounary[boundary_cell_idx] = (h, w)
                boundary_cell_idx += 1

    search_cell_offset = dict()
    search_cell_offset['left'] = distance_search(kernel, 'left')
    search_cell_offset['right'] = distance_search(kernel, 'right')
    search_cell_offset['top'] = distance_search(kernel, 'top')
    search_cell_offset['btm'] = distance_search(kernel, 'btm')

    candidates_covers = []
    ignored_b_cells = dict()
    max_bd_cells = 0
    for h in range(0, H):
        for w in range(0, W):
            if boundary_map[h, w] != 0:
                continue

            if h == 2 and w == 3:
                debug = True
                print('A')
            else:
                debug = False

            cover = None
            near_dist = False
            long_dist = False

            for search_direction in ['left', 'right', 'top', 'btm']:
                search_levels = search_cell_offset[search_direction]
                found_sub_sum = 0
                for l, search_cell_offset_l in enumerate(search_levels):

                    for offset in search_cell_offset_l:
                        cell_h = int(h + offset[0])
                        cell_w = int(w + offset[1])

                        if 0 <= cell_h < H and 0 <= cell_w < W and boundary_map[cell_h, cell_w] == -1:

                            if cover is None:
                                cover = dict()
                                cover['loc'] = (h, w)
                                cover['b_covers'] = []
                                cover['dir'] = [search_direction]

                            if search_direction not in cover['dir']:
                                cover['dir'].append(search_direction)

                            found_sub_sum += 1
                            b_key = '%d-%d' % (cell_h, cell_w)
                            b_id = boundary2idx[b_key]
                            if b_id not in ignored_b_cells:
                                ignored_b_cells[b_id] = b_key

                            if l == 0:
                                near_dist = True
                            elif l == len(search_levels) - 1:
                                long_dist = True

                            if b_id not in cover['b_covers']:
                                cover['b_covers'].append(b_id)
                                if debug:
                                    print(b_key, b_id)

                    if found_sub_sum == len(search_cell_offset_l):
                        break

            if cover is not None:
                cover['prior_cost'] = 1.0
                if long_dist is True and near_dist is False:
                    cover['prior_cost'] = 0.01
                elif long_dist is True and near_dist is True:
                    cover['prior_cost'] = 0.6
                elif long_dist is False and near_dist is True:
                    cover['prior_cost'] = 1.0

                candidates_covers.append(cover)

                if len(cover['b_covers']) > max_bd_cells:
                    max_bd_cells = len(cover['b_covers'])

    # build the mat and costs
    a_mat = np.zeros((n_boundaries, len(candidates_covers)), dtype=np.bool)
    a_cost = np.zeros(len(candidates_covers))

    for c_i, cover in enumerate(candidates_covers):
        cover_set = cover['b_covers']
        for b_id in cover_set:
            a_mat[b_id, c_i] = True

        ratio = 1.0 - len(cover_set)/max_bd_cells
        a_cost[c_i] = ratio + cover['prior_cost']

    a_cost = np.clip(a_cost, a_min=0.01, a_max=np.max(a_cost))
    a_cost = a_cost / np.linalg.norm(a_cost)

    # run min set
    g = setcover.SetCover(a_mat, a_cost, maxiters=50)
    solution, time_used = g.SolveSCP()

    res = []
    res_dirs = []
    res_covers = []
    for i in range(g.s.shape[0]):
        res_g = g.s[i]
        if res_g == True:
            res.append(candidates_covers[i]['loc'])
            res_dirs.append(candidates_covers[i]['dir'])

            covers = candidates_covers[i]['b_covers']
            covers_loc = [idx2bounary[i] for i in covers]
            res_covers.append(covers_loc)

    return res, res_dirs, res_covers

def manhattan_distance(vec1, vec2):
    vec1 = np.asarray(vec1).ravel()
    vec2 = np.asarray(vec2).ravel()
    return np.linalg.norm(vec1 - vec2, ord=1)


def gen_solution_space(boundary_map, res, res_dirs, rot_cost, dist_threshold=4):

    rot_op_cost = {'left-left':0,
                   'left-top': 1,
                   'left-right': 2,
                   'left-btm': 1,
                   'top-top': 1,
                   'top-left': 1,
                   'top-right': 1,
                   'top-btm': 2,
                   'right-right': 0,
                   'right-top': 1,
                   'right-left': 2,
                   'right-btm': 1,
                   'btm-btm': 0,
                   'btm-top': 2,
                   'btm-left': 1,
                   'btm-right': 1
                   }

    # H, W = boundary_map.shape
    N = len(res)

    dist_mat = np.zeros((N, N), dtype=np.float)
    for i in range(len(res)):
        for j in range(len(res)):
            if i >= j:
                continue
            dist = manhattan_distance(res[i], res[j])
            # if dist < dist_threshold:
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist

    rot2dist = {'left': 0, 'top': 1, 'right': 2, 'btm': 3}

    # determine the states
    states_dict = {}
    for i in range(len(res)):
        if dist > 0:
            if 'left' in res_dirs[i]:
                key = '%d-left' % (i)
                if key not in states_dict:
                    states_dict[key] = len(states_dict)
            if 'top' in res_dirs[i]:
                key = '%d-top' % (i)
                if key not in states_dict:
                    states_dict[key] = len(states_dict)
            if 'right' in res_dirs[i]:
                key = '%d-right' % (i)
                if key not in states_dict:
                    states_dict[key] = len(states_dict)
            if 'btm' in res_dirs[i]:
                key = '%d-btm' % (i)
                if key not in states_dict:
                    states_dict[key] = len(states_dict)

    # add distance among covered vertices
    n_states = len(states_dict)
    dist_connections = []
    for i in range(len(res)):
        for j in range(len(res)):
            if i >= j:
                continue
            dist = dist_mat[i, j]
            if dist > 0:

                for m in range(len(res_dirs[i])):
                    m_tag = res_dirs[i][m]
                    m_state_id = states_dict['%d-%s' % (i, m_tag)]
                    for n in range(len(res_dirs[j])):
                        n_tag = res_dirs[j][n]
                        n_state_id = states_dict['%d-%s' % (j, n_tag)]
                        cost = dist + rot_op_cost['%s-%s' % (m_tag, n_tag)] * rot_cost
                        dist_connections.append((m_state_id, n_state_id, cost))

    # add rot itself
    for i in range(len(res)):
        for m in range(len(res_dirs[i])):
            m_tag = res_dirs[i][m]
            m_state_id = states_dict['%d-%s' % (i, m_tag)]
            for n in range(len(res_dirs[i])):
                if m >= n:
                    continue

                n_tag = res_dirs[i][n]
                n_state_id = states_dict['%d-%s' % (i, n_tag)]
                cost = rot_op_cost['%s-%s' % (m_tag, n_tag)] * rot_cost
                dist_connections.append((m_state_id, n_state_id, cost))

    return dist_connections, states_dict
