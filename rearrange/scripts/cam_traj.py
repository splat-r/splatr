import numpy as np
import heapq
import time
from collections import deque
from scipy.spatial.transform import Rotation as R
from rearrange.scripts.interfaces import CameraState
from rearrange.scripts.config import MapConfig


map_config = MapConfig()


def voxelize_pcd(points, voxel_size=map_config.voxel_size, grid_size_m=map_config.grid_size_m):
    grid_size_res = (int(grid_size_m[0]/voxel_size),
                     int(grid_size_m[1]/voxel_size),
                     int(grid_size_m[2]/voxel_size))
    center = np.array([int(grid_size_res[0]/2), int(grid_size_res[1]/2), int(grid_size_res[2]/2)])
    grid = np.zeros(grid_size_res, dtype=np.int8)

    # map every point into a voxel
    for point in points:
        voxel_index = center + np.floor(point / voxel_size).astype(int)
        # check bounds
        if (0 <= voxel_index[0] < grid_size_res[0] and
            0 <= voxel_index[1] < grid_size_res[1] and
            0 <= voxel_index[2] < grid_size_res[2]):
            grid[voxel_index[0], voxel_index[1], voxel_index[2]] = 1

    return grid, center


def get_point_on_grid(point, grid_center, voxel_size=map_config.voxel_size, grid_size_m=map_config.grid_size_m):
    grid_size_res = (int(grid_size_m[0] / voxel_size),
                     int(grid_size_m[1] / voxel_size),
                     int(grid_size_m[2] / voxel_size))
    point_grid = np.floor(point / voxel_size).astype(int) + grid_center
    if (0 <= point_grid[0] < grid_size_res[0] and
        0 <= point_grid[1] < grid_size_res[1] and
        0 <= point_grid[2] < grid_size_res[2]):
        return point_grid
    else:
        return None


def calc_orientation(camera_pos, waypoint_pos):
    direction = waypoint_pos - camera_pos
    direction = direction / np.linalg.norm(direction)
    z_axis = np.array([0, 0, 1])
    rotation = R.from_rotvec(np.cross(z_axis, direction))
    return rotation.as_quat()


def rot_matrix_to_quat(rot_mat):
    rot = R.from_matrix(rot_mat)
    return rot.as_quat()

def quat_to_rot(quat):
    rot = R.from_quat(quat)
    return rot.as_matrix()


def a_star_path(start_state, goal_position, grid):
    """
    start and goal state are positions on the grid
    """
    # limiting the y movement
    # grid = grid_[:, 50:130, :]
    start_time = time.time()
    def get_neighbors(camera_state):
        neighbors = []
        # directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        directions = [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)] # stay at the same z level

        for d in directions:
            # Move to the next grid
            next_pos = camera_state.position + np.array(d)

            # Check if the next grid is within the bounds
            if (0 <= next_pos[0] < grid.shape[0] and
                0 <= next_pos[1] < grid.shape[1] and
                0 <= next_pos[2] < grid.shape[2]):

                # Check if the grid is free
                if grid[int(next_pos[0]), int(next_pos[1]), int(next_pos[2])] == 0:
                    goal_position_tr = camera_state.transform_point(goal_position)
                    next_orientation = calc_orientation(next_pos, goal_position_tr)
                    # next_orientation = np.array([-0.7068252, 0, 0, 0.7073883])
                    neighbors.append(CameraState(next_pos[0], next_pos[1], next_pos[2], next_orientation))
        return neighbors

    def heuristic(camera_state, goal_pos):
        return camera_state.distance_to(CameraState(goal_pos[0], goal_pos[1], goal_pos[2], [1, 0, 0, 1]))

    open_list = []
    heapq.heappush(open_list, (0, start_state))
    came_from = {}
    g_score = {start_state: 0}
    f_score = {start_state: heuristic(start_state, goal_position)}
    while open_list:
        if (time.time() - start_time) > 120:
            print("Took too long! Go to the next waypoint")
            return None
        current = heapq.heappop(open_list)[1]
        if np.allclose(current.position, goal_position):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_state)
            return path[::-1]

        for neighbor in get_neighbors(current):
            tentative_g_score = g_score[current] + current.distance_to(neighbor)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_position)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    # If path not found
    return None


def get_interior_points_flood_fill(voxel_grid, start_state):
    """
    start_state has to be within the navigable region
    """
    if len(voxel_grid.shape) == 3:
        print("starting flood fill...")
        t1 = time.time()
        X, Y, Z = voxel_grid.shape
        visited = np.zeros_like(voxel_grid, dtype=bool)
        interior_points = []
        queue = deque([start_state])

        while queue:
            x, y, z = queue.popleft()
            if visited[x, y, z] or voxel_grid[x, y, z] == 1:
                continue
            visited[x, y, z] = True
            interior_points.append((x, y, z))

            # explore the connected neighbours of the popped point on the grid
            for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z:
                    if not visited[nx, ny, nz] and voxel_grid[nx, ny, nz] == 0:
                        queue.append((nx, ny, nz))
        t2 = (time.time() - t1)
        print("flood fill done in : ", t2)
        return np.array(interior_points)
    else:
        print("starting flood fill...")
        t1 = time.time()
        X, Y = voxel_grid.shape
        visited = np.zeros_like(voxel_grid, dtype=bool)
        interior_points = []
        queue = deque([start_state])

        while queue:
            x, y = queue.popleft()
            if visited[x, y] or voxel_grid[x, y] == 1:
                continue
            visited[x, y] = True
            interior_points.append((x, y))

            # explore the connected neighbours of the popped point on the grid
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < X and 0 <= ny < Y:
                    if not visited[nx, ny] and voxel_grid[nx, ny] == 0:
                        queue.append((nx, ny))
        t2 = (time.time() - t1)
        print("flood fill done in : ", t2)
        return np.array(interior_points)


def stratified_random_sampling(grid_, n_strata_h, n_strata_w, samples_per_strata):
    grid = grid_[175:575, 175:575]
    strata_h = grid.shape[0] // n_strata_h
    strata_w = grid.shape[1] // n_strata_w
    n_strata = n_strata_h * n_strata_w

    sampled_points = []
    # We only start from the first strata to make sure that,
    # we don't look over edges of walls while rendering images
    for i in range(n_strata_h):
        for j in range(n_strata_w):
            y_start = i * strata_h
            y_end = min((i + 1) * strata_h, grid.shape[0])
            x_start = j * strata_w
            x_end = min((j + 1) * strata_w, grid.shape[1])

            curr_stratum = grid[y_start:y_end, x_start:x_end]
            free_grids_in_stratum = np.argwhere(curr_stratum == True)
            not_free_grids_in_stratum = np.argwhere(curr_stratum == False)
            if True:
                if len(free_grids_in_stratum) > 0:
                    # Random sample a coordinate from the true locations of the current stratum
                    random_sample = np.random.choice(len(free_grids_in_stratum),
                                                     min(samples_per_strata, len(free_grids_in_stratum)),
                                                     replace=False)
                    for idx in random_sample:
                        y, x = free_grids_in_stratum[idx]
                        # Converting points from local strata to global grid
                        print("y : ", y_start+y+175)
                        print("x : ", x_start+x+175)
                        sampled_points.append((y_start+y+175, x_start+x+175))
                else:
                    print("NO FREE SPACE IN THIS STRATA OF THE GRID")
            else:
                print("PENUMBRA STRATA")
    return sampled_points