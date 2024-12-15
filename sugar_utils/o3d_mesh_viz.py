mesh_path = "/home/nune/gaussian_splatting/lgsplat-mesh/output/coarse_mesh/FloorPlan202_physics/sugarmesh_3Dgs7000_sdfestim02_sdfnorm02_level03_decim200000.ply"
import open3d as o3d
o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)


print("Let's draw a box using o3d.geometry.LineSet.")

delta = 0.05
bbox_3d = {'min': [0.0267 - delta,  0.7292 - delta, -0.6246 - delta],
           'max': [0.6227 + delta,  1.0007 + delta, -0.0830 + delta]}

        # for viz TODO: ADD viz
points= [
    [bbox_3d['min'][0], bbox_3d['min'][1], bbox_3d['min'][2]],
    [bbox_3d['max'][0], bbox_3d['min'][1], bbox_3d['min'][2]],
    [bbox_3d['max'][0], bbox_3d['max'][1], bbox_3d['min'][2]],
    [bbox_3d['min'][0], bbox_3d['max'][1], bbox_3d['min'][2]],
    [bbox_3d['min'][0], bbox_3d['min'][1], bbox_3d['max'][2]],
    [bbox_3d['max'][0], bbox_3d['min'][1], bbox_3d['max'][2]],
    [bbox_3d['max'][0], bbox_3d['max'][1], bbox_3d['max'][2]],
    [bbox_3d['min'][0], bbox_3d['max'][1], bbox_3d['max'][2]],
]

    # [1.9294, 1, 0],
    # [1, 1, 0],
    # [0, 0, 1],
    # [1, 0, 1],
    # [0, 1, 1],
    # [1, 1, 1],

lines = [
    [0, 1],
    [0, 3],
    [1, 2],
    [2, 3],
    [4, 5],
    [5, 1],
    [5, 6],
    [4, 7],
    [0, 4],
    [7, 6],
    [2, 6],
    [3, 7],
]

p1 = [[-0.43401381, 1.4160459, -1.37577295], [0.33192331,  0.9017114,  -0.48842561]]
l1 = [[0, 1]]
p2 = [[0.33192331,  0.9017114,  -0.48842561], [0.3247,  0.8649, -0.3538]]
l2 = [[0, 1]]
colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
line_new1 = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(p1),
    lines=o3d.utility.Vector2iVector(l1),
)
colors_1 = [[0, 1, 0] for i in range(len(l1))]
line_new1.colors = o3d.utility.Vector3dVector(colors_1)

line_new2 = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(p2),
    lines=o3d.utility.Vector2iVector(l2),
)
colors_2 = [[0, 0, 1] for i in range(len(l2))]
line_new2.colors = o3d.utility.Vector3dVector(colors_2)

o3d.visualization.draw_geometries([o3d_mesh, line_set, line_new1, line_new2])