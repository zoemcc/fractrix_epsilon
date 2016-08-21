# Import libraries for simulation
import tensorflow as tf
import numpy as np

# Imports for visualization
import PIL.Image
from cStringIO import StringIO
#from IPython.display import clear_output, Image, display
import scipy.ndimage as nd
import scipy


from IPython import embed

import mcubes

def nth_power(v, n):
    """ 
    computes the nth spherical power of the tensor of 3 vectors
    v is a PointTensor
    n is the dimension to compute the power by
    from https://en.wikipedia.org/wiki/Mandelbulb
    """
    r = norm(v)
    r_n = tf.pow(r, n)
    theta = tf.acos(v.z / r)
    phi = tf.atan(v.y / (v.x + 1e-10))
    cos_theta = tf.cos(n * theta)
    sin_theta = tf.sin(n * theta)
    cos_phi = tf.cos(n * phi)
    sin_phi = tf.sin(n * phi)
    new_x = r_n * sin_theta * cos_phi
    new_y = r_n * sin_theta * sin_phi
    new_z = r_n * cos_theta
    new_v = PointTensor(new_x, new_y, new_z)
    return new_v


def sphere(region_v, center, radius):
    neg_center_v = PointTensor(-center[0:1], -center[1:2], -center[2:3])
    centered_region_v = add_point_tensor(region_v, neg_center_v)
    norm_from_center = norm(centered_region_v)
    int_ext_map = np.ones_like(norm_from_center)
    int_ext_map[norm_from_center > radius] = -1.0
    return int_ext_map

def union(int_ext_map_1, int_ext_map_2):
    return_int_ext_map = -np.ones_like(int_ext_map_1)
    return_int_ext_map[int_ext_map_1 > 0.0] = 1.0
    return_int_ext_map[int_ext_map_2 > 0.0] = 1.0
    return return_int_ext_map

def intersect(int_ext_map_1, int_ext_map_2):
    return_int_ext_map = -np.ones_like(int_ext_map_1)
    return_int_ext_map[np.logical_and(int_ext_map_1 > 0.0, int_ext_map_2 > 0.0)] = 1.0
    return return_int_ext_map

def torus(region_v, center, c, a):
    neg_center_v = PointTensor(-center[0:1], -center[1:2], -center[2:3])
    centered_region_v = add_point_tensor(region_v, neg_center_v)
    lhs = np.square(c - \
            np.sqrt(np.square(centered_region_v.x) +\
            np.square(centered_region_v.z))) +\
            np.square(centered_region_v.y)
    rhs = np.square(a)

    int_ext_map = np.ones_like(region_v.y)
    int_ext_map[rhs < lhs] = -1.0
    return int_ext_map



def connected_component(int_ext_map, connect_to_indices):
    if int_ext_map[connect_to_indices[0], connect_to_indices[1], connect_to_indices[2]] > 0:
        scale_interior = 1.0
    else:
        scale_interior = -1.0
    already_in_stack_or_looked_at = set((tuple(connect_to_indices), ))
    print int_ext_map[connect_to_indices[0], connect_to_indices[1], connect_to_indices[2]]
    connected_component_set = set((tuple(connect_to_indices), ))
    print connected_component_set
    int_ext_map_shape = int_ext_map.shape
    stack = []
    connected_int_ext_map = -scale_interior * np.ones_like(int_ext_map)
    local_indices = [[-1,0,0], [1,0,0], [0,-1,0], [0,1,0], [0,0,-1], [0,0,1]]
    print local_indices
    new_index = [0,0,0]

    for local_index in local_indices:
        within_range = True
        for i in range(3):
            new_index[i] = local_index[i] + connect_to_indices[i]
            if not (new_index[i] < int_ext_map_shape[i] and new_index[i] >= 0): # not within the region
                within_range = False
        if within_range:
            stack.append(new_index)




    while len(stack) > 0:
        index = stack.pop()
        #print index
        #print int_ext_map[index[0], index[1], index[2]]
        
        current_within_int = int_ext_map[index[0], index[1], index[2]] == scale_interior
        if current_within_int:
            neighbor_within_connected_comp = False
            for local_index in local_indices:
                within_range = True
                for i in range(3):
                    new_index[i] = local_index[i] + index[i]
                    if not (new_index[i] < int_ext_map_shape[i] and new_index[i] >= 0): # not within the region
                        within_range = False
                if within_range:
                    new_index_tuple = tuple(new_index)
                    if new_index_tuple in connected_component_set:
                        neighbor_within_connected_comp = True
            if neighbor_within_connected_comp:

                connected_int_ext_map[index[0], index[1], index[2]] = scale_interior
                connected_component_set.add(tuple(index))
                #print 'interior adding: ', index
                for local_index in local_indices:
                    within_range = True
                    for i in range(3):
                        new_index[i] = local_index[i] + index[i]
                        if not (new_index[i] < int_ext_map_shape[i] and new_index[i] >= 0): # not within the region
                            within_range = False
                    if within_range:
                        new_index_tuple = tuple(new_index)
                        if not (new_index_tuple in already_in_stack_or_looked_at):
                            already_in_stack_or_looked_at.add(new_index_tuple)
                            stack.append(new_index_tuple)
    print 'connected component size: ', len(connected_component_set)
    print 'all interior size: ', len(np.where(np.abs(int_ext_map.flatten() - scale_interior) < 1e-1)[0])

    return connected_int_ext_map

def find_closest_interior(center, index_range, int_ext_map_val):
    current_min_norm_index = None
    cur_min_norm = np.inf

    middle_cube_range = index_range
    for i in range(-middle_cube_range, middle_cube_range + 1):
        for j in range(-middle_cube_range, middle_cube_range + 1):
            for k in range(-middle_cube_range, middle_cube_range + 1):
                current_index = [i + center[0], j + center[1], k + center[2]]
                val = int_ext_map_val[current_index[0], current_index[1], current_index[2]]
                if val == 1.0:
                    cur_norm = np.sqrt(1e-10 + np.sum(np.square(np.array(current_index) - np.array(center))))
                    if cur_min_norm > cur_norm:
                        current_min_norm_index = current_index
                        cur_min_norm = cur_norm
                    #print val
    print cur_min_norm
    print np.array(current_min_norm_index) - np.array(center)
    print current_min_norm_index
    return current_min_norm_index


def norm(v):
    if isinstance(v.x, type(np.zeros((1,)))):
        return np.sqrt(v.x * v.x + v.y * v.y + v.z * v.z + 1e-10)
    else:
        return tf.sqrt(v.x * v.x + v.y * v.y + v.z * v.z + 1e-10)

def add_point_tensor(a, b):
    return PointTensor(a.x + b.x, a.y + b.y, a.z + b.z)

def cube_region(center, scale, resolution):
    negative_corner = center - scale / 2
    positive_corner = center + scale / 2
    x_tensor = np.linspace(negative_corner[0], positive_corner[0], resolution)
    y_tensor = np.linspace(negative_corner[1], positive_corner[1], resolution)
    z_tensor = np.linspace(negative_corner[2], positive_corner[2], resolution)
    #region = np.meshgrid(x_tensor, y_tensor, z_tensor)
    region = np.transpose(np.stack(np.meshgrid(x_tensor, y_tensor, z_tensor), -1), [1, 0, 2, 3])
    region_v = PointTensor(region[...,0], region[...,1], region[...,2])
    return region_v

def int_ext_map(ns, max_iter):
    in_region = max_iter - ns != 0
    out_region = max_iter - ns == 0
    return_regions = np.zeros_like(ns)
    return_regions[in_region] = 1.0
    return_regions[out_region] = -1.0
    
    return return_regions
    
    

    

def eval_point_tensor(v, sess):
    return PointTensor(sess.run(v.x), sess.run(v.y), sess.run(v.z))


class PointTensor(object):
    def __init__(self, x_tensor, y_tensor, z_tensor):
        """ all input tensors must have the same shape """
        print type(x_tensor)
        if isinstance(x_tensor, type(np.zeros((1,)))):
            assert (x_tensor.shape == y_tensor.shape) and \
                (x_tensor.shape == z_tensor.shape)
        elif isinstance(x_tensor, type(tf.zeros([1]))):
            assert (x_tensor.get_shape() == y_tensor.get_shape()) and \
                (x_tensor.get_shape() == z_tensor.get_shape())
        self.x = x_tensor
        self.y = y_tensor
        self.z = z_tensor


def DisplayFractal(a, fmt='png'):
  """Display an array of iteration counts as a
     colorful picture of a fractal."""
  a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
  img = np.concatenate([10+20*np.cos(a_cyclic),
                        30+50*np.sin(a_cyclic),
                        155-80*np.cos(a_cyclic)], 2)
  img[a==a.max()] = 0
  a = img
  a = np.uint8(np.clip(a, 0, 255))
  f = StringIO()
  PIL.Image.fromarray(a).save(f, fmt)
  display(Image(data=f.getvalue()))
  scipy.misc.imsave('mandelbrot.png', a)

def main():

    center = np.zeros((3,))
    scale = 4.0
    resolution = 400
    n = 8
    max_iter = 200
    julia = None

    region = cube_region(center=center, scale=scale, resolution=resolution)

    x_var = tf.Variable(region.x)
    y_var = tf.Variable(region.y)
    z_var = tf.Variable(region.z)
    region_var = PointTensor(x_var, y_var, z_var)
    print region.x.shape

    #xs = tf.constant(Z.astype("complex64"))
    #zs = tf.Variable(xs)

    sess = tf.InteractiveSession()

# Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]


    #Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
#Z = X+1j*Y

    
    #xs = tf.constant(Z.astype("complex64"))
    #xs = tf.Variable(xs)
    #zs = tf.Variable(xs)
    ns = tf.Variable(tf.zeros_like(region.x, "float32"))


    tf.initialize_all_variables().run()

# Compute the new values of z: z^2 + x
    new_region = add_point_tensor(nth_power(region_var, n), region)

# Have we diverged with this new value?
    not_diverged = norm(new_region) < 4

# Operation to update the zs and the iteration count.
#
# Note: We keep computing zs after they diverge! This
#       is very wasteful! There are better, if a little
#       less simple, ways to do this.
#
    step = tf.group(
    region_var.x.assign(new_region.x),
    region_var.y.assign(new_region.y),
    region_var.z.assign(new_region.z),
    ns.assign_add(tf.cast(not_diverged, "float32"))
    )

    for i in range(max_iter): step.run()
    print sess.run(ns)
    print sess.run(ns).min()
    print sess.run(ns).max()

    int_ext_map_val = int_ext_map(sess.run(ns), max_iter)
    print int_ext_map_val

    #current_min_norm_index = find_closest_interior([resolution / 2] * 3, 20, int_ext_map_val)
    #print int_ext_map_val.shape
    #print int_ext_map_val.shape
    center_index = [resolution / 2] * 3
    connected_int_ext_map = connected_component(-int_ext_map_val, center_index)

    #sphere_int_ext_map_1 = sphere(region, np.array([0,0,0]), 0.25)
    #sphere_int_ext_map_2 = sphere(region, np.array([1,0,0]), 0.25)
    #sphere_int_ext_map = union(sphere_int_ext_map_1, sphere_int_ext_map_2)
    #current_min_norm_index = find_closest_interior([resolution / 2] * 3, 20, sphere_int_ext_map)
    #print current_min_norm_index
    #connected_int_ext_map = connected_component(sphere_int_ext_map, current_min_norm_index)
    torus_int_ext_map = torus(region, np.array([0,0,-1.0]), 0.5, 0.1)
    torus_bulb_int_ext_map = union(connected_int_ext_map, torus_int_ext_map)



    #print connected_int_ext_map.shape
    #print connected_int_ext_map

    #vertices, triangles = mcubes.marching_cubes(connected_int_ext_map, 0)
    #vertices, triangles = mcubes.marching_cubes(sphere_int_ext_map, 0)
    #vertices, triangles = mcubes.marching_cubes(torus_int_ext_map, 0)
    vertices, triangles = mcubes.marching_cubes(torus_bulb_int_ext_map, 0)
    #vertices, triangles = mcubes.marching_cubes(connected_int_ext_map, 0)
    #center_index = np.array(center_index)
    #vertices = vertices - center_index
    print vertices
    print vertices.min()
    print vertices.max()

    #mcubes.export_mesh(vertices, triangles, "mandelbulb_connected_0.dae", "Mandelbulb_first_connected_rendering")
    #mcubes.export_mesh(vertices, triangles, "sphere.dae", "sphere_rendering")
    mcubes.export_mesh(vertices, triangles, "torus_bulb.dae", "torus_bulb_rendering")
    print 'rendering and exporting complete'

#DisplayFractal(ns.eval())

if __name__=='__main__':
    main()

