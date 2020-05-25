import matplotlib.pyplot as pyplot
import numpy as np
import threading
import noise
from multiprocessing import Pool
from matplotlib.colors import ListedColormap
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import misc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def perlin_array(array,shape = (200, 200),
			scale=100, octaves = 6,
			persistence = 0.55,
			lacunarity = 2.0,
			seed = None):
    if not seed:
        seed = np.random.randint(0, 100)
    arr = array
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            arr[i][j] = noise.pnoise2(i / scale,
                                      j / scale,
                                      octaves=octaves,
                                      persistence=persistence,
                                      lacunarity=lacunarity,
                                      repeatx=100,
                                      repeaty=100,
                                      base=seed)
    max_arr = np.max(arr)
    min_arr = np.min(arr)
    norm_me = lambda x: (x-min_arr)/(max_arr - min_arr)
    norm_me = np.vectorize(norm_me)
    arr = norm_me(arr)
    return arr

def setup_array(tam_x,tam_y):
    shape = (tam_x, tam_y)
    size = tam_x * tam_y / 2
    array = np.concatenate((np.ones(int(size)), np.zeros(int(size))), axis=0)
    np.random.shuffle(array)
    array = np.reshape(array, shape)
    return array

def check_neighbours_optimized2(array, n_vec):
    matrix_aux = array
    for x in range(20):
        for pos_x in range(array.shape[0]):
            for pos_y in range(array.shape[1]):
                alive = 0
                arr_x = [pos_x - 1, pos_x, pos_x + 1]
                arr_y = [pos_y - 1, pos_y, pos_y + 1]
                for x in arr_x:
                    for y in arr_y:
                        try:
                            alive += matrix_aux[x, y]
                        except:
                            pass
                if alive >= n_vec:
                    matrix_aux[pos_x, pos_y] = 1
                else:
                    matrix_aux[pos_x, pos_y] = 0
    return array

def principal_thread(x):

    array = setup_array(200,200)
    return check_neighbours_optimized2(array,5)



if __name__ == '__main__':
    pool = Pool(processes = 4)

    array_list = pool.map(principal_thread,range(1,7))
    result = array_list[0]
    for x in range(1,len(array_list) - 1):
        result = np.add(result,array_list[x])

    aux = perlin_array(result)
    x = np.arange(0, 200, 1)
    y = np.arange(0, 200, 1)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, aux, cmap=cm.terrain,
                           linewidth=1, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    pyplot.matshow(result)
    pyplot.show()

