# Code is modified from https://github.com/alsignoriello/vicsek_model

import cv2, os, sys

import numpy as np
import matplotlib.pyplot as plt

from cv2 import VideoWriter, VideoWriter_fourcc
from geometry import angle_2_vector, rand_angle, vector_2_angle
from neighbor import *

def plot_vectors(coords, thetas):

	# generate random color for every particle
	colors = ["b", "g", "y", "m", "c", "pink", "purple", "seagreen",
			"salmon", "orange", "paleturquoise", "midnightblue",
			"crimson", "lavender"]

	
	for i, (x, y) in enumerate(coords):

		c = colors[i % len(colors)]

		# plot point
		plt.scatter(x, y, color = c, marker = ".")

		# plot tail
		theta = thetas[i]
		v = angle_2_vector(theta)
		x1 = x - (0.025 * v[0])
		y1 = y - (0.025 * v[1])
		plt.plot([x, x1], [y, y1], color=c)

	return



def save_plot(path, fname, eta):

    # axes between 0 and 1
    plt.axis([0, 1, 0, 1])

    # remove tick marks
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])

    # title 
    plt.title("η = %.2f" % eta)

    # save plot
    plt.savefig(os.path.join(path, fname[:-4]+".jpg"))
    plt.close()

    # clear for next plot
    plt.cla()

    return


# ------------------------------- RUNS FROM HERE -------------------------------

if __name__ == '__main__':

    # keeping it tidy
    simdir = os.path.join(os.getcwd(), "simulation")
    particledir = os.path.join(simdir, "particles")
    plotdir = os.path.join(simdir, "plots")

    if not os.path.exists(simdir):
        os.mkdir(simdir)
    if not os.path.exists(particledir):
        os.mkdir(os.path.join(simdir, "particles"))
    if not os.path.exists(plotdir):
        os.mkdir(os.path.join(simdir, "plots"))
            
    N = 100           # num of particles
    eta = 0.2        # noise in [0,1]
    r = 0.5          # radius
    delta_t = 0.05   # time step

    # Maximum time
    t = 0.0
    T = 5.0

    # Generate random particle coordinates
    # particles[i,0] = x
    # particles[i,1] = y
    particles = np.random.uniform(0, 1, size=(N, 2))

    # initialize random angles
    thetas = np.zeros((N, 1))
    for i, theta in enumerate(thetas):
        thetas[i, 0] = rand_angle()

    os.chdir(particledir)

    print("Creating particle files", end='', flush=True)
    # Currently run until time ends
    while t < T:

        print(end='.', flush=True)
        # save coordinates & corresponding thetas to a text file
        output = np.concatenate((particles, thetas), axis=1)
        np.savetxt("%.2f.txt" % t, output)


        for i, (x, y) in enumerate(particles):

            # get neighbor indices for current particle
            neighbors = get_neighbors(particles, r, x, y)

            # get average theta vector
            avg = get_average(thetas, neighbors)

            # get noise vector
            nx = rand_angle()
            ny = rand_angle()
            noise = eta * np.array([nx,ny])

            # move to new position 
            particles[i,:] += delta_t * (avg + noise)

            # get new theta
            thetas[i] = vector_2_angle(avg + noise)

            # assure correct boundaries (xmax, ymax) = (1,1)
            if particles[i, 0] < 0:
                particles[i, 0] = 1 + particles[i, 0]

            if particles[i, 0] > 1:
                particles[i, 0] = particles[i, 0] - 1

            if particles[i, 1] < 0:
                particles[i, 1] = 1 + particles[i, 1]

            if particles[i, 1] > 1:
                particles[i, 1] = particles[i, 1] - 1

        # new time step
        t += delta_t
    print()

    print("Processing particles", end='', flush=True)
    txt_files = [i for i in os.listdir(particledir) if i.endswith(".txt")]
    for fname in txt_files:
        print(end = ".", flush=True)
        f = os.path.join(particledir, fname) # the actual file

        # read in data
        mat = np.loadtxt(f)
        coords = mat[:,0:2]
        thetas = mat[:,2]
        plot_vectors(coords, thetas)
        save_plot(plotdir, fname, eta)
    print()
    
    # ------------- make the video ---------------
    jpg_files = sorted([i for i in os.listdir(plotdir) if i.endswith("jpg")])

    img_array = []
    one_size = [0,0]
    for fname in jpg_files:
        f = os.path.join(plotdir, fname) # the actual file
        img = cv2.imread(f)
        height, width, layers = img.shape
        size = (width,height)
        one_size = [width, height]
        img_array.append(img)

    video_path = os.path.join(simdir, "zooming.mp4")
    print("Saving video as", video_path)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, tuple(one_size))

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

