#!/usr/bin/env python

# Set multiple initial states (one for each ) from which to build trees

import sst_nonlin_quadrotor_3 as sst
from sst_nonlin_quadrotor_3 import MultiGraph, Graph, Node, Edge
from lqr import *
from save_obj import load_object
import numpy as np
from numpy import sin, cos, tan
import pygame
from scipy import integrate
from pygame.locals import *
from numpy.linalg import multi_dot
import os

N = 199


# #--------------------------------# #
def main():
    # Initialization
    pygame.init()
    screen = pygame.display.set_mode(sst.WINSIZE)
    pygame.display.set_caption('SST')
    white = 255, 240, 240
    black = 20, 20, 40
    green = 50, 150, 50
    screen.fill(white)
    rects = []  # list of rectangles for redrawing
    colour_lst = sst.colour_lst

    rects = []  # list of rectangles for redrawing
    for i in range(sst.num_props):
        wh = sst.regionsize_list[i]  # pair (width, height)
        colour = colour_lst[i]
        region = sst.init_coord_list[i][:2]  # x, y values from init state [i]
        # adjust so goal in centre of region by subtracting half of wh[i]
        border_rect = pygame.Rect(region[0] - wh[0]/2., region[1] - wh[1]/2.,
                                  wh[0], wh[1])
        rects.append(border_rect)
        pygame.draw.rect(screen, colour, border_rect)
        pygame.draw.rect(screen, black, border_rect, 1)

    sst.obsDraw(screen, sst.OBS)
    pygame.display.update()

    r = str(np.random.randint(1000))
    os.chdir("./results/doubleint")
    os.mkdir("sampling"+r)
    os.chdir("sampling"+r)

    pygame.image.save(screen, "sample{:0>3}.bmp".format(0))
    iternum = 1
    for i in range(N):
        for j in range(10):
            px = int(np.random.rand()*sst.XDIM)
            py = int(np.random.rand()*sst.YDIM)
            pygame.draw.circle(screen, [30, 220, 70], (px, py), 4)
        pygame.display.update()
        pygame.image.save(screen, "sample{:0>3}.bmp".format(iternum))
        iternum += 1


if __name__ == '__main__':
    main()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
