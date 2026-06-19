/*
 * The original code is under the following copyright:
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE_GS.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * The modifications of the code are under the following copyright:
 * Copyright (C) 2024, University of Liege, KAUST and University of Oxford
 * TELIM research group, http://www.telecom.ulg.ac.be/
 * IVUL research group, https://ivul.kaust.edu.sa/
 * VGG research group, https://www.robots.ox.ac.uk/~vgg/
 * All rights reserved.
 * The modifications are under the LICENSE.md file.
 *
 * For inquiries contact jan.held@uliege.be
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16

#define MAX_NB_POINTS 3

// Depth interpolation mode (recompile after changing):
//   1 = per-pixel barycentric depth (depth varies across the triangle)
//   0 = constant per-triangle depth (centroid depth, original behavior)
// In constant mode the barycentric weights are forced to 1/3, which makes the
// per-pixel depth equal the mean of the vertex depths. Because the view-space z is
// affine in world position, that mean equals the centroid depth exactly, so this
// reproduces the original constant-depth forward and backward (the 1/3 gradient split).
#define DEPTH_BARYCENTRIC 0

#endif