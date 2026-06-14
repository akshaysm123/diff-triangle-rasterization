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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;



// Forward method for converting the input spherical harmonics
// coefficients of each Triangle to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 means, glm::vec3 campos, const float* shs, bool* clamped)
{
    // The implementation is loosely based on code for 
    // "Differentiable Point-Based Radiance Fields for 
    // Efficient View Synthesis" by Zhang et al. (2022)
    glm::vec3 pos = means;
    glm::vec3 dir = pos - campos;
    dir = dir / glm::length(dir);

    glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
    glm::vec3 result = SH_C0 * sh[0];

    if (deg > 0)
    {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result +
                SH_C2[0] * xy * sh[4] +
                SH_C2[1] * yz * sh[5] +
                SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                SH_C2[3] * xz * sh[7] +
                SH_C2[4] * (xx - yy) * sh[8];

            if (deg > 2)
            {
                result = result +
                    SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                    SH_C3[1] * xy * z * sh[10] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                    SH_C3[5] * z * (xx - yy) * sh[14] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    result += 0.5f;

    // RGB colors are clamped to positive values. If values are
    // clamped, we need to keep track of this for the backward pass.
    clamped[3 * idx + 0] = (result.x < 0);
    clamped[3 * idx + 1] = (result.y < 0);
    clamped[3 * idx + 2] = (result.z < 0);
    return glm::max(result, 0.0f);
}



// Perform initial steps for each Triangle prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
    const float* triangles_points,
    const float* sigma,
    const int* num_points_per_triangle,
    const int* cumsum_of_points_per_triangle,
    const float* opacities,
    float* scaling,
    float* density_factor,
    const float* shs,
    bool* clamped,
    const float* colors_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y,
    int* radii,
    float2* normals,
    float* offsets,
    float* p_w,
    float2* p_image,
    int* indices,
    float2* points_xy_image,
    float* depths,
    float* rgb,
    float4* conic_opacity,
    float* cov3Ds,
    float2* phi_center,
    uint2* rect_min,
    uint2* rect_max,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered)
{

    // the kernel is launched with *at least* P threads, any excess threads exit
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Initialize radius and touched tiles to 0. If this isn't changed,
    // this Triangle will not be processed further.

    // offset to grab 3D vertex positions of triangle
    const int cumsum_for_triangle = cumsum_of_points_per_triangle[idx];
    const int offset = 3 * cumsum_for_triangle;

    // zero intialization to track triangles that don't contribute
    radii[idx] = 0;
    tiles_touched[idx] = 0;
    scaling[idx] = 0.0f;
    density_factor[idx] = 0.0f;

    float stopping_influence = 0.01f;


    // if the opacity is too low, we can skip the Triangle
    if (opacities[idx] < stopping_influence)
        return;

    float3 center_triangle = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < num_points_per_triangle[idx]; i++) {
        // indices is written here so the backward pass can recover the local vertex index
        indices[cumsum_for_triangle + i] = i;
        center_triangle.x += triangles_points[offset + 3 * i];
        center_triangle.y += triangles_points[offset + 3 * i + 1];
        center_triangle.z += triangles_points[offset + 3 * i + 2];
    }

    // center as arihtmetic mean
    center_triangle.x /= num_points_per_triangle[idx];
    center_triangle.y /= num_points_per_triangle[idx];
    center_triangle.z /= num_points_per_triangle[idx];

    // Perform near culling, quit if outside.
    float3 p_view_triangle;
    // (auxiliary.h) tansforms the centroid with viewmatrix to obtain centroid location in camera space `p_view_triangle`
    // the triangle is discarded if the z-coord < near plane. 
    if (!in_frustum_triangle(idx, center_triangle, viewmatrix, projmatrix, prefiltered, p_view_triangle)){
        return;
    }

    // Calculate the normal of the Triangle
    float3 normal_cvx = {0.0f, 0.0f, 0.0f};
    float3 p0 = make_float3(
        triangles_points[offset + 0],
        triangles_points[offset + 1],
        triangles_points[offset + 2]
    );
    float3 p1 = make_float3(
        triangles_points[offset + 3],
        triangles_points[offset + 4],
        triangles_points[offset + 5]
    );
    float3 p2 = make_float3(
        triangles_points[offset + 6],
        triangles_points[offset + 7],
        triangles_points[offset + 8]
    );

    float3 v1 = make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
    float3 v2 = make_float3(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);

    float3 cross_prod = make_float3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    );
    // nonrmal vector to camera space
    cross_prod = transformVec4x3(cross_prod, viewmatrix);

    // unit length
    float length_cross = __fsqrt_rn(cross_prod.x*cross_prod.x + cross_prod.y*cross_prod.y + cross_prod.z*cross_prod.z);
    length_cross = max(length_cross, 1e-4f);
    cross_prod.x /= length_cross;
    cross_prod.y /= length_cross;
    cross_prod.z /= length_cross;

    normal_cvx = cross_prod;

    // normalize the camera viewpoint direction
    float length_viewpoint = __fsqrt_rn(p_view_triangle.x * p_view_triangle.x + 
        p_view_triangle.y * p_view_triangle.y + 
        p_view_triangle.z * p_view_triangle.z);
    length_viewpoint = max(length_viewpoint, 1e-4f);

    // this is the unit vector from camera center to center of the triangle in camera space
    float3 normalized_camera_center;
    normalized_camera_center.x = p_view_triangle.x / length_viewpoint;
    normalized_camera_center.y = p_view_triangle.y / length_viewpoint;
    normalized_camera_center.z = p_view_triangle.z / length_viewpoint;

    // Compute cosine (before flipping the normal)
    float cos_theta = normal_cvx.x * normalized_camera_center.x +
        normal_cvx.y * normalized_camera_center.y +
        normal_cvx.z * normalized_camera_center.z;

    // Flip the normal if needed (ensure it faces the camera)
    if (cos_theta > 0) {
        normal_cvx.x = -normal_cvx.x;
        normal_cvx.y = -normal_cvx.y;
        normal_cvx.z = -normal_cvx.z;
        cos_theta = -cos_theta; 
    }

    // if the triangle is edge-on to the camera
    const float threshold = 0.001f;
    if (fabsf(cos_theta) < threshold) {
        return;
    }


    float4 p_hom_center = transformPoint4x4(center_triangle, projmatrix);   // triangle center (world) to NDC
    float p_w_center = 1.0f / (p_hom_center.w + 0.0000001f);    // homogeneous division factor
    float3 center_triangle_camera_view = { p_hom_center.x * p_w_center, p_hom_center.y * p_w_center, p_hom_center.z * p_w_center }; // center in NDC
    float2 center_triangle_2D = { ndc2Pix(center_triangle_camera_view.x, W), ndc2Pix(center_triangle_camera_view.y, H) }; // center in pix


    float distance = 0.0f;
    float distance_points = 0.0f;

    // world to screen for triangle vertices
    for (int i = 0; i < num_points_per_triangle[idx]; i++) {
        float3 triangle_point = {triangles_points[offset + 3 * i], triangles_points[offset + 3 * i + 1], triangles_points[offset + 3 * i + 2]};
        float4 p_hom = transformPoint4x4(triangle_point, projmatrix);
        p_w[cumsum_for_triangle + i] = 1.0f / (p_hom.w + 0.0000001f);   // store homogeneous division factor
        float3 p_proj = { p_hom.x * p_w[cumsum_for_triangle + i], p_hom.y * p_w[cumsum_for_triangle + i], p_hom.z * p_w[cumsum_for_triangle + i] }; // NDC of vertices
        p_image[cumsum_for_triangle + i] = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };  // screen coordinates of vertices

        // calculate distance from vertex to centroid in screenspace
        distance = __fsqrt_rn(  (p_image[cumsum_for_triangle + i].x - center_triangle_2D.x) * (p_image[cumsum_for_triangle + i].x - center_triangle_2D.x)+ 
                                (p_image[cumsum_for_triangle + i].y - center_triangle_2D.y) * (p_image[cumsum_for_triangle + i].y - center_triangle_2D.y));

        // save furthest distance
        if (distance > distance_points) {
            distance_points = distance;
        }
    }

    // Get the three projected 2D points
    float2 A1 = p_image[cumsum_for_triangle + 0];
    float2 B1 = p_image[cumsum_for_triangle + 1];
    float2 C1 = p_image[cumsum_for_triangle + 2];

    // Compute side lengths (opposite each vertex)
    float a = __fsqrt_rn((B1.x - C1.x) * (B1.x - C1.x) + (B1.y - C1.y) * (B1.y - C1.y)); // Opposite A
    float b = __fsqrt_rn((A1.x - C1.x) * (A1.x - C1.x) + (A1.y - C1.y) * (A1.y - C1.y)); // Opposite B
    float c = __fsqrt_rn((A1.x - B1.x) * (A1.x - B1.x) + (A1.y - B1.y) * (A1.y - B1.y)); // Opposite C

    float sum = a + b + c;

    // Incenter weighted by opposite side lengths
    float2 incenter;
    incenter.x = (a * A1.x + b * B1.x + c * C1.x) / sum;
    incenter.y = (a * A1.y + b * B1.y + c * C1.y) / sum;

    float dist = 0.0f;

    for (int i = 0; i < 3; i++) {
        // Points forming the segment
        float2 p1_conv = p_image[cumsum_for_triangle + i];
        float2 p2_conv = p_image[cumsum_for_triangle + (i + 1) % 3];

        // 90deg rotation of line segment p2_conv - p1_conv = [nx, ny]
        float nx = p2_conv.y - p1_conv.y;
        float ny = -(p2_conv.x - p1_conv.x);
        float norm = __fsqrt_rn(nx * nx + ny * ny);
        float inv_norm = 1.0f / norm;

        // Calculate normalized normal and offset
        float2 normal = {nx * inv_norm, ny * inv_norm};

        // from the normal form of a 2D line:  n dot p + offset = 0 --> -(n dot p) = offset
        float offset = - (normal.x * p1_conv.x + normal.y * p1_conv.y);

        // normal is unit length: n dot p + offset = signed distance perp to line segment
        dist = normal.x * incenter.x + normal.y * incenter.y + offset;

        // incenter is inside the triangle: define inside the triangle to be negative distance
        if (dist > 0) {
            normal.x = -normal.x;
            normal.y = -normal.y;
            offset = -offset;
            dist = -dist;
        }

        // store normals and offsets
        normals[cumsum_for_triangle + i] = normal;
        offsets[cumsum_for_triangle + i] = offset;
    }

    // distance_points: largest distance vertex to centroid in screen. This 1600 might need to change
    // dist: distance incenter to edge. -1 indicates the incenter is almost on the edge
    if (distance_points > 1600 or distance_points < 1 or dist > -1) {
        radii[idx] = 0;
        tiles_touched[idx] = 0;
        scaling[idx] = 0.0f;
        return;
    }

    // left/right extent
    float pix_min_x = fminf(fminf(A1.x, B1.x), C1.x);
    float pix_max_x = fmaxf(fmaxf(A1.x, B1.x), C1.x);
    // top/bottom extent
    float pix_min_y = fminf(fminf(A1.y, B1.y), C1.y);
    float pix_max_y = fmaxf(fmaxf(A1.y, B1.y), C1.y);

    // bounding box in screen tiles
    uint2 rect_min_triangle_test;
    uint2 rect_max_triangle_test;
    rect_min_triangle_test.x = min(grid.x, max(0, (uint)(pix_min_x / BLOCK_X)));
    rect_min_triangle_test.y = min(grid.y, max(0, (uint)(pix_min_y / BLOCK_Y)));
    rect_max_triangle_test.x = min(grid.x, max(0, (uint)((pix_max_x + BLOCK_X - 1) / BLOCK_X)));
    rect_max_triangle_test.y = min(grid.y, max(0, (uint)((pix_max_y + BLOCK_Y - 1) / BLOCK_Y)));

    rect_max[idx] = rect_max_triangle_test;
    rect_min[idx] = rect_min_triangle_test;

    // empty bounding box: set values to zero so they can be identified later
    if ((rect_max_triangle_test.x - rect_min_triangle_test.x) * (rect_max_triangle_test.y - rect_min_triangle_test.y) == 0){
        radii[idx] = 0;
        tiles_touched[idx] = 0;
        scaling[idx] = 0.0f;
        return;
    }

    float phi_center_min = dist;
    float max_distance = ceil(distance_points);

    // We save the 2D Size in Image Space
    scaling[idx] = max_distance;
    density_factor[idx] = -dist;

    // If colors have been precomputed, use them, otherwise convert
    // spherical harmonics coefficients to RGB color.
    if (colors_precomp == nullptr)
    {
        glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z), *cam_pos, shs, clamped);
        rgb[idx * C + 0] = result.x;
        rgb[idx * C + 1] = result.y;
        rgb[idx * C + 2] = result.z;
    }


    phi_center[idx] = {1.0f / phi_center_min, 0.0f}; // [0] = reciprocal of sigend distance to incenter (negative)
    depths[idx] = p_view_triangle.z; 
    radii[idx] = max_distance; // this is essentially a visibility check for both tile-binning and backward
    points_xy_image[idx] = center_triangle_2D;
    conic_opacity[idx] = {normal_cvx.x, normal_cvx.y, normal_cvx.z, opacities[idx]}; // naming is off, this are normals and opacity
    tiles_touched[idx] = (rect_max_triangle_test.y - rect_min_triangle_test.y) * (rect_max_triangle_test.x - rect_min_triangle_test.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float2* __restrict__ normals,
    const float* __restrict__ offsets,
    const float2* __restrict__ points_xy_image,
    const float* __restrict__ sigma,
    const int* __restrict__ num_points_per_triangle,
    const int* __restrict__ cumsum_of_points_per_triangle,
    const float* __restrict__ features,
    const float4* __restrict__ conic_opacity,
    const float* __restrict__ depths,
    const float2* __restrict__ phi_center,
    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    const float* __restrict__ bg_color,
    float* __restrict__ out_color,
    float* __restrict__ out_others,
    float* __restrict__ max_blending)
{
    // Identify current tile and associated min/max pixel range.
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X; // n.o horizontal blocks
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y }; // min pix values of this block
    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) }; // max pixel values of this block (excl)
    uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y }; // pixel value for this thread
    uint32_t pix_id = W * pix.y + pix.x; // pixel ID in flattaned list of pixels
    float2 pixf = { (float)pix.x, (float)pix.y }; // pixel coords as float

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W&& pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    // ranges is indexed on tiles, but flattened
    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    // ceiling division to determine the number of rounds for fetching data
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    // Allocate storage for batches of collectively fetched data.
    __shared__ int collected_id[BLOCK_SIZE]; // id in point list
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE]; // float4 of .xyz camera space normal, .w opacity

    /*
    ADDED FOR TRIANGLE PURPOSES ==========================================================================
    */
    __shared__ float2 collected_normals[BLOCK_SIZE * MAX_NB_POINTS]; // 2d normals of triangle edges
    __shared__ float collected_offsets[BLOCK_SIZE * MAX_NB_POINTS]; // offsets for each edge in the normal form equation
    __shared__ float collected_sigma[BLOCK_SIZE]; // per triangle sigma
    __shared__ float collected_depths[BLOCK_SIZE]; // camera space depth
    __shared__ float2 collected_xy[BLOCK_SIZE]; // 2D pixel position of the triangle centroid
    __shared__ float2 collected_phi_center[BLOCK_SIZE]; // [0] = reciprocal of sigend distance to incenter (negative)
    /*
    ===================================================================================================
    */

    // Initialize helper variables
    float T = 1.0f;
    uint32_t contributor = 0;
    uint32_t last_contributor = 0;
    float C[CHANNELS] = { 0 };
    float C_random[3] = { 0 };

    // Added from 2DGS
    float N[3] = {0};
    float D = { 0 };
    float M1 = {0};
    float M2 = {0};
    float distortion = {0};
    float median_depth = {0};
    float median_contributor = {-1};
    // Global id of the triangle that forms the surface (median) at this pixel.
    int median_j_id = -1;

    // Iterate over batches until all done or range is complete
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // End if entire block votes that it is done rasterizing
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        // Collectively fetch per-Triangle data from global to shared
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            collected_sigma[block.thread_rank()] = sigma[coll_id];
            collected_depths[block.thread_rank()] = depths[coll_id];
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            for (int k = 0; k < 3; k++) {
                collected_normals[MAX_NB_POINTS * block.thread_rank() + k] = normals[cumsum_of_points_per_triangle[coll_id] + k];
                collected_offsets[MAX_NB_POINTS * block.thread_rank() + k] = offsets[cumsum_of_points_per_triangle[coll_id] + k];
            }
            collected_phi_center[block.thread_rank()] = phi_center[coll_id];
        }
        block.sync();

        // Iterate over current batch
        // iterate over candidate triangles for the current batch
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {
            // Keep track of current position in range
            contributor++;

            int j_id = collected_id[j];
            float4 con_o = collected_conic_opacity[j];
            float normal[3] = {con_o.x, con_o.y, con_o.z};
            float2 phi_center_min = collected_phi_center[j]; // .x = reciprocal of sigend distance to incenter (negative)
            float sigma_pre = collected_sigma[j];
            float max_val = -INFINITY;
            int base = j * MAX_NB_POINTS;
            bool outside = false;

            for (int k = 0; k < 3; k++) {
                // Compute the current distance
                // normal form of edge equation: n dot p + offset = distance
                float dist = (collected_normals[base + k].x * pixf.x
                    + collected_normals[base + k].y * pixf.y
                    + collected_offsets[base + k]);

                // break out of loop if a pixel is outside the triangle
                if (dist > 0) {
                    outside = true;
                    break;
                }

                // maximum distance
                max_val = fmaxf(max_val, dist);
            }

            // skip the current triangle
            if (outside)
                continue;

            // from the windowing function: (phi(p) / phi(center))^sigma
            float phi_x = max_val;
            float phi_final = phi_x * phi_center_min.x; // these are both negative: phi_final is positive
            float Cx = fmaxf(0.0f,  __powf(phi_final, sigma_pre));

            // falloff * opacity = alpha
            float alpha = min(0.99f, con_o.w * Cx); 
            // skip nearly invisible triangles
            if (alpha < 1.0f / 255.0f)
                continue;
            // transmittance after this triangle
            float test_T = T * (1 - alpha);
            // stop once all light is consumed
            if (test_T < 0.0001f)
            {
                done = true;
                continue;
            }



            float blending_weight = alpha * T;
            // Update the maximum blending weight in a thread-safe way
            // this is a running max of blending weights. This is used to track the importance of triangles
            // and to then guide pruning. 
            atomicMax(((int*)max_blending) + j_id, *((int*)(&blending_weight)));

            // collected_depths[j] should become a convex combination of depths of the triangle vertices
            // -> barycentrics

            // distortion
            float A = 1-T;
            float m = far_n / (far_n - near_n) * (1 - near_n / collected_depths[j]);
            distortion += (m * m * A + M2 - 2 * m * M1) * blending_weight;
            D  += collected_depths[j] * blending_weight; // expected depth
            // first and second moment for distortion
            M1 += m * blending_weight;
            M2 += m * m * blending_weight;

            // median contributor
            if (T > 0.5) {
                median_depth = collected_depths[j];
                median_contributor = contributor;
                median_j_id = j_id;
            }
            // Render normal map
            for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * blending_weight;

            // rgb color
            for (int ch = 0; ch < CHANNELS; ch++)
                C[ch] += features[j_id * CHANNELS + ch] * alpha * T;

            // randomized color for visualization
            const float3 random_rgb = hsv2rgb(
                random_hue((uint32_t)j_id),
                RANDOM_COLOR_SATURATION,
                RANDOM_COLOR_VALUE);
            C_random[0] += random_rgb.x * blending_weight;
            C_random[1] += random_rgb.y * blending_weight;
            C_random[2] += random_rgb.z * blending_weight;

            T = test_T;

            // Keep track of last range entry to update this
            // pixel.
            last_contributor = contributor;
        }
    }

    // All threads that treat valid pixel write out their final
    // rendering data to the frame and auxiliary buffers.
    if (inside)
    {
        out_others[pix_id + 0 * H * W] = last_contributor; // this gets overwritten as DEPTH_OFFSET=0
        final_T[pix_id] = T;
        n_contrib[pix_id] = last_contributor;
        for (int ch = 0; ch < CHANNELS; ch++)
            out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

        n_contrib[pix_id + H * W] = median_contributor;
        final_T[pix_id + H * W] = M1;
        final_T[pix_id + 2 * H * W] = M2;
        out_others[pix_id + DEPTH_OFFSET * H * W] = D;
        out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
        for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
        out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
        out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
        for (int ch = 0; ch < 3; ch++)
            out_others[pix_id + (RANDOM_COLOR_OFFSET + ch) * H * W] = C_random[ch] + T * bg_color[ch];
        // Surface-triangle id stored as float (exact for ids < 2^24, well above max_shapes).
        // -1 marks background / empty pixels with no surface.
        out_others[pix_id + SURFACE_ID_OFFSET * H * W] = (float)median_j_id;
    }
}

void FORWARD::render(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float2* normals,
    const float* offsets,
    const float2* points_xy_image,
    const float* sigma,
    const int* num_points_per_triangle,
    const int* cumsum_of_points_per_triangle,
    const float* colors,
    const float4* conic_opacity,
    const float* depths,
    const float2* phi_center,
    float* final_T,
    uint32_t* n_contrib,
    const float* bg_color,
    float* out_color,
    float* out_others,
    float* max_blending)
{
    renderCUDA<NUM_CHANNELS> << <grid, block >> > (
        ranges,
        point_list,
        W, H,
        normals,
        offsets,
        points_xy_image,
        sigma,
        num_points_per_triangle,
        cumsum_of_points_per_triangle,
        colors,
        conic_opacity,
        depths,
        phi_center,
        final_T,
        n_contrib,
        bg_color,
        out_color,
        out_others,
        max_blending
        );
}

void FORWARD::preprocess(int P, int D, int M,
    const float* triangles_points,
    const float* sigma,
    const int* num_points_per_triangle,
    const int* cumsum_of_points_per_triangle,
    const float* opacities,
    float* scaling,
    float* density_factor,
    const float* shs,
    bool* clamped,
    const float* colors_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    int* radii,
    float2* normals,
    float* offsets,
    float* p_w,
    float2* p_image,
    int* indices,
    float2* means2D,
    float* depths,
    float* rgb,
    float4* conic_opacity,
    float* cov3Ds,
    float2* phi_center,
    uint2* rect_min,
    uint2* rect_max,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered)
{
    preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
        P, D, M,
        triangles_points,
        sigma,
        num_points_per_triangle,
        cumsum_of_points_per_triangle,
        opacities,
        scaling,
        density_factor,
        shs,
        clamped,
        colors_precomp,
        viewmatrix, 
        projmatrix,
        cam_pos,
        W, H,
        tan_fovx, tan_fovy,
        focal_x, focal_y,
        radii,
        normals,
        offsets,
        p_w,
        p_image,
        indices,
        means2D,
        depths,
        rgb,
        conic_opacity,
        cov3Ds,
        phi_center,
        rect_min,
        rect_max,
        grid,
        tiles_touched,
        prefiltered
        );
}