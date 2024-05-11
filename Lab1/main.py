""" CS4277/CS5477 Lab 1: Metric Rectification and Robust Homography Estimation.
See accompanying file (lab1.pdf) for instructions.

Name: Nicholas Sun Jun Yang
Email: e0543645@u.nus.edu
Student ID: A0217609B
"""

import numpy as np
import cv2
from helper import *
from math import floor, ceil, sqrt



def compute_homography(src, dst):
    """Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    """

    h_matrix = np.eye(3, dtype=np.float64)

    """ YOUR CODE STARTS HERE """
    # Compute normalization matrix
    centroid_src = np.mean(src, axis=0)
    d_src = np.linalg.norm(src - centroid_src[None, :], axis=1)
    s_src = sqrt(2) / np.mean(d_src)
    T_norm_src = np.array([[s_src, 0.0, -s_src * centroid_src[0]],
                           [0.0, s_src, -s_src * centroid_src[1]],
                           [0.0, 0.0, 1.0]])

    centroid_dst = np.mean(dst, axis=0)
    d_dst = np.linalg.norm(dst - centroid_dst[None, :], axis=1)
    s_dst = sqrt(2) / np.mean(d_dst)
    T_norm_dst = np.array([[s_dst, 0.0, -s_dst * centroid_dst[0]],
                           [0.0, s_dst, -s_dst * centroid_dst[1]],
                           [0.0, 0.0, 1.0]])

    srcn = transform_homography(src, T_norm_src)
    dstn = transform_homography(dst, T_norm_dst)

    # Compute homography
    n_corr = srcn.shape[0]
    A = np.zeros((n_corr*2, 9), dtype=np.float64)
    for i in range(n_corr):
        A[2 * i, 0] = srcn[i, 0]
        A[2 * i, 1] = srcn[i, 1]
        A[2 * i, 2] = 1.0
        A[2 * i, 6] = -dstn[i, 0] * srcn[i, 0]
        A[2 * i, 7] = -dstn[i, 0] * srcn[i, 1]
        A[2 * i, 8] = -dstn[i, 0] * 1.0

        A[2 * i + 1, 3] = srcn[i, 0]
        A[2 * i + 1, 4] = srcn[i, 1]
        A[2 * i + 1, 5] = 1.0
        A[2 * i + 1, 6] = -dstn[i, 1] * srcn[i, 0]
        A[2 * i + 1, 7] = -dstn[i, 1] * srcn[i, 1]
        A[2 * i + 1, 8] = -dstn[i, 1] * 1.0

    u, s, vt = np.linalg.svd(A)
    h_matrix_n = np.reshape(vt[-1, :], (3, 3))

    # Unnormalize homography
    h_matrix = np.linalg.inv(T_norm_dst) @ h_matrix_n @ T_norm_src
    h_matrix /= h_matrix[2, 2]

    # src = src.astype(np.float32)
    # dst = dst.astype(np.float32)
    # h_matrix = cv2.findHomography(src, dst)[0].astype(np.float64)
    """ YOUR CODE ENDS HERE """

    return h_matrix


def transform_homography(src, h_matrix):
    """Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    Prohibited functions:
        cv2.perspectiveTransform()

    """
    transformed = None
    """ YOUR CODE STARTS HERE """
    # Add a column of ones to the source coordinates to make them homogeneous
    hom_src = np.column_stack((src, np.ones((len(src), 1), dtype=np.float32)))
    # Apply the homography matrix to the homogeneous source coordinates -> x' = Hx
    transformed = np.dot(h_matrix, hom_src.T).T
    # Convert the transformed homogeneous coordinates back to Cartesian coordinates
    transformed = transformed[:, :2] / transformed[:, 2:]
    """ YOUR CODE ENDS HERE """
    return transformed


def warp_image(src, dst, h_matrix):
    """Applies perspective transformation to source image to warp it onto the
    destination (background) image

    Args:
        src (np.ndarray): Source image to be warped
        dst (np.ndarray): Background image to warp template onto
        h_matrix (np.ndarray): Warps coordinates from src to the dst, i.e.
                                 x_{dst} = h_matrix * x_{src},
                               where x_{src}, x_{dst} are the homogeneous
                               coordinates in I_{src} and I_{dst} respectively

    Returns:
        dst (np.ndarray): Source image warped onto destination image

    Prohibited functions:
        cv2.warpPerspective()
    You may use the following functions: np.meshgrid(), cv2.remap(), transform_homography()
    """
    dst = dst.copy()  # deep copy to avoid overwriting the original image
    """ YOUR CODE STARTS HERE """
    h, w = dst.shape[:2]
    # Create meshgrid of coordinates for the destination image
    dst_x, dst_y = np.meshgrid(np.arange(w), np.arange(h),indexing='ij')
    # Coordinates = {{0,0},.....,{w,h}}
    dst_coords = np.column_stack((dst_x.flatten(), dst_y.flatten()))
    transformed = transform_homography(dst_coords, np.linalg.inv(h_matrix))
    map_x = transformed[:, 0].reshape(h, w, order="F").astype(np.float32)
    map_y = transformed[:, 1].reshape(h, w, order="F").astype(np.float32)
    dst = cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_TRANSPARENT, dst=dst)
    """ YOUR CODE ENDS HERE """
    # cv2.warpPerspective(src, h_matrix, dsize=dst.shape[1::-1],
    #                     dst=dst, borderMode=cv2.BORDER_TRANSPARENT)
    return dst

def compute_affine_rectification(src_img:np.ndarray,lines_vec: list):
    '''
       The first step of the stratification method for metric rectification. Compute
       the projective transformation matrix Hp with line at infinity. At least two
       parallel line pairs are required to obtain the vanishing line. Then warping
       the image with the predicted projective transformation Hp to recover the affine
       properties. X_dst=Hp*X_src

       Args:
           src_img: Original image X_src
           lines_vec: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0)
       Returns:
           Xa: Affinely rectified image by removing projective distortion

    '''
    dst = np.zeros_like(src_img)  # deep copy to avoid overwriting the original image
    #Hp= np.zeros((3,3))
    """ YOUR CODE STARTS HERE """
    #1. The imaged vanishing line of the plane l is computed from 
    #the intersection of two sets of imaged parallel lines
    pt1 = lines_vec[0].intersetion_point(lines_vec[1])
    pt2 = lines_vec[2].intersetion_point(lines_vec[3])
    l1,l2,l3 = Line_Equation(pt1,pt2)
    #Get inverse of the projection matrix, reference to lect 2 slides, pg 34
    hp_inv = np.linalg.inv(np.array([[1, 0, 0], [0, 1, 0], [-l1/l3, -l2/l3, 1/l3]]))
    # Since the line of infinity is fixed in an affine transfomration,
    # and applying hp_inv on the vanishing line maps it to the line of inifinity,
    # this allows the affine properties of a fixed line of inifinity to be recovered.
    h, w = dst.shape[:2]
    # Generate grid of coordinates using meshgrid
    x, y = np.meshgrid(range(w), range(h),indexing='ij')
    # Get mtarix of x,y coordinates, of size N x 2 where N = w * h
    mat = np.column_stack((x.flatten(), y.flatten()))
    #Apply the homography to remove the projective distortion
    transformed = transform_homography(mat, hp_inv)
    #get the dimensions of the affine transformed image
    w, h = np.max(transformed, axis=0)
    #prepare the destination matrix to be warped into using the dimensions of the affine transformed image
    dst = np.zeros((ceil(h), ceil(w), 3))
    #warp the image with the predicted projective transformation Hp to recover the affine properties
    dst = warp_image(src_img,dst, hp_inv)
    """ YOUR CODE ENDS HERE """
   
    return dst



def compute_metric_rectification_step2(src_img:np.ndarray,line_vecs: list):
    '''
       The second step of the stratification method for metric rectification. Compute
       the affine transformation Ha with the degenerate conic from at least two
       orthogonal line pairs. Then warping the image with the predicted affine
       transformation Ha to recover the metric properties. X_dst=Ha*X_src

       Args:
           src_img: Affinely rectified image X_src
           line_vecs: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0)
       Returns:
           X_dst: Image after metric rectification

    '''
    dst = np.zeros_like(src_img)  # deep copy to avoid overwriting the original image
    #Ha = np.zeros((3, 3))
    """ YOUR CODE STARTS HERE """
    cstrnts = []
    #Reference to lect 2 slides, pg 55
    for i in range(0, 4, 2):
        l1, l2, l3 = line_vecs[i].vec_para
        m1, m2, m3 = line_vecs[i+1].vec_para
        cstrnts.append([l1*m1, (l1*m2 + l2*m1), l2*m2])
    cstrnts = np.array(cstrnts)
    #Use SVD to find the solution to cstrnts * s = 0, right orthogonal vector corersponding to the least significant eigenvalue
    U,S,Vh = np.linalg.svd(cstrnts)
    S = Vh[-1, :]
    # Make S a 2x2 vector
    S = np.array([[S[0], S[1]], [S[1], S[2]]])
    #S = K*K.T
    K = np.linalg.cholesky(S).transpose()
    #normalise by terminant
    K = K / np.linalg.det(K)
    # Ha = K  0, where K = 2x2 matrix
    #      0  1
    ha_inv = np.linalg.inv(np.array([np.append(K[0, :], 0), np.append(K[1, :], 0), [0, 0, 1]]))
    h, w = src_img.shape[:2]
    # Create meshgrid of coordinates for the destination image
    x, y = np.meshgrid(np.arange(w), np.arange(h),indexing='ij')
    # Coordinates = {{0,0},.....,{w,h}}
    mat = np.column_stack((x.flatten(), y.flatten()))
    transformed = transform_homography(mat, ha_inv)
    w, h = np.max(transformed, axis=0)
    dst = np.zeros((ceil(h), ceil(w), 3))
    dst = warp_image(src_img, dst, ha_inv)
    """ YOUR CODE ENDS HERE """
    return dst

def compute_metric_rectification_one_step(src_img:np.ndarray,line_vecs: list):
    '''
       One-step metric rectification. Compute the transformation matrix H (i.e. H=HaHp) directly
       from five orthogonal line pairs. Then warping the image with the predicted affine
       transformation Ha to recover the metric properties. X_dst=H*X_src
       Args:
           src_img: Original image Xc
           line_infinity: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0)
       Returns:
           Xa: Image after metric rectification

    '''
    dst = np.zeros_like(src_img)  # deep copy to avoid overwriting the original image
    H = np.zeros((3, 3))
    """ YOUR CODE STARTS HERE """
    cstrnts = []
    #Reference to lect 2 slides  pg 57
    for i in range(0, 10, 2):
        l1, l2, l3 = line_vecs[i].vec_para
        m1, m2, m3 = line_vecs[i+1].vec_para
        cstrnts.append([l1*m1, (l1*m2 + l2*m1)/2, l2*m2,(l1*m3 + l3*m1)/2, (l2*m3 + l3*m2)/2, l3*m3])
    cstrnts = np.array(cstrnts)
    # solve strnts * c = 0 using svd to get c
    U,S,Vh = np.linalg.svd(cstrnts)
    #c = (a,b,c,d,e,f)
    a, b, c, d, e, f = Vh[-1, :]
    # Get conic equation
    C = np.array([[a, b/2, d/2], [b/2, c, e/2], [d/2, e/2, f]])
    # C'* = U *D* U.T
    #solve toget D
    CU, CS, CVh = np.linalg.svd(C)
    #ignore the last factor, wejust need the first two
    CS = np.append(CS[:-1], [1])
    # square root it to make it the same as lect 2 slide 50
    CS = np.sqrt(CS)
    D = np.array([[CS[0], 0, 0], [0, CS[1], 0], [0, 0, CS[2]]])
    # H = UD upto similarity sqrt S
    h_inv = np.linalg.inv(np.matmul(CU, D))
    h, w = src_img.shape[:-1]
    # Create meshgrid of coordinates for the destination image
    x, y = np.meshgrid(np.arange(w), np.arange(h),indexing='ij')
    # Coordinates = {{0,0},.....,{w,h}}
    mat = np.column_stack((x.flatten(), y.flatten()))
    transformed = transform_homography(mat, h_inv)
    # get dimensions for the similarity transformed image
    x_min, y_min = np.min(transformed, axis=0)
    x_max, y_max = np.max(transformed, axis=0)
    #get scale factor in both x and y directions
    SX = w/(x_max - x_min)
    SY = h/(y_max - y_min)
    theta = np.radians(0)
    #construct the similarity matrix bsaed on lect 1 slied 49 and 56
    sim_trans = np.array([[SX * np.cos(theta) , -SX * np.sin(theta), SX * abs(x_min)], [SY * np.sin(theta), SY * np.cos(theta), SY*abs(y_min)], [0, 0, 1]])
    #Combine all the homography
    H = np.matmul(sim_trans, h_inv)
    dst = warp_image(src_img, dst, H)
    """ YOUR CODE ENDS HERE """
    return dst

def compute_homography_error(src, dst, homography):
    """Compute the squared bidirectional pixel reprojection error for
    provided correspondences

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)
        homography (np.ndarray): Homography matrix that transforms src to dst.

    Returns:
        err (np.ndarray): Array of size (N, ) containing the error d for each
        correspondence, computed as:
          d(x,x') = ||x - inv(H)x'||^2 +  ||x' - Hx||^2,
        where ||a|| denotes the l2 norm (euclidean distance) of vector a.
    """
    d = np.zeros(src.shape[0], np.float64)
    """ YOUR CODE STARTS HERE """
    homo_src = np.hstack((src, np.ones((src.shape[0], 1))))
    homo_dst = np.hstack((dst, np.ones((dst.shape[0], 1))))
    # Transform src points to dst using homography
    trans_src = np.dot(homography, homo_src.T).T
    trans_src /= trans_src[:, 2][:, np.newaxis]
    # Compute the bidirectional pixel reprojection error
    err_fwd = np.linalg.norm(dst - trans_src[:, :2], axis=1)**2
    # Transform dst points to src using inverse homography
    inv_homo = np.linalg.inv(homography)
    trans_dst = np.dot(inv_homo, homo_dst.T).T
    trans_dst /= trans_dst[:, 2][:, np.newaxis]
    # Compute the bidirectional pixel reprojection error
    err_bwd = np.linalg.norm(src - trans_dst[:, :2], axis=1)**2
    # Total bidirectional error
    d = err_fwd + err_bwd
    """ YOUR CODE ENDS HERE """
    return d


def compute_homography_ransac(src, dst, thresh=16.0, num_tries=200):
    """Calculates the perspective transform from at least 4 points of
    corresponding points in a robust manner using RANSAC. After RANSAC, all the
    inlier correspondences will be used to re-estimate the homography matrix.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)
        thresh (float): Maximum allowed squared bidirectional pixel reprojection
          error to treat a point pair as an inlier (default: 16.0). Pixel
          reprojection error is computed as:
            d(x,x') = ||x - inv(H)x'||^2 +  ||x' - Hx||^2,
          where ||a|| denotes the l2 norm (euclidean distance) of vector a.
        num_tries (int): Number of trials for RANSAC

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.
        mask (np.ndarraay): Output mask with dtype np.bool where 1 indicates
          inliers

    Prohibited functions:
        cv2.findHomography()
    """
    h_matrix = np.eye(3, dtype=np.float64)
    mask = np.ones(src.shape[0], dtype=np.bool)
    """ YOUR CODE STARTS HERE """
    max_inliers = 0
    for _ in range(num_tries):
        # Randomly sample 4 point correspondences
        rand_inds = np.random.choice(src.shape[0], 4, replace=False)
        src_samp = src[rand_inds]
        dst_samp = dst[rand_inds]
        # Compute the homography matrix using the sampled correspondences
        curr_h_mat = compute_homography(src_samp, dst_samp)
        # Compute bidirectional pixel reprojection error for all points
        errs = compute_homography_error(src, dst, curr_h_mat)
        # Identify inliers based on the reprojection error threshold
        curr_mask = errs < thresh
        # Update the best homography matrix if the current set has more inliers
        curr_num_inliers = np.sum(curr_mask)
        if curr_num_inliers > max_inliers:
            max_inliers = curr_num_inliers
            h_matrix = curr_h_mat
            mask = curr_mask
    """ YOUR CODE ENDS HERE """
    return h_matrix, mask


