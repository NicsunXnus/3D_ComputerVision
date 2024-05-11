""" CS4277/CS5477 Lab 4: Plane Sweep Stereo
See accompanying Jupyter notebook (lab4.ipynb) for instructions.

Name: Nicholas Sun Jun Yang
Email: e0543645@u.nus.edu
Student ID: A0217609B

"""
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import scipy.ndimage

"""Helper functions: You should not have to touch the following functions.
"""
class Image(object):
    """
    Image class. You might find the following member variables useful:
    - image: RGB image (HxWx3) of dtype np.float64
    - pose_mat: 3x4 Camera extrinsics that transforms points from world to
        camera frame
    """
    def __init__(self, qvec, tvec, name, root_folder=''):
        self.qvec = qvec
        self.tvec = tvec
        self.name = name  # image filename
        self._image = self.load_image(os.path.join(root_folder, name))

        # Extrinsic matrix: Transforms from world to camera frame
        self.pose_mat = self.make_extrinsic(qvec, tvec)

    def __repr__(self):
        return '{}: qvec={}\n tvec={}'.format(
            self.name, self.qvec, self.tvec
        )

    @property
    def image(self):
        return self._image.copy()

    @staticmethod
    def load_image(path):
        """Loads image and converts it to float64"""
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im.astype(np.float64) / 255.0

    @staticmethod
    def make_extrinsic(qvec, tvec):
        """ Make 3x4 camera extrinsic matrix from colmap pose

        Args:
            qvec: Quaternion as per colmap format (q_cv) in the order
                  q_w, q_x, q_y, q_z
            tvec: translation as per colmap format (t_cv)

        Returns:

        """
        rotation = Rotation.from_quat(np.roll(qvec, -1))
        return np.concatenate([rotation.as_matrix(), tvec[:, None]], axis=1)

def write_json(outfile, images, intrinsic_matrix, img_hw):
    """Write metadata to json file.

    Args:
        outfile (str): File to write to
        images (list): List of Images
        intrinsic_matrix (np.ndarray): 3x3 intrinsic matrix
        img_hw (tuple): (image height, image width)
    """

    img_height, img_width = img_hw

    images_meta = []
    for im in images:
        images_meta.append({
            'name': im.name,
            'qvec': im.qvec.tolist(),
            'tvec': im.tvec.tolist(),
        })

    data = {
        'img_height': img_height,
        'img_width': img_width,
        'K': intrinsic_matrix.tolist(),
        'images': images_meta
    }
    with open(outfile, 'w') as fid:
        json.dump(data, fid, indent=2)


def load_data(root_folder):
    """Loads dataset.

    Args:
        root_folder (str): Path to data folder. Should contain metadata.json

    Returns:
        images, K, img_hw
    """
    print('Loading data from {}...'.format(root_folder))
    with open(os.path.join(root_folder, 'metadata.json')) as fid:
        metadata = json.load(fid)

    images = []
    for im in metadata['images']:
        images.append(Image(np.array(im['qvec']), np.array(im['tvec']),
                            im['name'], root_folder=root_folder))
    img_hw = (metadata['img_height'], metadata['img_width'])
    K = np.array(metadata['K'])

    print('Loaded data containing {} images.'.format(len(images)))
    return images, K, img_hw


def invert_extrinsic(cam_matrix):
    """Invert extrinsic matrix"""
    irot_mat = cam_matrix[:3, :3].transpose()
    trans_vec = cam_matrix[:3, 3, None]

    inverted = np.concatenate([irot_mat,  -irot_mat @ trans_vec], axis=1)
    return inverted


def concat_extrinsic_matrix(mat1, mat2):
    """Concatenate two 3x4 extrinsic matrices, i.e. result = mat1 @ mat2
      (ignoring matrix dimensions)
    """
    r1, t1 = mat1[:3, :3], mat1[:3, 3:]
    r2, t2 = mat2[:3, :3], mat2[:3, 3:]
    rot = r1 @ r2
    trans = r1@t2 + t1
    concatenated = np.concatenate([rot, trans], axis=1)
    return concatenated


def rgb2hex(rgb):
    """Converts color representation into hexadecimal representation for K3D

    Args:
        rgb (np.ndarray): (N, 3) array holding colors

    Returns:
        hex (np.ndarray): array (N, ) of size N, each element indicates the
          color, e.g. 0x0000FF = blue
    """
    rgb_uint = (rgb * 255).astype(np.uint8)
    hex = np.sum(rgb_uint * np.array([[256 ** 2, 256, 1]]),
                 axis=1).astype(np.uint32)
    return hex

"""Functions to be implemented
"""
# Part 1
def get_plane_sweep_homographies(K, relative_pose, inv_depths):
    """Compute plane sweep homographies, assuming fronto parallel planes w.r.t.
    reference camera

    Args:
        K (np.ndarray): Camera intrinsic matrix (3,3)
        relative_pose (np.ndarray): Relative pose between the two cameras
          of shape (3, 4)
        inv_depths (np.ndarray): Inverse depths to warp of size (D, )

    Returns:
        homographies (D, 3, 3)
    """
    #Reference to Lecture 11 Page 68
    homographies = []
    """ YOUR CODE STARTS HERE """
    R_k = (relative_pose[:, :-1]).reshape((3, 3))
    C_k = relative_pose[:, -1].reshape((3, 1))
    n_m = np.array([0, 0, 1]).reshape((3, 1))
    K_ref_inv = np.linalg.inv(K)
    for i,d_m_inv in enumerate(inv_depths):
        homographies.append(np.dot(K, np.dot(R_k + d_m_inv * np.dot(C_k, n_m.T)  , K_ref_inv)))
    """ YOUR CODE ENDS HERE """

    return np.array(homographies)

#Part 2
def compute_plane_sweep_volume(images, ref_pose, K, inv_depths, img_hw):
    """Compute plane sweep volume, by warping all images to the reference camera
    fronto-parallel planes, before computing the variance for each pixel and
    depth.

    Args:
        images (list[Image]): List of images which contains information about
          the camera extrinsics for each image
        ref_pose (np.ndarray): Reference camera pose
        K (np.ndarray): 3x3 intrinsic matrix (assumed same for all cameras)
        inv_depths (list): List of inverse depths to consider for plane sweep
        img_hw (tuple): tuple containing (H, W), which are the output height
          and width for the plane sweep volume.

    Returns:
        ps_volume (np.ndarray):
          Plane sweep volume of size (D, H, W), with dtype=np.float64, where
          D is len(inv_depths), and (H, W) are the image heights and width
          respectively. Each element should contain the variance of all pixel
          intensities that warp onto it.
        accum_count (np.ndarray):
          Accumulator count of same size as ps_volume, and dtype=np.int32.
          Keeps track of how many images are warped into a certain pixel,
          i.e. the number of pixels used to compute the variance.
    """

    D = len(inv_depths)
    H, W = img_hw
    ps_volume = np.zeros((D, H, W), dtype=np.float64)
    accum_count = np.zeros((D, H, W), dtype=np.int32)

    """ YOUR CODE STARTS HERE """
    ref_id = 4
    ref_img = images[ref_id].image
    relative_poses = []
    #w.r.t ref image, calculate the relative pose of each image
    for image in images:
        relative_pose = concat_extrinsic_matrix(ref_pose, invert_extrinsic(image.pose_mat))
        relative_poses.append(relative_pose)
    for i in range(D):
        #w.r.t ref image, calculate the homography of each image
        #then warp each image to ref frame
        for j, image in enumerate(images):
            homo = get_plane_sweep_homographies(K, relative_poses[j], [inv_depths[i]])[0]
            #pixels that are outside the bounds of the input image after the transformation are filled with the value -1
            warp_image = cv2.warpPerspective(src=image.image, M=homo,dsize=None,borderValue=-1)
            #get total number of pixels mapped successfully
            mask = (warp_image == -1).any(axis=2)
            accum_count[i] += ~mask.astype(int)
            #computes the absolute pixel-wise intensity differences between the reference image and the warped image
            diff = np.abs(ref_img - warp_image)
            #computes the average intensity difference across all color channels for each pixel
            var = np.average(diff, axis=2)
            #ignore the pixels that were not mapped
            var[mask] = 0
            ps_volume[i] = ps_volume[i] + var
    """ YOUR CODE ENDS HERE """
    print(ps_volume.shape)


    return ps_volume, accum_count

def compute_depths(ps_volume, inv_depths):
    """Computes inverse depth map from plane sweep volume as the
    argmin over plane sweep volume variances.

    Args:
        ps_volume (np.ndarray): Plane sweep volume of size (D, H, W) from
          compute_plane_sweep_volume()
        inv_depths (np.ndarray): List of depths considered in the plane
          sweeping (D,)

    Returns:
        inv_depth_image (np.ndarray): inverse-depth estimate (H, W)
    """

    inv_depth_image = np.zeros(ps_volume.shape[1:], dtype=np.float64)

    """ YOUR CODE STARTS HERE """
    _, height, width = ps_volume.shape

    for w in range(width):
        for h in range(height):
            volumes_variances = ps_volume[:,h,w]
            min_var_index = np.argmin(volumes_variances)
            inv_depth_image[h,w] =  inv_depths[min_var_index]
    """ YOUR CODE ENDS HERE """

    return inv_depth_image

# Part 3
def unproject_depth_map(image, inv_depth_image, K, mask=None):
    """Converts the depth map into points by unprojecting depth map into 3D

    Note: You will also need to implement the case where no mask is provided

    Args:
        image (np.ndarray): Image bitmap (H, W, 3)
        inv_depth_image (np.ndarray): Inverse depth image (H, W)
        K (np.ndarray): 3x3 Camera intrinsics
        mask (np.ndarray): Optional mask of size (H, W) and dtype=np.bool.

    Returns:
        xyz (np.ndarray): Nx3 coordinates of points, dtype=np.float64.
        rgb (np.ndarray): Nx3 RGB colors, where rgb[i, :] is the (Red,Green,Blue)
          colors for the points at position xyz[i, :]. Should be in the range
          [0, 1] and have dtype=np.float64.
    """

    xyz = np.zeros([0, 3], dtype=np.float64)
    rgb = np.zeros([0, 3], dtype=np.float64)  # values should be within (0, 1)
    H, W = image.shape[0:2]
    """ YOUR CODE STARTS HERE """
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    f_x = K[0, 0]
    c_x = K[0, 2]
    f_y = K[1, 1]
    c_y = K[1, 2]
    #center x coords by principal point 
    x = x - c_x
    #scales x coords by depth to convert to 3D point
    x /= inv_depth_image
    #scale by focal length to convert to world frame
    x /= f_x
    #same for y coords
    y = (y - c_y) / inv_depth_image / f_y
    
    #Case when mask is not provided
    if mask is None:
        inv_depth_image = 1/inv_depth_image
    else:
        inv_depth_image = 1/inv_depth_image * mask
        x = x * mask
        y = y * mask

    points3d = np.dstack((x, y, inv_depth_image))
    points3d = points3d.reshape(-1, 3)
    pointsrgb = image.reshape(-1, 3)
    """ YOUR CODE ENDS HERE """

    xyz = np.array(points3d)
    rgb = np.array(pointsrgb)
    return xyz, rgb

# Part 4
def post_process(ps_volume, inv_depths, accum_count):
    """Post processes the plane sweep volume and compute a mask to indicate
    which pixels have confident estimates of the depth

    Args:
        ps_volume: Plane sweep volume from compute_plane_sweep_volume()
          of size (D, H, W)
        inv_depths (List[float]): List of depths considered in the plane
          sweeping
        accum_count: Accumulator count from compute_plane_sweep_volume(), which
          can be used to indicate which pixels are not observed by many other
          images.

    Returns:
        inv_depth_image: Denoised Inverse depth image (similar to compute_depths)
        mask: np.ndarray of size (H, W) and dtype np.bool.
          Pixels with values TRUE indicate valid pixels.
    """

    mask = np.ones(ps_volume.shape[1:], dtype=np.bool)
    inv_depth_image = np.zeros(ps_volume.shape[1:], dtype=np.float64)
    #print(accum_count)
    """ YOUR CODE STARTS HERE """
    inv_depth_image = compute_depths(ps_volume, inv_depths)
    mask = inv_depth_image <= np.mean(inv_depth_image) + (2.5 * np.std(inv_depth_image))
    inv_depth_image = scipy.ndimage.gaussian_filter(inv_depth_image, 2)
    """ YOUR CODE ENDS HERE """

    return inv_depth_image, mask
