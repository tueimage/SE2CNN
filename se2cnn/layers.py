# -*- coding: utf-8 -*-
"""
se2cnn/layers.py

Implementation of tensorflow layers for operations in SE2N.
Details in MICCAI 2018 paper: "Roto-Translation Covariant Convolutional Networks for Medical Image Analysis".

Released in June 2018
@author: EJ Bekkers, Eindhoven University of Technology, The Netherlands
@author: MW Lafarge, Eindhoven University of Technology, The Netherlands
________________________________________________________________________

Copyright 2018 Erik J Bekkers and Maxime W Lafarge, Eindhoven University 
of Technology, the Netherlands

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
________________________________________________________________________
"""

import tensorflow as tf
import numpy as np
from . import rotation_matrix


# THE CONVOLUTION LAYERS

def z2_se2n(
        input_tensor,
        kernel,
        orientations_nb,

        # Optional:
        periodicity=2 * np.pi,
        diskMask=True,
        padding='VALID'):
    """ Constructs a group convolutional layer.
        (lifting layer from Z2 to SE2N with N input number of orientations)

        INPUT:
            - input_tensor in Z2, a tensorflow Tensor with expected shape:
                [BatchSize, Height, Width, ChannelsIN]
            - kernel, a tensorflow Tensor with expected shape:
                [kernelSize, kernelSize, ChannelsIN, ChannelsOUT]
            - orientations_nb, an integer specifying the number of rotations

        INPUT (optional):
            - periodicity, rotate in total over 2*np.pi or np.pi
            - disk_mask, True or False, specifying whether or not to mask the kernels spatially

        OUTPUT:
            - output_tensor, the tensor after group convolutions with shape
                [BatchSize, Height', Width', orientations_nb, ChannelsOut]
                (Height', Width' are reduced sizes due to the valid convolution)
            - kernels_formatted, the formated kernels, i.e., the full stack of rotated kernels with shape:
                [orientations_nb, kernelSize, kernelSize, ChannelsIN, ChannelsOUT]
    """

    # Preparation for group convolutions
    # Precompute a rotated stack of kernels
    kernel_stack = rotate_lifting_kernels(
        kernel, orientations_nb, periodicity=periodicity, diskMask=diskMask)
    print("Z2-SE2N ROTATED KERNEL SET SHAPE:",
          kernel_stack.get_shape())  # Debug

    # Format the kernel stack as a 2D kernel stack (merging the rotation and
    # channelsOUT axis)
    kernels_as_if_2D = tf.transpose(kernel_stack, [1, 2, 3, 0, 4])
    kernelSizeH, kernelSizeW, channelsIN, channelsOUT = map(int, kernel.shape)
    kernels_as_if_2D = tf.reshape(
        kernels_as_if_2D, [kernelSizeH, kernelSizeW, channelsIN, orientations_nb * channelsOUT])

    # Perform the 2D convolution
    layer_output = tf.nn.conv2d(
        input=input_tensor,
        filter=kernels_as_if_2D,
        strides=[1, 1, 1, 1],
        padding=padding)

    # Reshape to an SE2 image (split the orientation and channelsOUT axis)
    # Note: the batch size is unknown, hence this dimension needs to be
    # obtained using the tensorflow function tf.shape, for the other
    # dimensions we keep using tensor.shape since this allows us to keep track
    # of the actual shapes (otherwise the shapes get convert to
    # "Dimensions(None)").
    layer_output = tf.reshape(
        layer_output, [tf.shape(layer_output)[0], int(layer_output.shape[1]), int(layer_output.shape[2]), orientations_nb, channelsOUT])
    print("OUTPUT SE2N ACTIVATIONS SHAPE:", layer_output.get_shape())  # Debug

    return layer_output, kernel_stack


def se2n_se2n(
        input_tensor,
        kernel,

        # Optional:
        periodicity=2 * np.pi,
        diskMask=True,
        padding='VALID'):
    """ Constructs a group convolutional layer.
        (group convolution layer from SE2N to SE2N with N input number of orientations)
        INPUT:
            - input_tensor in SE2n, a tensor flow tensor with expected shape:
                [BatchSize, nbOrientations, Height, Width, ChannelsIN]
            - kernel, a tensorflow Tensor with expected shape:
                [kernelSize, kernelSize, nbOrientations, ChannelsIN, ChannelsOUT]

        INPUT (optional):
            - periodicity, rotate in total over 2*np.pi or np.pi
            - disk_mask, True or False, specifying whether or not to mask the
                kernels spatially

        OUTPUT:
            - output_tensor, the tensor after group convolutions with shape
                [BatchSize, Height', Width', nbOrientations, ChannelsOut]
                (Height', Width' are the reduced sizes due to the valid convolution)
            - kernels_formatted, the formated kernels, i.e., the full stack of
                rotated kernels with shape [nbOrientations, kernelSize, kernelSize, nbOrientations, channelsIn, channelsOut]
    """

    # Kernel dimensions
    kernelSizeH, kernelSizeW, orientations_nb, channelsIN, channelsOUT = map(
        int, kernel.shape)

    # Preparation for group convolutions
    # Precompute a rotated stack of se2 kernels
    # With shape: [orientations_nb, kernelSizeH, kernelSizeW, orientations_nb,
    # channelsIN, channelsOUT]
    kernel_stack = rotate_gconv_kernels(kernel, periodicity, diskMask)
    print("SE2N-SE2N ROTATED KERNEL SET SHAPE:",
          kernel_stack.get_shape())  # Debug

    # Group convolutions are done by integrating over [x,y,theta,input-channels] for each translation and rotation of the kernel
    # We compute this integral by doing standard 2D convolutions (translation part) for each rotated version of the kernel (rotation part)
    # In order to efficiently do this we use 2D convolutions where the theta
    # and input-channel axes are merged (thus treating the SE2 image as a 2D
    # feature map)

    # Prepare the input tensor (merge the orientation and channel axis) for
    # the 2D convolutions:
    input_tensor_as_if_2D = tf.reshape(
        input_tensor, [tf.shape(input_tensor)[0], int(input_tensor.shape[1]), int(input_tensor.shape[2]), orientations_nb * channelsIN])

    # Reshape the kernels for 2D convolutions (orientation+channelsIN axis are
    # merged, rotation+channelsOUT axis are merged)
    kernels_as_if_2D = tf.transpose(kernel_stack, [1, 2, 3, 4, 0, 5])
    kernels_as_if_2D = tf.reshape(
        kernels_as_if_2D, [kernelSizeH, kernelSizeW, orientations_nb * channelsIN, orientations_nb * channelsOUT])

    # Perform the 2D convolutions
    layer_output = tf.nn.conv2d(
        input=input_tensor_as_if_2D,
        filter=kernels_as_if_2D,
        strides=[1, 1, 1, 1],
        padding=padding)

    # Reshape into an SE2 image (split the orientation and channelsOUT axis)
    layer_output = tf.reshape(
        layer_output, [tf.shape(layer_output)[0], int(layer_output.shape[1]), int(layer_output.shape[2]), orientations_nb, channelsOUT])
    print("OUTPUT SE2N ACTIVATIONS SHAPE:", layer_output.get_shape())  # Debug

    return layer_output, kernel_stack


# THE MAX-POOLING LAYER

def spatial_max_pool(input_tensor, nbOrientations, padding='VALID'):
    """ Performs spatial max-pooling on every orientation of the SE2N tensor.
        INPUT:
            - input_tensor in SE2n, a tensor flow tensor with expected shape:
                [BatchSize, Height, Width, nbOrientations, ChannelsIN]

        OUTPUT:
            - output_tensor, the tensor after spatial max-pooling
                [BatchSize, Height/2, Width/2, nbOrientations, ChannelsOut]
    """

    # 2D max-pooling is applied to each orientation
    activations = [None] * nbOrientations
    for i in range(nbOrientations):
        activations[i] = tf.nn.max_pool(
            value=input_tensor[:, :, :, i, :],
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding=padding)

    # Re-stack all the pooled activations along the orientation dimension
    tensor_pooled = tf.concat(
        values=[tf.expand_dims(t, 3) for t in activations], axis=3)

    return tensor_pooled


# KERNEL ROTATION FUNCTIONS

def rotate_lifting_kernels(kernel, orientations_nb, periodicity=2 * np.pi, diskMask=True):
    """ Rotates the set of 2D lifting kernels.

        INPUT:
            - kernel, a tensor flow tensor with expected shape:
                [Height, Width, ChannelsIN, ChannelsOUT]
            - orientations_nb, an integer specifying the number of rotations

        INPUT (optional):
            - periodicity, rotate in total over 2*np.pi or np.pi
            - disk_mask, True or False, specifying whether or not to mask the kernels spatially

        OUTPUT:
            - set_of_rotated_kernels, a tensorflow tensor with dimensions:
                [nbOrientations, Height, Width, ChannelsIN, ChannelsOUT]
    """

    # Unpack the shape of the input kernel
    kernelSizeH, kernelSizeW, channelsIN, channelsOUT = map(int, kernel.shape)
    print("Z2-SE2N BASE KERNEL SHAPE:", kernel.get_shape())  # Debug

    # Flatten the baseline kernel
    # Resulting shape: [kernelSizeH*kernelSizeW, channelsIN*channelsOUT]
    kernel_flat = tf.reshape(
        kernel, [kernelSizeH * kernelSizeW, channelsIN * channelsOUT])

    # Generate a set of rotated kernels via rotation matrix multiplication
    # For efficiency purpose, the rotation matrix is implemented as a sparse matrix object
    # Result: The non-zero indices and weights of the rotation matrix
    idx, vals = rotation_matrix.MultiRotationOperatorMatrixSparse(
        [kernelSizeH, kernelSizeW],
        orientations_nb,
        periodicity=periodicity,
        diskMask=diskMask)

    # Sparse rotation matrix
    # Resulting shape: [nbOrientations*kernelSizeH*kernelSizeW,
    # kernelSizeH*kernelSizeW]
    rotOp_matrix = tf.SparseTensor(
        idx, vals,
        [orientations_nb * kernelSizeH * kernelSizeW, kernelSizeH * kernelSizeW])

    # Matrix multiplication
    # Resulting shape: [nbOrientations*kernelSizeH*kernelSizeW,
    # channelsIN*channelsOUT]
    set_of_rotated_kernels = tf.sparse_tensor_dense_matmul(
        rotOp_matrix, kernel_flat)

    # Reshaping
    # Resulting shape: [nbOrientations, kernelSizeH, kernelSizeW, channelsIN,
    # channelsOUT]
    set_of_rotated_kernels = tf.reshape(
        set_of_rotated_kernels, [orientations_nb, kernelSizeH, kernelSizeW, channelsIN, channelsOUT])

    return set_of_rotated_kernels


def rotate_gconv_kernels(kernel, periodicity=2 * np.pi, diskMask=True):
    """ Rotates the set of SE2 kernels. 
        Rotation of SE2 kernels involves planar rotations and a shift in orientation,
        see e.g. the left-regular representation L_g of the roto-translation group on SE(2) images,
        (Eq. 3) of the MICCAI 2018 paper.

        INPUT:
            - kernel, a tensor flow tensor with expected shape:
                [Height, Width, nbOrientations, ChannelsIN, ChannelsOUT]

        INPUT (optional):
            - periodicity, rotate in total over 2*np.pi or np.pi
            - disk_mask, True or False, specifying whether or not to mask the kernels spatially

        OUTPUT:
            - set_of_rotated_kernels, a tensorflow tensor with dimensions:
                [nbOrientations, Height, Width, nbOrientations, ChannelsIN, ChannelsOUT]
              I.e., for each rotation angle a rotated (shift-twisted) version of the input kernel.
    """

    # Rotation of an SE2 kernel consists of two parts:
    # PART 1. Planar rotation
    # PART 2. A shift in theta direction

    # Unpack the shape of the input kernel
    kernelSizeH, kernelSizeW, orientations_nb, channelsIN, channelsOUT = map(
        int, kernel.shape)
    print("SE2N-SE2N BASE KERNEL SHAPE:", kernel.get_shape())  # Debug

    # PART 1 (planar rotation)
    # Flatten the baseline kernel
    # Resulting shape: [kernelSizeH*kernelSizeW,orientations_nb*channelsIN*channelsOUT]
    #
    kernel_flat = tf.reshape(
        kernel, [kernelSizeH * kernelSizeW, orientations_nb * channelsIN * channelsOUT])

    # Generate a set of rotated kernels via rotation matrix multiplication
    # For efficiency purpose, the rotation matrix is implemented as a sparse matrix object
    # Result: The non-zero indices and weights of the rotation matrix
    idx, vals = rotation_matrix.MultiRotationOperatorMatrixSparse(
        [kernelSizeH, kernelSizeW],
        orientations_nb,
        periodicity=periodicity,
        diskMask=diskMask)

    # The corresponding sparse rotation matrix
    # Resulting shape: [nbOrientations*kernelSizeH*kernelSizeW,kernelSizeH*kernelSizeW]
    #
    rotOp_matrix = tf.SparseTensor(
        idx, vals,
        [orientations_nb * kernelSizeH * kernelSizeW, kernelSizeH * kernelSizeW])

    # Matrix multiplication (each 2D plane is now rotated)
    # Resulting shape: [nbOrientations*kernelSizeH*kernelSizeW, orientations_nb*channelsIN*channelsOUT]
    #
    kernels_planar_rotated = tf.sparse_tensor_dense_matmul(
        rotOp_matrix, kernel_flat)
    kernels_planar_rotated = tf.reshape(
        kernels_planar_rotated, [orientations_nb, kernelSizeH, kernelSizeW, orientations_nb, channelsIN, channelsOUT])

    # PART 2 (shift in theta direction)
    set_of_rotated_kernels = [None] * orientations_nb
    for orientation in range(orientations_nb):
        # [kernelSizeH,kernelSizeW,orientations_nb,channelsIN,channelsOUT]
        kernels_temp = kernels_planar_rotated[orientation]
        # [kernelSizeH,kernelSizeW,channelsIN,channelsOUT,orientations_nb]
        kernels_temp = tf.transpose(kernels_temp, [0, 1, 3, 4, 2])
        # [kernelSizeH*kernelSizeW*channelsIN*channelsOUT*orientations_nb]
        kernels_temp = tf.reshape(
            kernels_temp, [kernelSizeH * kernelSizeW * channelsIN * channelsOUT, orientations_nb])
        # Roll along the orientation axis
        roll_matrix = tf.constant(
            np.roll(np.identity(orientations_nb), orientation, axis=1), dtype=tf.float32)
        kernels_temp = tf.matmul(kernels_temp, roll_matrix)
        kernels_temp = tf.reshape(
            kernels_temp, [kernelSizeH, kernelSizeW, channelsIN, channelsOUT, orientations_nb])  # [Nx,Ny,Nin,Nout,Ntheta]
        kernels_temp = tf.transpose(kernels_temp, [0, 1, 4, 2, 3])
        set_of_rotated_kernels[orientation] = kernels_temp

    return tf.stack(set_of_rotated_kernels)


if __name__ == "__main__":
    help(z2_se2n)
    help(se2n_se2n)
    help(spatial_max_pool)
    help(rotate_lifting_kernels)
    help(rotate_gconv_kernels)
