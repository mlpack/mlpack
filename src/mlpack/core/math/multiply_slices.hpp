/**
 * @file core/math/multiply_slices.hpp
 * @author Mrityunjay Tripathi
 *
 * Function to perform matrix multiplication on cubes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_MATH_MULTIPLY_SLICES_HPP
#define MLPACK_CORE_MATH_MULTIPLY_SLICES_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Matrix multiplication of slices of two cubes. This function expects
 * both cubes to have the same number of slices. For example, a valid operation
 * would be: cube A of shape (m, p, s) multiplied by cube B of shape (p, n, s)
 * resulting in a cube of shape (m, n, s).
 *
 * @param cubeA First cube.
 * @param cubeB Second cube.
 * @param aTranspose Whether slices of first cube have to be transposed.
 * @param bTranspose Whether slices of second cube have to be transposed.
 */
template <typename CubeType>
CubeType MultiplyCube2Cube(const CubeType& cubeA,
                           const CubeType& cubeB,
                           const bool aTranspose = false,
                           const bool bTranspose = false);
/**
 * Matrix multiplication of a matrix and all the slices of a cube. This function
 * is used when the first object is a matrix and the second object is a cube.
 * For example, a valid operation would be: matrix A of shape (m, p)
 * multiplied by cube B of shape (p, n, s) resulting in a cube
 * of shape (m, n, s).
 *
 * @param matA The matrix as the first operand.
 * @param cubeB The cube as the second operand.
 * @param aTranspose Whether matrix has to be transposed.
 * @param bTranspose Whether slices of cube have to be transposed.
 */
template <typename MatType, typename CubeType>
CubeType MultiplyMat2Cube(const MatType& matA,
                          const CubeType& cubeB,
                          const bool aTranspose = false,
                          const bool bTranspose = false);
/**
 * Matrix multiplication of all slices of a cube with a matrix. This function
 * is used when the first object is a cube and the second object is a matrix.
 * For example, a valid operation would be: cube A of shape (m, p, s)
 * multiplied by a matrix of shape (p, n) resulting in a cube
 * of shape (m, n, s).
 *
 * @param cubeA The cube as the first operand.
 * @param matB The matrix as the second operand.
 * @param aTranspose Whether slices of cube have to be transposed.
 * @param bTranspose Whether matrix has to be transposed.
 */
template <typename CubeType, typename MatType>
CubeType MultiplyCube2Mat(const CubeType& cubeA,
                          const MatType& matB,
                          const bool aTranspose = false,
                          const bool bTranspose = false);

} // namespace mlpack

// Include implementation.
#include "multiply_slices_impl.hpp"

#endif
