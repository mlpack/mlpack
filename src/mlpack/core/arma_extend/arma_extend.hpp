/***
 * @file core/arma_extend/arma_extend.hpp
 * @author Ryan Curtin
 *
 * Include Armadillo serialization function which currently are 
 * not part of the main Armadillo codebase.
 *
 */
#ifndef MLPACK_CORE_ARMA_EXTEND_ARMA_EXTEND_HPP
#define MLPACK_CORE_ARMA_EXTEND_ARMA_EXTEND_HPP

// Force definition of old HDF5 API.  Thanks to Mike Roberts for helping find
// this workaround.
#if !defined(H5_USE_110_API)
  #define H5_USE_110_API
#endif

// Include everything we'll need for serialize().
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>

#include <armadillo>

#include <mlpack/core/arma_extend/SpMat_extra_bones.hpp>
#include <mlpack/core/arma_extend/SpMat_extra_meat.hpp>
#include <mlpack/core/arma_extend/Mat_extra_bones.hpp>
#include <mlpack/core/arma_extend/Mat_extra_meat.hpp>
#include <mlpack/core/arma_extend/Cube_extra_bones.hpp>
#include <mlpack/core/arma_extend/Cube_extra_meat.hpp>

#endif
