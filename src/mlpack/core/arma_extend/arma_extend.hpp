/***
 * @file core/arma_extend/arma_extend.hpp
 * @author Ryan Curtin
 *
 * Include Armadillo extensions which currently are not part of the main
 * Armadillo codebase.
 *
 */
#ifndef MLPACK_CORE_ARMA_EXTEND_ARMA_EXTEND_HPP
#define MLPACK_CORE_ARMA_EXTEND_ARMA_EXTEND_HPP

// Add batch constructor for sparse matrix (if version <= 3.810.0).
#define ARMA_EXTRA_SPMAT_PROTO mlpack/core/arma_extend/SpMat_extra_bones.hpp
#define ARMA_EXTRA_SPMAT_MEAT  mlpack/core/arma_extend/SpMat_extra_meat.hpp

// Add row_col_iterator and row_col_const_iterator for Mat.
#define ARMA_EXTRA_MAT_PROTO mlpack/core/arma_extend/Mat_extra_bones.hpp
#define ARMA_EXTRA_MAT_MEAT mlpack/core/arma_extend/Mat_extra_meat.hpp

// Add boost serialization for Cube.
#define ARMA_EXTRA_CUBE_PROTO mlpack/core/arma_extend/Cube_extra_bones.hpp
#define ARMA_EXTRA_CUBE_MEAT mlpack/core/arma_extend/Cube_extra_meat.hpp

// Manually set ARMA_{64,32}BIT_WORD for _WIN64 or win32
#if defined(_MSC_VER)
    #ifdef _WIN64
        #define ARMA_64BIT_WORD
        #ifdef ARMA_32BIT_WORD
            #undef ARMA_32BIT_WORD
        #endif
    #else
        #define ARMA_32BIT_WORD
        #ifdef ARMA_64BIT_WORD
            #undef ARMA_64BIT_WORD
        #endif
    #endif
#endif

// Force definition of old HDF5 API.  Thanks to Mike Roberts for helping find
// this workaround.
#if !defined(H5_USE_110_API)
  #define H5_USE_110_API
#endif

// Include everything we'll need for serialize().
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/array.hpp>

#include <armadillo>

#endif
