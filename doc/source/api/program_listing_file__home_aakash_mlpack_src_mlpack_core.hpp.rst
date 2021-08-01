
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core.hpp:

Program Listing for File core.hpp
=================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core.hpp>` (``/home/aakash/mlpack/src/mlpack/core.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_HPP
   #define MLPACK_CORE_HPP
   
   // First, include all of the prerequisites.
   #include <mlpack/prereqs.hpp>
   
   // Now the core mlpack classes.
   #include <mlpack/core/util/arma_traits.hpp>
   #include <mlpack/core/util/log.hpp>
   #include <mlpack/core/util/io.hpp>
   #include <mlpack/core/util/deprecated.hpp>
   #include <mlpack/core/data/load.hpp>
   #include <mlpack/core/data/save.hpp>
   #include <mlpack/core/data/normalize_labels.hpp>
   #include <mlpack/core/math/clamp.hpp>
   #include <mlpack/core/math/random.hpp>
   #include <mlpack/core/math/random_basis.hpp>
   #include <mlpack/core/math/lin_alg.hpp>
   #include <mlpack/core/math/range.hpp>
   #include <mlpack/core/math/round.hpp>
   #include <mlpack/core/math/shuffle_data.hpp>
   #include <mlpack/core/math/ccov.hpp>
   #include <mlpack/core/math/make_alias.hpp>
   #include <mlpack/core/dists/discrete_distribution.hpp>
   #include <mlpack/core/dists/gaussian_distribution.hpp>
   #include <mlpack/core/dists/laplace_distribution.hpp>
   #include <mlpack/core/dists/gamma_distribution.hpp>
   #include <mlpack/core/dists/diagonal_gaussian_distribution.hpp>
   #include <mlpack/core/data/confusion_matrix.hpp>
   #include <mlpack/core/data/one_hot_encoding.hpp>
   
   // mlpack::backtrace only for linux
   #ifdef HAS_BFD_DL
     #include <mlpack/core/util/backtrace.hpp>
   #endif
   
   // Include kernel traits.
   #include <mlpack/core/kernels/kernel_traits.hpp>
   #include <mlpack/core/kernels/linear_kernel.hpp>
   #include <mlpack/core/kernels/polynomial_kernel.hpp>
   #include <mlpack/core/kernels/cosine_distance.hpp>
   #include <mlpack/core/kernels/gaussian_kernel.hpp>
   #include <mlpack/core/kernels/epanechnikov_kernel.hpp>
   #include <mlpack/core/kernels/hyperbolic_tangent_kernel.hpp>
   #include <mlpack/core/kernels/laplacian_kernel.hpp>
   #include <mlpack/core/kernels/pspectrum_string_kernel.hpp>
   #include <mlpack/core/kernels/spherical_kernel.hpp>
   #include <mlpack/core/kernels/triangular_kernel.hpp>
   #include <mlpack/core/kernels/cauchy_kernel.hpp>
   
   // Use OpenMP if compiled with -DHAS_OPENMP.
   #ifdef HAS_OPENMP
     #include <omp.h>
   #endif
   
   // Use Armadillo's C++ version detection.
   #ifdef ARMA_USE_CXX11
     #define MLPACK_USE_CX11
   #endif
   
   #endif
