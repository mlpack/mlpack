
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_rann_ra_typedef.hpp:

Program Listing for File ra_typedef.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_rann_ra_typedef.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/rann/ra_typedef.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_RANN_RA_TYPEDEF_HPP
   #define MLPACK_RANN_RA_TYPEDEF_HPP
   
   // In case someone included this directly.
   #include "ra_search.hpp"
   
   #include <mlpack/core/metrics/lmetric.hpp>
   
   #include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>
   #include <mlpack/methods/neighbor_search/sort_policies/furthest_neighbor_sort.hpp>
   
   namespace mlpack {
   namespace neighbor {
   
   typedef RASearch<> KRANN;
   
   typedef RASearch<FurthestNeighborSort> KRAFN;
   
   } // namespace neighbor
   } // namespace mlpack
   
   #endif
