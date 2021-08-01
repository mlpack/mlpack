
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_unmap.cpp:

Program Listing for File unmap.cpp
==================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_unmap.cpp>` (``/home/aakash/mlpack/src/mlpack/methods/neighbor_search/unmap.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include "unmap.hpp"
   
   namespace mlpack {
   namespace neighbor {
   
   // Useful in the dual-tree setting.
   void Unmap(const arma::Mat<size_t>& neighbors,
              const arma::mat& distances,
              const std::vector<size_t>& referenceMap,
              const std::vector<size_t>& queryMap,
              arma::Mat<size_t>& neighborsOut,
              arma::mat& distancesOut,
              const bool squareRoot)
   {
     // Set matrices to correct size.
     neighborsOut.set_size(neighbors.n_rows, neighbors.n_cols);
     distancesOut.set_size(distances.n_rows, distances.n_cols);
   
     // Unmap distances.
     for (size_t i = 0; i < distances.n_cols; ++i)
     {
       // Map columns to the correct place.  The ternary operator does not work
       // here...
       if (squareRoot)
         distancesOut.col(queryMap[i]) = sqrt(distances.col(i));
       else
         distancesOut.col(queryMap[i]) = distances.col(i);
   
       // Map indices of neighbors.
       for (size_t j = 0; j < distances.n_rows; ++j)
         neighborsOut(j, queryMap[i]) = referenceMap[neighbors(j, i)];
     }
   }
   
   // Useful in the single-tree setting.
   void Unmap(const arma::Mat<size_t>& neighbors,
              const arma::mat& distances,
              const std::vector<size_t>& referenceMap,
              arma::Mat<size_t>& neighborsOut,
              arma::mat& distancesOut,
              const bool squareRoot)
   {
     // Set matrices to correct size.
     neighborsOut.set_size(neighbors.n_rows, neighbors.n_cols);
   
     // Take square root of distances, if necessary.
     if (squareRoot)
       distancesOut = sqrt(distances);
     else
       distancesOut = distances;
   
     // Map neighbors back to original locations.
     for (size_t j = 0; j < neighbors.n_elem; ++j)
       neighborsOut[j] = referenceMap[neighbors[j]];
   }
   
   } // namespace neighbor
   } // namespace mlpack
