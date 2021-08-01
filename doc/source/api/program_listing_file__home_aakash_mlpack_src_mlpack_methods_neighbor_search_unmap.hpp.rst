
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_unmap.hpp:

Program Listing for File unmap.hpp
==================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_unmap.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/neighbor_search/unmap.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_UNMAP_HPP
   #define MLPACK_METHODS_NEIGHBOR_SEARCH_UNMAP_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace neighbor {
   
   void Unmap(const arma::Mat<size_t>& neighbors,
              const arma::mat& distances,
              const std::vector<size_t>& referenceMap,
              const std::vector<size_t>& queryMap,
              arma::Mat<size_t>& neighborsOut,
              arma::mat& distancesOut,
              const bool squareRoot = false);
   
   void Unmap(const arma::Mat<size_t>& neighbors,
              const arma::mat& distances,
              const std::vector<size_t>& referenceMap,
              arma::Mat<size_t>& neighborsOut,
              arma::mat& distancesOut,
              const bool squareRoot = false);
   
   } // namespace neighbor
   } // namespace mlpack
   
   #endif
