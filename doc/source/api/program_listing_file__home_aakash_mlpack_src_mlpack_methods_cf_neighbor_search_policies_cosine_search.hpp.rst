
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_neighbor_search_policies_cosine_search.hpp:

Program Listing for File cosine_search.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_cf_neighbor_search_policies_cosine_search.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/cf/neighbor_search_policies/cosine_search.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_CF_COSINE_SEARCH_HPP
   #define MLPACK_METHODS_CF_COSINE_SEARCH_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/neighbor_search/neighbor_search.hpp>
   
   namespace mlpack {
   namespace cf {
   
   class CosineSearch
   {
    public:
     CosineSearch(const arma::mat& referenceSet)
     {
       // Normalize all vectors to unit length.
       arma::mat normalizedSet = arma::normalise(referenceSet, 2, 0);
   
       neighborSearch.Train(std::move(normalizedSet));
     }
   
     void Search(const arma::mat& query, const size_t k,
                 arma::Mat<size_t>& neighbors, arma::mat& similarities)
     {
       // Normalize query vectors to unit length.
       arma::mat normalizedQuery = arma::normalise(query, 2, 0);
   
       neighborSearch.Search(normalizedQuery, k, neighbors, similarities);
   
       // Resulting similarities from Search() are Euclidean distance.
       // For unit vectors a and b, cos(a, b) = 1 - dis(a, b) ^ 2 / 2,
       // where dis(a, b) is Euclidean distance.
       // Furthermore, we restrict the range of similarity to be [0, 1]:
       // similarities = (cos(a,b) + 1) / 2.0. As a result we have the following
       // formula.
       similarities = 1 - arma::pow(similarities, 2) / 4.0;
     }
   
    private:
     neighbor::KNN neighborSearch;
   };
   
   } // namespace cf
   } // namespace mlpack
   
   #endif
