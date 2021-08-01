
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_neighbor_search_policies_pearson_search.hpp:

Program Listing for File pearson_search.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_cf_neighbor_search_policies_pearson_search.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/cf/neighbor_search_policies/pearson_search.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_CF_PEARSON_SEARCH_HPP
   #define MLPACK_METHODS_CF_PEARSON_SEARCH_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/neighbor_search/neighbor_search.hpp>
   
   namespace mlpack {
   namespace cf {
   
   class PearsonSearch
   {
    public:
     PearsonSearch(const arma::mat& referenceSet)
     {
       // Normalize all vectors in referenceSet.
       // For each vector x, first subtract mean(x) from each element in x.
       // Then normalize the vector to unit length.
       arma::mat normalizedSet(arma::size(referenceSet));
       normalizedSet = arma::normalise(
           referenceSet.each_row() - arma::mean(referenceSet));
   
       neighborSearch.Train(std::move(normalizedSet));
     }
   
     void Search(const arma::mat& query, const size_t k,
                 arma::Mat<size_t>& neighbors, arma::mat& similarities)
     {
       // Normalize all vectors in query.
       // For each vector x, first subtract mean(x) from each element in x.
       // Then normalize the vector to unit length.
       arma::mat normalizedQuery;
       normalizedQuery = arma::normalise(query.each_row() - arma::mean(query));
   
       neighborSearch.Search(normalizedQuery, k, neighbors, similarities);
   
       // Resulting similarities from Search() are Euclidean distance.
       // For normalized vectors a and b, pearson(a, b) = 1 - dis(a, b) ^ 2 / 2,
       // where dis(a, b) is Euclidean distance.
       // Furthermore, we restrict the range of similarity to be [0, 1]:
       // similarities = (pearson(a,b) + 1) / 2.0. As a result we have the
       // following formula.
       similarities = 1 - arma::pow(similarities, 2) / 4.0;
     }
   
    private:
     neighbor::KNN neighborSearch;
   };
   
   } // namespace cf
   } // namespace mlpack
   
   #endif
