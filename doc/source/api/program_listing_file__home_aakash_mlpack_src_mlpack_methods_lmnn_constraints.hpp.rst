
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_lmnn_constraints.hpp:

Program Listing for File constraints.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_lmnn_constraints.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/lmnn/constraints.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LMNN_CONSTRAINTS_HPP
   #define MLPACK_METHODS_LMNN_CONSTRAINTS_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/neighbor_search/neighbor_search.hpp>
   
   namespace mlpack {
   namespace lmnn {
   template<typename MetricType = metric::SquaredEuclideanDistance>
   class Constraints
   {
    public:
     typedef neighbor::NeighborSearch<neighbor::NearestNeighborSort, MetricType>
         KNN;
   
     Constraints(const arma::mat& dataset,
                 const arma::Row<size_t>& labels,
                 const size_t k);
   
     void TargetNeighbors(arma::Mat<size_t>& outputMatrix,
                          const arma::mat& dataset,
                          const arma::Row<size_t>& labels,
                          const arma::vec& norms);
   
     void TargetNeighbors(arma::Mat<size_t>& outputMatrix,
                          const arma::mat& dataset,
                          const arma::Row<size_t>& labels,
                          const arma::vec& norms,
                          const size_t begin,
                          const size_t batchSize);
   
     void Impostors(arma::Mat<size_t>& outputMatrix,
                    const arma::mat& dataset,
                    const arma::Row<size_t>& labels,
                    const arma::vec& norms);
   
     void Impostors(arma::Mat<size_t>& outputNeighbors,
                    arma::mat& outputDistance,
                    const arma::mat& dataset,
                    const arma::Row<size_t>& labels,
                    const arma::vec& norms);
   
     void Impostors(arma::Mat<size_t>& outputMatrix,
                    const arma::mat& dataset,
                    const arma::Row<size_t>& labels,
                    const arma::vec& norms,
                    const size_t begin,
                    const size_t batchSize);
   
     void Impostors(arma::Mat<size_t>& outputNeighbors,
                    arma::mat& outputDistance,
                    const arma::mat& dataset,
                    const arma::Row<size_t>& labels,
                    const arma::vec& norms,
                    const size_t begin,
                    const size_t batchSize);
   
     void Impostors(arma::Mat<size_t>& outputNeighbors,
                    arma::mat& outputDistance,
                    const arma::mat& dataset,
                    const arma::Row<size_t>& labels,
                    const arma::vec& norms,
                    const arma::uvec& points,
                    const size_t numPoints);
   
     void Triplets(arma::Mat<size_t>& outputMatrix,
                   const arma::mat& dataset,
                   const arma::Row<size_t>& labels,
                   const arma::vec& norms);
   
     const size_t& K() const { return k; }
     size_t& K() { return k; }
   
     const bool& PreCalulated() const { return precalculated; }
     bool& PreCalulated() { return precalculated; }
   
    private:
     size_t k;
   
     arma::Row<size_t> uniqueLabels;
   
     std::vector<arma::uvec> indexSame;
   
     std::vector<arma::uvec> indexDiff;
   
     bool precalculated;
   
     inline void Precalculate(const arma::Row<size_t>& labels);
   
     inline void ReorderResults(const arma::mat& distances,
                                arma::Mat<size_t>& neighbors,
                                const arma::vec& norms);
   };
   
   } // namespace lmnn
   } // namespace mlpack
   
   // Include implementation.
   #include "constraints_impl.hpp"
   
   #endif
