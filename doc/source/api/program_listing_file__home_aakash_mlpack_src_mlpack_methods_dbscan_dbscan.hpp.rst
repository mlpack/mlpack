
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_dbscan_dbscan.hpp:

Program Listing for File dbscan.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_dbscan_dbscan.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/dbscan/dbscan.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_DBSCAN_DBSCAN_HPP
   #define MLPACK_METHODS_DBSCAN_DBSCAN_HPP
   
   #include <mlpack/core.hpp>
   #include <mlpack/methods/range_search/range_search.hpp>
   #include <mlpack/methods/emst/union_find.hpp>
   #include "random_point_selection.hpp"
   #include "ordered_point_selection.hpp"
   #include <boost/dynamic_bitset.hpp>
   
   namespace mlpack {
   namespace dbscan {
   
   template<typename RangeSearchType = range::RangeSearch<>,
            typename PointSelectionPolicy = OrderedPointSelection>
   class DBSCAN
   {
    public:
     DBSCAN(const double epsilon,
            const size_t minPoints,
            const bool batchMode = true,
            RangeSearchType rangeSearch = RangeSearchType(),
            PointSelectionPolicy pointSelector = PointSelectionPolicy());
   
     template<typename MatType>
     size_t Cluster(const MatType& data,
                    arma::mat& centroids);
   
     template<typename MatType>
     size_t Cluster(const MatType& data,
                    arma::Row<size_t>& assignments);
   
     template<typename MatType>
     size_t Cluster(const MatType& data,
                    arma::Row<size_t>& assignments,
                    arma::mat& centroids);
   
    private:
     double epsilon;
   
     size_t minPoints;
   
     bool batchMode;
   
     RangeSearchType rangeSearch;
   
     PointSelectionPolicy pointSelector;
   
     template<typename MatType>
     void PointwiseCluster(const MatType& data,
                           emst::UnionFind& uf);
   
     template<typename MatType>
     void BatchCluster(const MatType& data,
                       emst::UnionFind& uf);
   };
   
   } // namespace dbscan
   } // namespace mlpack
   
   // Include implementation.
   #include "dbscan_impl.hpp"
   
   #endif
