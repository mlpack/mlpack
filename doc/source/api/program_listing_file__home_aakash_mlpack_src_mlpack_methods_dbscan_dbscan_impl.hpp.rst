
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_dbscan_dbscan_impl.hpp:

Program Listing for File dbscan_impl.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_dbscan_dbscan_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/dbscan/dbscan_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_DBSCAN_DBSCAN_IMPL_HPP
   #define MLPACK_METHODS_DBSCAN_DBSCAN_IMPL_HPP
   
   #include "dbscan.hpp"
   
   namespace mlpack {
   namespace dbscan {
   
   template<typename RangeSearchType, typename PointSelectionPolicy>
   DBSCAN<RangeSearchType, PointSelectionPolicy>::DBSCAN(
       const double epsilon,
       const size_t minPoints,
       const bool batchMode,
       RangeSearchType rangeSearch,
       PointSelectionPolicy pointSelector) :
       epsilon(epsilon),
       minPoints(minPoints),
       batchMode(batchMode),
       rangeSearch(rangeSearch),
       pointSelector(pointSelector)
   {
     // Nothing to do.
   }
   
   template<typename RangeSearchType, typename PointSelectionPolicy>
   template<typename MatType>
   size_t DBSCAN<RangeSearchType, PointSelectionPolicy>::Cluster(
       const MatType& data,
       arma::mat& centroids)
   {
     // These assignments will be thrown away, but there is no way to avoid
     // calculating them.
     arma::Row<size_t> assignments(data.n_cols);
     assignments.fill(SIZE_MAX);
   
     return Cluster(data, assignments, centroids);
   }
   
   template<typename RangeSearchType, typename PointSelectionPolicy>
   template<typename MatType>
   size_t DBSCAN<RangeSearchType, PointSelectionPolicy>::Cluster(
       const MatType& data,
       arma::Row<size_t>& assignments,
       arma::mat& centroids)
   {
     const size_t numClusters = Cluster(data, assignments);
   
     // Now calculate the centroids.
     centroids.zeros(data.n_rows, numClusters);
   
     // Calculate number of points in each cluster.
     arma::Row<size_t> counts;
     counts.zeros(numClusters);
     for (size_t i = 0; i < data.n_cols; ++i)
     {
       if (assignments[i] != SIZE_MAX)
       {
         centroids.col(assignments[i]) += data.col(i);
         ++counts[assignments[i]];
       }
     }
   
     // We should be guaranteed that the number of clusters is always greater than
     // zero.
     for (size_t i = 0; i < numClusters; ++i)
       centroids.col(i) /= counts[i];
   
     return numClusters;
   }
   
   template<typename RangeSearchType, typename PointSelectionPolicy>
   template<typename MatType>
   size_t DBSCAN<RangeSearchType, PointSelectionPolicy>::Cluster(
       const MatType& data,
       arma::Row<size_t>& assignments)
   {
     // Initialize the UnionFind object.
     emst::UnionFind uf(data.n_cols);
     rangeSearch.Train(data);
   
     if (batchMode)
       BatchCluster(data, uf);
     else
       PointwiseCluster(data, uf);
   
     // Now set assignments.
     assignments.set_size(data.n_cols);
     for (size_t i = 0; i < data.n_cols; ++i)
       assignments[i] = uf.Find(i);
   
     // Get a count of all clusters.
     const size_t numClusters = arma::max(assignments) + 1;
     arma::Col<size_t> counts(numClusters, arma::fill::zeros);
     for (size_t i = 0; i < assignments.n_elem; ++i)
       counts[assignments[i]]++;
   
     // Now assign clusters to new indices.
     size_t currentCluster = 0;
     arma::Col<size_t> newAssignments(numClusters);
     for (size_t i = 0; i < counts.n_elem; ++i)
     {
       if (counts[i] >= minPoints)
         newAssignments[i] = currentCluster++;
       else
         newAssignments[i] = SIZE_MAX;
     }
   
     // Now reassign.
     for (size_t i = 0; i < assignments.n_elem; ++i)
       assignments[i] = newAssignments[assignments[i]];
   
     Log::Info << currentCluster << " clusters found." << std::endl;
   
     return currentCluster;
   }
   
   template<typename RangeSearchType, typename PointSelectionPolicy>
   template<typename MatType>
   void DBSCAN<RangeSearchType, PointSelectionPolicy>::PointwiseCluster(
       const MatType& data,
       emst::UnionFind& uf)
   {
     std::vector<std::vector<size_t>> neighbors;
     std::vector<std::vector<double>> distances;
   
     for (size_t i = 0; i < data.n_cols; ++i)
     {
       if (i % 10000 == 0 && i > 0)
         Log::Info << "DBSCAN clustering on point " << i << "..." << std::endl;
   
       // Do the range search for only this point.
       rangeSearch.Search(data.col(i), math::Range(0.0, epsilon), neighbors,
           distances);
   
       // Union to all neighbors.
       for (size_t j = 0; j < neighbors[0].size(); ++j)
         uf.Union(i, neighbors[0][j]);
     }
   }
   
   template<typename RangeSearchType, typename PointSelectionPolicy>
   template<typename MatType>
   void DBSCAN<RangeSearchType, PointSelectionPolicy>::BatchCluster(
       const MatType& data,
       emst::UnionFind& uf)
   {
     // For each point, find the points in epsilon-nighborhood and their distances.
     std::vector<std::vector<size_t>> neighbors;
     std::vector<std::vector<double>> distances;
     Log::Info << "Performing range search." << std::endl;
     rangeSearch.Train(data);
     rangeSearch.Search(data, math::Range(0.0, epsilon), neighbors, distances);
     Log::Info << "Range search complete." << std::endl;
   
     // Now loop over all points.
     for (size_t i = 0; i < data.n_cols; ++i)
     {
       // Get the next index.
       const size_t index = pointSelector.Select(i, data);
       for (size_t j = 0; j < neighbors[index].size(); ++j)
         uf.Union(index, neighbors[index][j]);
     }
   }
   
   } // namespace dbscan
   } // namespace mlpack
   
   #endif
