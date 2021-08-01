
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_naive_kmeans_impl.hpp:

Program Listing for File naive_kmeans_impl.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_naive_kmeans_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/naive_kmeans_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KMEANS_NAIVE_KMEANS_IMPL_HPP
   #define MLPACK_METHODS_KMEANS_NAIVE_KMEANS_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "naive_kmeans.hpp"
   
   namespace mlpack {
   namespace kmeans {
   
   template<typename MetricType, typename MatType>
   NaiveKMeans<MetricType, MatType>::NaiveKMeans(const MatType& dataset,
                                                 MetricType& metric) :
       dataset(dataset),
       metric(metric),
       distanceCalculations(0)
   { /* Nothing to do. */ }
   
   // Run a single iteration.
   template<typename MetricType, typename MatType>
   double NaiveKMeans<MetricType, MatType>::Iterate(const arma::mat& centroids,
                                                    arma::mat& newCentroids,
                                                    arma::Col<size_t>& counts)
   {
     newCentroids.zeros(centroids.n_rows, centroids.n_cols);
     counts.zeros(centroids.n_cols);
   
     // Find the closest centroid to each point and update the new centroids.
     // Computed in parallel over the complete dataset
     #pragma omp parallel
     {
       // The current state of the K-means is private for each thread
       arma::mat localCentroids(centroids.n_rows, centroids.n_cols,
           arma::fill::zeros);
       arma::Col<size_t> localCounts(centroids.n_cols, arma::fill::zeros);
   
       #pragma omp for
       for (omp_size_t i = 0; i < (omp_size_t) dataset.n_cols; ++i)
       {
         // Find the closest centroid to this point.
         double minDistance = std::numeric_limits<double>::infinity();
         size_t closestCluster = centroids.n_cols; // Invalid value.
   
         for (size_t j = 0; j < centroids.n_cols; ++j)
         {
           const double distance = metric.Evaluate(dataset.col(i),
               centroids.unsafe_col(j));
           if (distance < minDistance)
           {
             minDistance = distance;
             closestCluster = j;
           }
         }
   
         Log::Assert(closestCluster != centroids.n_cols);
   
         // We now have the minimum distance centroid index.  Update that centroid.
         localCentroids.unsafe_col(closestCluster) += dataset.col(i);
         localCounts(closestCluster)++;
       }
       // Combine calculated state from each thread
       #pragma omp critical
       {
         newCentroids += localCentroids;
         counts += localCounts;
       }
     }
   
     // Now normalize the centroid.
     for (size_t i = 0; i < centroids.n_cols; ++i)
       if (counts(i) != 0)
         newCentroids.col(i) /= counts(i);
   
     distanceCalculations += centroids.n_cols * dataset.n_cols;
   
     // Calculate cluster distortion for this iteration.
     double cNorm = 0.0;
     for (size_t i = 0; i < centroids.n_cols; ++i)
     {
       cNorm += std::pow(metric.Evaluate(centroids.col(i), newCentroids.col(i)),
           2.0);
     }
     distanceCalculations += centroids.n_cols;
   
     return std::sqrt(cNorm);
   }
   
   } // namespace kmeans
   } // namespace mlpack
   
   #endif
