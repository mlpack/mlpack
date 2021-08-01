
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_kmeans_plus_plus_initialization.hpp:

Program Listing for File kmeans_plus_plus_initialization.hpp
============================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_kmeans_plus_plus_initialization.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/kmeans_plus_plus_initialization.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KMEANS_KMEANS_PLUS_PLUS_INITIALIZATION_HPP
   #define MLPACK_METHODS_KMEANS_KMEANS_PLUS_PLUS_INITIALIZATION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   class KMeansPlusPlusInitialization
   {
    public:
     KMeansPlusPlusInitialization() { }
   
     template<typename MatType>
     inline static void Cluster(const MatType& data,
                                const size_t clusters,
                                arma::mat& centroids)
     {
       centroids.set_size(data.n_rows, clusters);
   
       // We'll sample our first point fully randomly.
       size_t firstPoint = mlpack::math::RandInt(0, data.n_cols);
       centroids.col(0) = data.col(firstPoint);
   
       // Utility variable.
       arma::vec distribution(data.n_cols);
   
       // Now, sample other points...
       for (size_t i = 1; i < clusters; ++i)
       {
         // We must compute the CDF for sampling... this depends on the computation
         // of the minimum distance between each point and its closest
         // already-chosen centroid.
         //
         // This computation is ripe for speedup with trees!  I am not sure exactly
         // how much we would need to approximate, but I think it could be done
         // without breaking the O(log k)-competitive guarantee (I think).
         for (size_t p = 0; p < data.n_cols; ++p)
         {
           double minDistance = std::numeric_limits<double>::max();
           for (size_t j = 0; j < i; ++j)
           {
             const double distance =
                 mlpack::metric::SquaredEuclideanDistance::Evaluate(data.col(p),
                 centroids.col(j));
             minDistance = std::min(distance, minDistance);
           }
   
           distribution[p] = minDistance;
         }
   
         // Next normalize the distribution (actually technically we could avoid
         // this).
         distribution /= arma::accu(distribution);
   
         // Turn it into a CDF for convenience...
         for (size_t j = 1; j < distribution.n_elem; ++j)
           distribution[j] += distribution[j - 1];
   
         // Sample a point...
         const double sampleValue = mlpack::math::Random();
         const double* elem = std::lower_bound(distribution.begin(),
             distribution.end(), sampleValue);
         const size_t position = (size_t)
             (elem - distribution.begin()) / sizeof(double);
         centroids.col(i) = data.col(position);
       }
     }
   };
   
   #endif
