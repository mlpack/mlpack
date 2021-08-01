
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_sample_initialization.hpp:

Program Listing for File sample_initialization.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_sample_initialization.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/sample_initialization.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef __MLPACK_METHODS_KMEANS_SAMPLE_INITIALIZATION_HPP
   #define __MLPACK_METHODS_KMEANS_SAMPLE_INITIALIZATION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/math/random.hpp>
   
   namespace mlpack {
   namespace kmeans {
   
   class SampleInitialization
   {
    public:
     SampleInitialization() { }
   
     template<typename MatType>
     inline static void Cluster(const MatType& data,
                                const size_t clusters,
                                arma::mat& centroids)
     {
       centroids.set_size(data.n_rows, clusters);
       for (size_t i = 0; i < clusters; ++i)
       {
         // Randomly sample a point.
         const size_t index = math::RandInt(0, data.n_cols);
         centroids.col(i) = data.col(index);
       }
     }
   };
   
   } // namespace kmeans
   } // namespace mlpack
   
   #endif
