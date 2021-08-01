
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_nystroem_method_kmeans_selection.hpp:

Program Listing for File kmeans_selection.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_nystroem_method_kmeans_selection.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/nystroem_method/kmeans_selection.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_NYSTROEM_METHOD_KMEANS_SELECTION_HPP
   #define MLPACK_METHODS_NYSTROEM_METHOD_KMEANS_SELECTION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/kmeans/kmeans.hpp>
   
   namespace mlpack {
   namespace kernel {
   
   template<typename ClusteringType = kmeans::KMeans<>, size_t maxIterations = 5>
   class KMeansSelection
   {
    public:
     const static arma::mat* Select(const arma::mat& data, const size_t m)
     {
       arma::Row<size_t> assignments;
       arma::mat* centroids = new arma::mat;
   
       // Perform the K-Means clustering method.
       ClusteringType kmeans(maxIterations);
       kmeans.Cluster(data, m, assignments, *centroids);
   
       return centroids;
     }
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   #endif
