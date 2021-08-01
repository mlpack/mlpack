
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_random_partition.hpp:

Program Listing for File random_partition.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_random_partition.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/random_partition.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KMEANS_RANDOM_PARTITION_HPP
   #define MLPACK_METHODS_KMEANS_RANDOM_PARTITION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace kmeans {
   
   class RandomPartition
   {
    public:
     RandomPartition() { }
   
     template<typename MatType>
     inline static void Cluster(const MatType& data,
                                const size_t clusters,
                                arma::Row<size_t>& assignments)
     {
       // Implementation is so simple we'll put it here in the header file.
       assignments = arma::shuffle(arma::linspace<arma::Row<size_t>>(0,
           (clusters - 1), data.n_cols));
     }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   };
   
   } // namespace kmeans
   } // namespace mlpack
   
   #endif
