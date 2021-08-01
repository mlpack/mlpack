
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_refined_start.hpp:

Program Listing for File refined_start.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_refined_start.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/refined_start.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KMEANS_REFINED_START_HPP
   #define MLPACK_METHODS_KMEANS_REFINED_START_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace kmeans {
   
   class RefinedStart
   {
    public:
     RefinedStart(const size_t samplings = 100,
                  const double percentage = 0.02) :
         samplings(samplings), percentage(percentage) { }
   
     template<typename MatType>
     void Cluster(const MatType& data,
                  const size_t clusters,
                  arma::mat& centroids) const;
   
     template<typename MatType>
     void Cluster(const MatType& data,
                  const size_t clusters,
                  arma::Row<size_t>& assignments) const;
   
     size_t Samplings() const { return samplings; }
     size_t& Samplings() { return samplings; }
   
     double Percentage() const { return percentage; }
     double& Percentage() { return percentage; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(samplings));
       ar(CEREAL_NVP(percentage));
     }
   
    private:
     size_t samplings;
     double percentage;
   };
   
   } // namespace kmeans
   } // namespace mlpack
   
   // Include implementation.
   #include "refined_start_impl.hpp"
   
   #endif
