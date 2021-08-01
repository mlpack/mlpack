
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_nca_nca.hpp:

Program Listing for File nca.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_nca_nca.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/nca/nca.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_NCA_NCA_HPP
   #define MLPACK_METHODS_NCA_NCA_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   #include <ensmallen.hpp>
   
   #include "nca_softmax_error_function.hpp"
   
   namespace mlpack {
   namespace nca  {
   
   template<typename MetricType = metric::SquaredEuclideanDistance,
            typename OptimizerType = ens::StandardSGD>
   class NCA
   {
    public:
     NCA(const arma::mat& dataset,
         const arma::Row<size_t>& labels,
         MetricType metric = MetricType());
   
     template<typename... CallbackTypes>
     void LearnDistance(arma::mat& outputMatrix, CallbackTypes&&... callbacks);
   
     const arma::mat& Dataset() const { return dataset; }
     const arma::Row<size_t>& Labels() const { return labels; }
   
     const OptimizerType& Optimizer() const { return optimizer; }
     OptimizerType& Optimizer() { return optimizer; }
   
    private:
     const arma::mat& dataset;
     const arma::Row<size_t>& labels;
   
     MetricType metric;
   
     SoftmaxErrorFunction<MetricType> errorFunction;
   
     OptimizerType optimizer;
   };
   
   } // namespace nca
   } // namespace mlpack
   
   // Include the implementation.
   #include "nca_impl.hpp"
   
   #endif
