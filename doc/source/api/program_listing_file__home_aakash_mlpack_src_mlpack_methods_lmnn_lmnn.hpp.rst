
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_lmnn_lmnn.hpp:

Program Listing for File lmnn.hpp
=================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_lmnn_lmnn.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/lmnn/lmnn.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LMNN_LMNN_HPP
   #define MLPACK_METHODS_LMNN_LMNN_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   #include <ensmallen.hpp>
   
   #include "lmnn_function.hpp"
   
   namespace mlpack {
   namespace lmnn  {
   
   template<typename MetricType = metric::SquaredEuclideanDistance,
            typename OptimizerType = ens::AMSGrad>
   class LMNN
   {
    public:
     LMNN(const arma::mat& dataset,
          const arma::Row<size_t>& labels,
          const size_t k,
          const MetricType metric = MetricType());
   
   
     template<typename... CallbackTypes>
     void LearnDistance(arma::mat& outputMatrix, CallbackTypes&&... callbacks);
   
   
     const arma::mat& Dataset() const { return dataset; }
   
     const arma::Row<size_t>& Labels() const { return labels; }
   
     const double& Regularization() const { return regularization; }
     double& Regularization() { return regularization; }
   
     const size_t& Range() const { return range; }
     size_t& Range() { return range; }
   
     const size_t& K() const { return k; }
     size_t K() { return k; }
   
     const OptimizerType& Optimizer() const { return optimizer; }
     OptimizerType& Optimizer() { return optimizer; }
   
    private:
     const arma::mat& dataset;
   
     const arma::Row<size_t>& labels;
   
     size_t k;
   
     double regularization;
   
     size_t range;
   
     MetricType metric;
   
     OptimizerType optimizer;
   }; // class LMNN
   
   } // namespace lmnn
   } // namespace mlpack
   
   // Include the implementation.
   #include "lmnn_impl.hpp"
   
   #endif
