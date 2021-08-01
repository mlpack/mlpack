
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_batch_norm.hpp:

Program Listing for File batch_norm.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_batch_norm.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/batch_norm.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_BATCHNORM_HPP
   #define MLPACK_METHODS_ANN_LAYER_BATCHNORM_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
     typename InputDataType = arma::mat,
     typename OutputDataType = arma::mat
   >
   class BatchNorm
   {
    public:
     BatchNorm();
   
     BatchNorm(const size_t size,
               const double eps = 1e-8,
               const bool average = true,
               const double momentum = 0.1);
   
     void Reset();
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     template<typename eT>
     void Gradient(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& error,
                   arma::Mat<eT>& gradient);
   
     OutputDataType const& Parameters() const { return weights; }
     OutputDataType& Parameters() { return weights; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     OutputDataType const& Gradient() const { return gradient; }
     OutputDataType& Gradient() { return gradient; }
   
     bool Deterministic() const { return deterministic; }
     bool& Deterministic() { return deterministic; }
   
     OutputDataType const& TrainingMean() const { return runningMean; }
     OutputDataType& TrainingMean() { return runningMean; }
   
     OutputDataType const& TrainingVariance() const { return runningVariance; }
     OutputDataType& TrainingVariance() { return runningVariance; }
   
     size_t InputSize() const { return size; }
   
     double Epsilon() const { return eps; }
   
     double Momentum() const { return momentum; }
   
     bool Average() const { return average; }
   
     size_t WeightSize() const { return 2 * size; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t size;
   
     double eps;
   
     bool average;
   
     double momentum;
   
     bool loading;
   
     OutputDataType gamma;
   
     OutputDataType beta;
   
     OutputDataType mean;
   
     OutputDataType variance;
   
     OutputDataType weights;
   
     bool deterministic;
   
     size_t count;
   
     double averageFactor;
   
     OutputDataType runningMean;
   
     OutputDataType runningVariance;
   
     OutputDataType gradient;
   
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     arma::cube normalized;
   
     arma::cube inputMean;
   }; // class BatchNorm
   
   } // namespace ann
   } // namespace mlpack
   
   // Include the implementation.
   #include "batch_norm_impl.hpp"
   
   #endif
