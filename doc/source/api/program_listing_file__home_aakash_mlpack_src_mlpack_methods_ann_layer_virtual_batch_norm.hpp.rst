
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_virtual_batch_norm.hpp:

Program Listing for File virtual_batch_norm.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_virtual_batch_norm.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/virtual_batch_norm.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_VIRTUALBATCHNORM_HPP
   #define MLPACK_METHODS_ANN_LAYER_VIRTUALBATCHNORM_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
     typename InputDataType = arma::mat,
     typename OutputDataType = arma::mat
   >
   class VirtualBatchNorm
   {
    public:
     VirtualBatchNorm();
   
     template<typename eT>
     VirtualBatchNorm(const arma::Mat<eT>& referenceBatch,
                      const size_t size,
                      const double eps = 1e-8);
   
     void Reset();
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     template<typename eT>
     void Gradient(const arma::Mat<eT>& /* input */,
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
   
     size_t InSize() const { return size; }
   
     double Epsilon() const { return eps; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t size;
   
     double eps;
   
     bool loading;
   
     OutputDataType gamma;
   
     OutputDataType beta;
   
     OutputDataType weights;
   
     OutputDataType referenceBatchMean;
   
     OutputDataType referenceBatchMeanSquared;
   
     double oldCoefficient;
   
     double newCoefficient;
   
     OutputDataType mean;
   
     OutputDataType variance;
   
     OutputDataType gradient;
   
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     OutputDataType inputParameter;
   
     OutputDataType normalized;
   
     OutputDataType inputSubMean;
   }; // class VirtualBatchNorm
   
   } // namespace ann
   } // namespace mlpack
   
   // Include the implementation.
   #include "virtual_batch_norm_impl.hpp"
   
   #endif
