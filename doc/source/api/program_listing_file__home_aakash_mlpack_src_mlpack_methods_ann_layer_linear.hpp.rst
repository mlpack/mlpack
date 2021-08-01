
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_linear.hpp:

Program Listing for File linear.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_linear.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/linear.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_LINEAR_HPP
   #define MLPACK_METHODS_ANN_LAYER_LINEAR_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/ann/regularizer/no_regularizer.hpp>
   
   #include "layer_types.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat,
       typename RegularizerType = NoRegularizer
   >
   class Linear
   {
    public:
     Linear();
   
     Linear(const size_t inSize,
            const size_t outSize,
            RegularizerType regularizer = RegularizerType());
   
     Linear(const Linear& layer);
   
     Linear(Linear&&);
   
     Linear& operator=(const Linear& layer);
   
     Linear& operator=(Linear&& layer);
   
     /*
      * Reset the layer parameter.
      */
     void Reset();
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     /*
      * Calculate the gradient using the output delta and the input activation.
      *
      * @param input The input parameter used for calculating the gradient.
      * @param error The calculated error.
      * @param gradient The calculated gradient.
      */
     template<typename eT>
     void Gradient(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& error,
                   arma::Mat<eT>& gradient);
   
     OutputDataType const& Parameters() const { return weights; }
     OutputDataType& Parameters() { return weights; }
   
     InputDataType const& InputParameter() const { return inputParameter; }
     InputDataType& InputParameter() { return inputParameter; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     size_t InputSize() const { return inSize; }
   
     size_t OutputSize() const { return outSize; }
   
     OutputDataType const& Gradient() const { return gradient; }
     OutputDataType& Gradient() { return gradient; }
   
     OutputDataType const& Weight() const { return weight; }
     OutputDataType& Weight() { return weight; }
   
     OutputDataType const& Bias() const { return bias; }
     OutputDataType& Bias() { return bias; }
   
     size_t WeightSize() const
     {
       return (inSize * outSize) + outSize;
     }
   
     size_t InputShape() const
     {
       return inSize;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t inSize;
   
     size_t outSize;
   
     OutputDataType weights;
   
     OutputDataType weight;
   
     OutputDataType bias;
   
     OutputDataType delta;
   
     OutputDataType gradient;
   
     InputDataType inputParameter;
   
     OutputDataType outputParameter;
   
     RegularizerType regularizer;
   }; // class Linear
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "linear_impl.hpp"
   
   #endif
