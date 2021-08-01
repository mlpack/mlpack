
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_minibatch_discrimination.hpp:

Program Listing for File minibatch_discrimination.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_minibatch_discrimination.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/minibatch_discrimination.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_MINIBATCH_DISCRIMINATION_HPP
   #define MLPACK_METHODS_ANN_LAYER_MINIBATCH_DISCRIMINATION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "layer_types.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class MiniBatchDiscrimination
   {
    public:
     MiniBatchDiscrimination();
   
     MiniBatchDiscrimination(const size_t inSize,
                             const size_t outSize,
                             const size_t features);
   
     void Reset();
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     template<typename eT>
     void Gradient(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& /* error */,
                   arma::Mat<eT>& gradient);
   
     OutputDataType const& Parameters() const { return weights; }
     OutputDataType& Parameters() { return weights; }
   
     InputDataType const& InputParameter() const { return inputParameter; }
     InputDataType& InputParameter() { return inputParameter; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     OutputDataType const& Gradient() const { return gradient; }
     OutputDataType& Gradient() { return gradient; }
   
     size_t InputShape() const
     {
       return A;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t A, B, C;
   
     size_t batchSize;
   
     arma::mat tempM;
   
     OutputDataType weights;
   
     OutputDataType weight;
   
     arma::cube M;
   
     arma::cube deltaM;
   
     arma::cube distances;
   
     OutputDataType delta;
   
     OutputDataType deltaTemp;
   
     OutputDataType gradient;
   
     InputDataType inputParameter;
   
     OutputDataType outputParameter;
   }; // class MiniBatchDiscrimination
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "minibatch_discrimination_impl.hpp"
   
   #endif
