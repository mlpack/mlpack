
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_c_relu.hpp:

Program Listing for File c_relu.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_c_relu.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/c_relu.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_C_RELU_HPP
   #define MLPACK_METHODS_ANN_LAYER_C_RELU_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class CReLU
   {
    public:
     CReLU();
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& input, OutputType& output);
   
     template<typename DataType>
     void Backward(const DataType& input, const DataType& gy, DataType& g);
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     size_t WeightSize() const { return 0; }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */);
   
    private:
     OutputDataType delta;
   
     OutputDataType outputParameter;
   }; // class CReLU
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "c_relu_impl.hpp"
   
   #endif
