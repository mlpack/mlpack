
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_constant.hpp:

Program Listing for File constant.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_constant.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/constant.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_CONSTANT_HPP
   #define MLPACK_METHODS_ANN_LAYER_CONSTANT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class Constant
   {
    public:
     Constant(const size_t outSize = 0, const double scalar = 0.0);
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& input, OutputType& output);
   
     template<typename DataType>
     void Backward(const DataType& /* input */,
                   const DataType& /* gy */,
                   DataType& g);
   
     OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     size_t OutSize() const { return outSize; }
   
     size_t WeightSize() const
     {
       return 0;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t inSize;
   
     size_t outSize;
   
     OutputDataType constantOutput;
   
     OutputDataType delta;
   
     OutputDataType outputParameter;
   }; // class ConstantLayer
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "constant_impl.hpp"
   
   #endif
