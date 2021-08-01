
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_multiply_constant.hpp:

Program Listing for File multiply_constant.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_multiply_constant.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/multiply_constant.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_MULTIPLY_CONSTANT_HPP
   #define MLPACK_METHODS_ANN_LAYER_MULTIPLY_CONSTANT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class MultiplyConstant
   {
    public:
     MultiplyConstant(const double scalar = 1.0);
   
     MultiplyConstant(const MultiplyConstant& layer);
   
     MultiplyConstant(MultiplyConstant&& layer);
   
     MultiplyConstant& operator=(const MultiplyConstant& layer);
   
     MultiplyConstant& operator=(MultiplyConstant&& layer);
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& input, OutputType& output);
   
     template<typename DataType>
     void Backward(const DataType& /* input */, const DataType& gy, DataType& g);
   
     OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     double Scalar() const { return scalar; }
     double& Scalar() { return scalar; }
   
     size_t WeightSize() const { return 0; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     double scalar;
   
     OutputDataType delta;
   
     OutputDataType outputParameter;
   }; // class MultiplyConstant
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "multiply_constant_impl.hpp"
   
   #endif
