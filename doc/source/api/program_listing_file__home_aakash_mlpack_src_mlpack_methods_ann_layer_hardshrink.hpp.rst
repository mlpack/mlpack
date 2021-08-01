
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_hardshrink.hpp:

Program Listing for File hardshrink.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_hardshrink.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/hardshrink.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_HARDSHRINK_HPP
   #define MLPACK_METHODS_ANN_LAYER_HARDSHRINK_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class HardShrink
   {
    public:
     HardShrink(const double lambda = 0.5);
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& input, OutputType& output);
   
     template<typename DataType>
     void Backward(const DataType& input,
                   DataType& gy,
                   DataType& g);
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     double const& Lambda() const { return lambda; }
     double& Lambda() { return lambda; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     double lambda;
   }; // class HardShrink
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "hardshrink_impl.hpp"
   
   #endif
