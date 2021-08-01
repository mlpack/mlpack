
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_hard_tanh.hpp:

Program Listing for File hard_tanh.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_hard_tanh.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/hard_tanh.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_HARD_TANH_HPP
   #define MLPACK_METHODS_ANN_LAYER_HARD_TANH_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class HardTanH
   {
    public:
     HardTanH(const double maxValue = 1, const double minValue = -1);
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& input, OutputType& output);
   
     template<typename DataType>
     void Backward(const DataType& input,
                   const DataType& gy,
                   DataType& g);
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     double const& MaxValue() const { return maxValue; }
     double& MaxValue() { return maxValue; }
   
     double const& MinValue() const { return minValue; }
     double& MinValue() { return minValue; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     double maxValue;
   
     double minValue;
   }; // class HardTanH
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "hard_tanh_impl.hpp"
   
   #endif
