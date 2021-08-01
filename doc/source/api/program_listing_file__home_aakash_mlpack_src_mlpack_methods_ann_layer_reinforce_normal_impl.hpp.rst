
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_reinforce_normal_impl.hpp:

Program Listing for File reinforce_normal_impl.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_reinforce_normal_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/reinforce_normal_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_REINFORCE_NORMAL_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_REINFORCE_NORMAL_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "reinforce_normal.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   ReinforceNormal<InputDataType, OutputDataType>::ReinforceNormal(
       const double stdev) : stdev(stdev), reward(0.0), deterministic(false)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void ReinforceNormal<InputDataType, OutputDataType>::Forward(
       const arma::Mat<eT>& input, arma::Mat<eT>& output)
   {
     if (!deterministic)
     {
       // Multiply by standard deviations and re-center the means to the mean.
       output = output.randn(input.n_rows, input.n_cols) * stdev + input;
   
       moduleInputParameter.push_back(input);
     }
     else
     {
       // Use maximum a posteriori.
       output = input;
     }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename DataType>
   void ReinforceNormal<InputDataType, OutputDataType>::Backward(
       const DataType& input, const DataType& /* gy */, DataType& g)
   {
     g = (input - moduleInputParameter.back()) / std::pow(stdev, 2.0);
   
     // Multiply by reward and multiply by -1.
     g *= reward;
     g *= -1;
   
     moduleInputParameter.pop_back();
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void ReinforceNormal<InputDataType, OutputDataType>::serialize(
       Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(stdev));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
