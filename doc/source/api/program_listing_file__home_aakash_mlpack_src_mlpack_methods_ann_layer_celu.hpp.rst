
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_celu.hpp:

Program Listing for File celu.hpp
=================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_celu.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/celu.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     author  = {Jonathan T. Barron},
     title   = {Continuously Differentiable Exponential Linear Units},
     year    = {2017},
     url     = {https://arxiv.org/pdf/1704.07483}
   }
   
   #ifndef MLPACK_METHODS_ANN_LAYER_CELU_HPP
   #define MLPACK_METHODS_ANN_LAYER_CELU_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class CELU
   {
    public:
     CELU(const double alpha = 1.0);
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& input, OutputType& output);
   
     template<typename DataType>
     void Backward(const DataType& input, const DataType& gy, DataType& g);
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     double const& Alpha() const { return alpha; }
     double& Alpha() { return alpha; }
   
     bool Deterministic() const { return deterministic; }
     bool& Deterministic() { return deterministic; }
   
     size_t WeightSize() { return 0; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     arma::mat derivative;
   
     double alpha;
   
     bool deterministic;
   }; // class CELU
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "celu_impl.hpp"
   
   #endif
