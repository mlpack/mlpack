
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_elu.hpp:

Program Listing for File elu.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_elu.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/elu.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_ELU_HPP
   #define MLPACK_METHODS_ANN_LAYER_ELU_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class ELU
   {
    public:
     ELU();
   
     ELU(const double alpha);
   
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
   
     double const& Lambda() const { return lambda; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     arma::mat derivative;
   
     double alpha;
   
     double lambda;
   
     bool deterministic;
   }; // class ELU
   
   // Template alias for SELU using ELU class.
   using SELU = ELU<arma::mat, arma::mat>;
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "elu_impl.hpp"
   
   #endif
