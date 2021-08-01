
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_reinforce_normal.hpp:

Program Listing for File reinforce_normal.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_reinforce_normal.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/reinforce_normal.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_REINFORCE_NORMAL_HPP
   #define MLPACK_METHODS_ANN_LAYER_REINFORCE_NORMAL_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class ReinforceNormal
   {
    public:
     ReinforceNormal(const double stdev = 1.0);
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename DataType>
     void Backward(const DataType& input, const DataType& /* gy */, DataType& g);
   
     OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     bool Deterministic() const { return deterministic; }
     bool& Deterministic() { return deterministic; }
   
     double Reward() const { return reward; }
     double& Reward() { return reward; }
   
     double StandardDeviation() const { return stdev; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     double stdev;
   
     double reward;
   
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     std::vector<arma::mat> moduleInputParameter;
   
     bool deterministic;
   }; // class ReinforceNormal
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "reinforce_normal_impl.hpp"
   
   #endif
