
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_poisson_nll_loss.hpp:

Program Listing for File poisson_nll_loss.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_poisson_nll_loss.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/poisson_nll_loss.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_POISSON_NLL_LOSS_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_POISSON_NLL_LOSS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
     typename InputDataType = arma::mat,
     typename OutputDataType = arma::mat
   >
   class PoissonNLLLoss
   {
    public:
     PoissonNLLLoss(const bool logInput = true,
                    const bool full = false,
                    const typename InputDataType::elem_type eps = 1e-08,
                    const bool mean = true);
   
     template<typename PredictionType, typename TargetType>
     typename InputDataType::elem_type Forward(const PredictionType& prediction,
                                               const TargetType& target);
   
     template<typename PredictionType, typename TargetType, typename LossType>
     void Backward(const PredictionType& prediction,
                   const TargetType& target,
                   LossType& loss);
   
     InputDataType& InputParameter() const { return inputParameter; }
     InputDataType& InputParameter() { return inputParameter; }
   
     OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     bool LogInput() const { return logInput; }
     bool& LogInput() { return logInput; }
   
     bool Full() const { return full; }
     bool& Full() { return full; }
   
     typename InputDataType::elem_type Eps() const { return eps; }
     typename InputDataType::elem_type& Eps() { return eps; }
   
     bool Mean() const { return mean; }
     bool& Mean() { return mean; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     template<typename eT>
     void CheckProbs(const arma::Mat<eT>& probs)
     {
       for (size_t i = 0; i < probs.size(); ++i)
       {
         if (probs[i] > 1.0 || probs[i] < 0.0)
           Log::Fatal << "Probabilities cannot be greater than 1 "
                      << "or smaller than 0." << std::endl;
       }
     }
   
     InputDataType inputParameter;
   
     OutputDataType outputParameter;
   
     bool logInput;
   
     // approximation term.
     bool full;
   
     typename InputDataType::elem_type eps;
   
     bool mean;
   }; // class PoissonNLLLoss
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "poisson_nll_loss_impl.hpp"
   
   #endif
