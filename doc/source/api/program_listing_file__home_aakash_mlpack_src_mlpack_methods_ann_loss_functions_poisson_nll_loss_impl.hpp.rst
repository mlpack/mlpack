
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_poisson_nll_loss_impl.hpp:

Program Listing for File poisson_nll_loss_impl.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_poisson_nll_loss_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/poisson_nll_loss_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_POISSON_NLL_LOSS_IMPL_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_POISSON_NLL_LOSS_IMPL_HPP
   
   
   // In case it hasn't yet been included.
   #include "poisson_nll_loss.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   PoissonNLLLoss<InputDataType, OutputDataType>::PoissonNLLLoss(
       const bool logInput,
       const bool full,
       const typename InputDataType::elem_type eps,
       const bool mean):
       logInput(logInput),
       full(full),
       eps(eps),
       mean(mean)
   {
     Log::Assert(eps >= 0, "Epsilon (eps) must be greater than or equal to zero.");
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   typename InputDataType::elem_type
   PoissonNLLLoss<InputDataType, OutputDataType>::Forward(
       const PredictionType& prediction,
       const TargetType& target)
   {
     PredictionType loss(arma::size(prediction));
   
     if (logInput)
       loss = arma::exp(prediction) - target % prediction;
     else
     {
       CheckProbs(prediction);
       loss = prediction - target % arma::log(prediction + eps);
     }
   
     if (full)
     {
       const auto mask = target > 1.0;
       const PredictionType approx = target % arma::log(target) - target
           + 0.5 * arma::log(2 * M_PI * target);
       loss.elem(arma::find(mask)) += approx.elem(arma::find(mask));
     }
   
     return mean ? arma::accu(loss) / loss.n_elem : arma::accu(loss);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType, typename LossType>
   void PoissonNLLLoss<InputDataType, OutputDataType>::Backward(
       const PredictionType& prediction,
       const TargetType& target,
       LossType& loss)
   {
     loss.set_size(size(prediction));
   
     if (logInput)
       loss = (arma::exp(prediction) - target);
     else
       loss = (1 - target / (prediction + eps));
   
     if (mean)
       loss = loss / loss.n_elem;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void PoissonNLLLoss<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(logInput));
     ar(CEREAL_NVP(full));
     ar(CEREAL_NVP(eps));
     ar(CEREAL_NVP(mean));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
