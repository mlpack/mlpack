
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_gmm_gmm_impl.hpp:

Program Listing for File gmm_impl.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_gmm_gmm_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/gmm/gmm_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_GMM_GMM_IMPL_HPP
   #define MLPACK_METHODS_GMM_GMM_IMPL_HPP
   
   // In case it hasn't already been included.
   #include "gmm.hpp"
   
   namespace mlpack {
   namespace gmm {
   
   template<typename FittingType>
   double GMM::Train(const arma::mat& observations,
                     const size_t trials,
                     const bool useExistingModel,
                     FittingType fitter)
   {
     double bestLikelihood; // This will be reported later.
   
     // We don't need to store temporary models if we are only doing one trial.
     if (trials == 1)
     {
       // Train the model.  The user will have been warned earlier if the GMM was
       // initialized with no parameters (0 gaussians, dimensionality of 0).
       fitter.Estimate(observations, dists, weights, useExistingModel);
       bestLikelihood = LogLikelihood(observations, dists, weights);
     }
     else
     {
       if (trials == 0)
         return -DBL_MAX; // It's what they asked for...
   
       // If each trial must start from the same initial location, we must save it.
       std::vector<distribution::GaussianDistribution> distsOrig;
       arma::vec weightsOrig;
       if (useExistingModel)
       {
         distsOrig = dists;
         weightsOrig = weights;
       }
   
       // We need to keep temporary copies.  We'll do the first training into the
       // actual model position, so that if it's the best we don't need to copy it.
       fitter.Estimate(observations, dists, weights, useExistingModel);
   
       bestLikelihood = LogLikelihood(observations, dists, weights);
   
       Log::Info << "GMM::Train(): Log-likelihood of trial 0 is "
           << bestLikelihood << "." << std::endl;
   
       // Now the temporary model.
       std::vector<distribution::GaussianDistribution> distsTrial(gaussians,
           distribution::GaussianDistribution(dimensionality));
       arma::vec weightsTrial(gaussians);
   
       for (size_t trial = 1; trial < trials; ++trial)
       {
         if (useExistingModel)
         {
           distsTrial = distsOrig;
           weightsTrial = weightsOrig;
         }
   
         fitter.Estimate(observations, distsTrial, weightsTrial, useExistingModel);
   
         // Check to see if the log-likelihood of this one is better.
         double newLikelihood = LogLikelihood(observations, distsTrial,
             weightsTrial);
   
         Log::Info << "GMM::Train(): Log-likelihood of trial " << trial << " is "
             << newLikelihood << "." << std::endl;
   
         if (newLikelihood > bestLikelihood)
         {
           // Save new likelihood and copy new model.
           bestLikelihood = newLikelihood;
   
           dists = distsTrial;
           weights = weightsTrial;
         }
       }
     }
   
     // Report final log-likelihood and return it.
     Log::Info << "GMM::Train(): log-likelihood of trained GMM is "
         << bestLikelihood << "." << std::endl;
     return bestLikelihood;
   }
   
   template<typename FittingType>
   double GMM::Train(const arma::mat& observations,
                     const arma::vec& probabilities,
                     const size_t trials,
                     const bool useExistingModel,
                     FittingType fitter)
   {
     double bestLikelihood; // This will be reported later.
   
     // We don't need to store temporary models if we are only doing one trial.
     if (trials == 1)
     {
       // Train the model.  The user will have been warned earlier if the GMM was
       // initialized with no parameters (0 gaussians, dimensionality of 0).
       fitter.Estimate(observations, probabilities, dists, weights,
           useExistingModel);
       bestLikelihood = LogLikelihood(observations, dists, weights);
     }
     else
     {
       if (trials == 0)
         return -DBL_MAX; // It's what they asked for...
   
       // If each trial must start from the same initial location, we must save it.
       std::vector<distribution::GaussianDistribution> distsOrig;
       arma::vec weightsOrig;
       if (useExistingModel)
       {
         distsOrig = dists;
         weightsOrig = weights;
       }
   
       // We need to keep temporary copies.  We'll do the first training into the
       // actual model position, so that if it's the best we don't need to copy it.
       fitter.Estimate(observations, probabilities, dists, weights,
           useExistingModel);
   
       bestLikelihood = LogLikelihood(observations, dists, weights);
   
       Log::Debug << "GMM::Train(): Log-likelihood of trial 0 is "
           << bestLikelihood << "." << std::endl;
   
       // Now the temporary model.
       std::vector<distribution::GaussianDistribution> distsTrial(gaussians,
           distribution::GaussianDistribution(dimensionality));
       arma::vec weightsTrial(gaussians);
   
       for (size_t trial = 1; trial < trials; ++trial)
       {
         if (useExistingModel)
         {
           distsTrial = distsOrig;
           weightsTrial = weightsOrig;
         }
   
         fitter.Estimate(observations, probabilities, distsTrial, weightsTrial,
             useExistingModel);
   
         // Check to see if the log-likelihood of this one is better.
         double newLikelihood = LogLikelihood(observations, distsTrial,
             weightsTrial);
   
         Log::Debug << "GMM::Train(): Log-likelihood of trial " << trial << " is "
             << newLikelihood << "." << std::endl;
   
         if (newLikelihood > bestLikelihood)
         {
           // Save new likelihood and copy new model.
           bestLikelihood = newLikelihood;
   
           dists = distsTrial;
           weights = weightsTrial;
         }
       }
     }
   
     // Report final log-likelihood and return it.
     Log::Info << "GMM::Train(): log-likelihood of trained GMM is "
         << bestLikelihood << "." << std::endl;
     return bestLikelihood;
   }
   
   template<typename Archive>
   void GMM::serialize(Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(gaussians));
     ar(CEREAL_NVP(dimensionality));
   
     // Load (or save) the gaussians.  Not going to use the default std::vector
     // serialize here because it won't call out correctly to serialize() for each
     // Gaussian distribution.
     if (cereal::is_loading<Archive>())
       dists.resize(gaussians);
   
     ar(CEREAL_NVP(dists));
   
     ar(CEREAL_NVP(weights));
   }
   
   } // namespace gmm
   } // namespace mlpack
   
   #endif
   
