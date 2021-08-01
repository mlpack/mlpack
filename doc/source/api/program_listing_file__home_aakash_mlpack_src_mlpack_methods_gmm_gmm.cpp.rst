
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_gmm_gmm.cpp:

Program Listing for File gmm.cpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_gmm_gmm.cpp>` (``/home/aakash/mlpack/src/mlpack/methods/gmm/gmm.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include "gmm.hpp"
   #include <mlpack/core/math/log_add.hpp>
   
   namespace mlpack {
   namespace gmm {
   
   GMM::GMM(const size_t gaussians, const size_t dimensionality) :
       gaussians(gaussians),
       dimensionality(dimensionality),
       dists(gaussians, distribution::GaussianDistribution(dimensionality)),
       weights(gaussians)
   {
     // Set equal weights.  Technically this model is still valid, but only barely.
     weights.fill(1.0 / gaussians);
   }
   
   // Copy constructor for when the other GMM uses the same fitting type.
   GMM::GMM(const GMM& other) :
       gaussians(other.Gaussians()),
       dimensionality(other.dimensionality),
       dists(other.dists),
       weights(other.weights) { /* Nothing to do. */ }
   
   GMM& GMM::operator=(const GMM& other)
   {
     gaussians = other.gaussians;
     dimensionality = other.dimensionality;
     dists = other.dists;
     weights = other.weights;
   
     return *this;
   }
   
   double GMM::LogProbability(const arma::vec& observation) const
   {
     // Sum the probability for each Gaussian in our mixture (and we have to
     // multiply by the prior for each Gaussian too).
     double sum = -std::numeric_limits<double>::infinity();
     for (size_t i = 0; i < gaussians; ++i)
       sum = math::LogAdd(sum, log(weights[i]) +
           dists[i].LogProbability(observation));
   
     return sum;
   }
   
   void GMM::LogProbability(const arma::mat& observation,
                            arma::vec& logProbs) const
   {
     // Sum the probability for each Gaussian in our mixture (and we have to
     // multiply by the prior for each Gaussian too).
     logProbs.set_size(observation.n_cols);
   
     // Store log-probability value in a matrix.
     arma::mat logProb(observation.n_cols, gaussians);
   
     // Assign value to the matrix.
     for (size_t i = 0; i < gaussians; i++)
     {
       arma::vec temp(logProb.colptr(i), observation.n_cols, false, true);
       dists[i].LogProbability(observation, temp);
     }
   
     // Save log(weights) as a vector.
     arma::vec logWeights = arma::log(weights);
   
     // Compute log-probability.
     logProb += repmat(logWeights.t(), logProb.n_rows, 1);
     math::LogSumExp(logProb, logProbs);
   }
   
   double GMM::Probability(const arma::vec& observation) const
   {
     return exp(LogProbability(observation));
   }
   
   void GMM::Probability(const arma::mat& observation,
                         arma::vec& probs) const
   {
     LogProbability(observation, probs);
     probs = exp(probs);
   }
   
   
   double GMM::LogProbability(const arma::vec& observation,
                           const size_t component) const
   {
     // We are only considering one Gaussian component -- so we only need to call
     // Probability() once.  We do consider the prior probability!
     return log(weights[component]) + dists[component].LogProbability(observation);
   }
   
   double GMM::Probability(const arma::vec& observation,
                           const size_t component) const
   {
     return exp(LogProbability(observation, component));
   }
   
   arma::vec GMM::Random() const
   {
     // Determine which Gaussian it will be coming from.
     double gaussRand = math::Random();
     size_t gaussian = 0;
   
     double sumProb = 0;
     for (size_t g = 0; g < gaussians; g++)
     {
       sumProb += weights(g);
       if (gaussRand <= sumProb)
       {
         gaussian = g;
         break;
       }
     }
   
     arma::mat cholDecomp;
     if (!arma::chol(cholDecomp, dists[gaussian].Covariance()))
     {
       Log::Fatal << "Cholesky decomposition failed." << std::endl;
     }
     return trans(cholDecomp) *
         arma::randn<arma::vec>(dimensionality) + dists[gaussian].Mean();
   }
   
   void GMM::Classify(const arma::mat& observations,
                      arma::Row<size_t>& labels) const
   {
     // This is not the best way to do this!
   
     // We should not have to fill this with values, because each one should be
     // overwritten.
     labels.set_size(observations.n_cols);
     for (size_t i = 0; i < observations.n_cols; ++i)
     {
       // Find maximum probability component.
       double probability = -std::numeric_limits<double>::infinity();
       for (size_t j = 0; j < gaussians; ++j)
       {
         // We have to use LogProbability() otherwise Probability() would overflow
         // easily.
         double newProb = LogProbability(observations.unsafe_col(i), j);
         if (newProb >= probability)
         {
           probability = newProb;
           labels[i] = j;
         }
       }
     }
   }
   
   double GMM::LogLikelihood(
       const arma::mat& data,
       const std::vector<distribution::GaussianDistribution>& distsL,
       const arma::vec& weightsL) const
   {
     double loglikelihood = 0;
     arma::vec logPhis;
     arma::mat logLikelihoods(gaussians, data.n_cols);
   
     // It has to be LogProbability() otherwise Probability() would overflow easily
     for (size_t i = 0; i < gaussians; ++i)
     {
       distsL[i].LogProbability(data, logPhis);
       logLikelihoods.row(i) = log(weightsL(i)) + trans(logPhis);
     }
   
     // Now sum over every point.
     for (size_t j = 0; j < data.n_cols; ++j)
       loglikelihood += mlpack::math::AccuLog(logLikelihoods.col(j));
     return loglikelihood;
   }
   
   } // namespace gmm
   } // namespace mlpack
