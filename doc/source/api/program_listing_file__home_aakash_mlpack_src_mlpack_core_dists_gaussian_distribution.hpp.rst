
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_dists_gaussian_distribution.hpp:

Program Listing for File gaussian_distribution.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_dists_gaussian_distribution.hpp>` (``/home/aakash/mlpack/src/mlpack/core/dists/gaussian_distribution.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_HPP
   #define MLPACK_CORE_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace distribution {
   
   class GaussianDistribution
   {
    private:
     arma::vec mean;
     arma::mat covariance;
     arma::mat covLower;
     arma::mat invCov;
     double logDetCov;
   
     static const constexpr double log2pi = 1.83787706640934533908193770912475883;
   
    public:
     GaussianDistribution() : logDetCov(0.0) { /* nothing to do */ }
   
     GaussianDistribution(const size_t dimension) :
         mean(arma::zeros<arma::vec>(dimension)),
         covariance(arma::eye<arma::mat>(dimension, dimension)),
         covLower(arma::eye<arma::mat>(dimension, dimension)),
         invCov(arma::eye<arma::mat>(dimension, dimension)),
         logDetCov(0)
     { /* Nothing to do. */ }
   
     GaussianDistribution(const arma::vec& mean, const arma::mat& covariance);
   
     // TODO(stephentu): do we want a (arma::vec&&, arma::mat&&) ctor?
   
     size_t Dimensionality() const { return mean.n_elem; }
   
     double Probability(const arma::vec& observation) const
     {
       return exp(LogProbability(observation));
     }
   
     double LogProbability(const arma::vec& observation) const;
   
     void Probability(const arma::mat& x, arma::vec& probabilities) const
     {
       // Use LogProbability(), then transform the log-probabilities out of
       // logspace.
       arma::vec logProbs;
       LogProbability(x, logProbs);
       probabilities = arma::exp(logProbs);
     }
   
     void LogProbability(const arma::mat& x, arma::vec& logProbabilities) const
     {
       // Column i of 'diffs' is the difference between x.col(i) and the mean.
       arma::mat diffs = x;
       diffs.each_col() -= mean;
   
       // Now, we only want to calculate the diagonal elements of (diffs' * cov^-1
       // * diffs).  We just don't need any of the other elements.
       logProbabilities = -0.5 * x.n_rows * log2pi - 0.5 * logDetCov +
           sum(diffs % (-0.5 * invCov * diffs), 0).t();
     }
   
     arma::vec Random() const;
   
     void Train(const arma::mat& observations);
   
     void Train(const arma::mat& observations,
                const arma::vec& probabilities);
   
     const arma::vec& Mean() const { return mean; }
   
     arma::vec& Mean() { return mean; }
   
     const arma::mat& Covariance() const { return covariance; }
   
     void Covariance(const arma::mat& covariance);
   
     void Covariance(arma::mat&& covariance);
   
     const arma::mat& InvCov() const { return invCov; }
   
     double LogDetCov() const { return logDetCov; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       // We just need to serialize each of the members.
       ar(CEREAL_NVP(mean));
       ar(CEREAL_NVP(covariance));
       ar(CEREAL_NVP(covLower));
       ar(CEREAL_NVP(invCov));
       ar(CEREAL_NVP(logDetCov));
     }
   
    private:
     void FactorCovariance();
   };
   
   } // namespace distribution
   } // namespace mlpack
   
   #endif
