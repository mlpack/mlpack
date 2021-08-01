
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_dists_diagonal_gaussian_distribution.hpp:

Program Listing for File diagonal_gaussian_distribution.hpp
===========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_dists_diagonal_gaussian_distribution.hpp>` (``/home/aakash/mlpack/src/mlpack/core/dists/diagonal_gaussian_distribution.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DISTRIBUTIONS_DIAGONAL_GAUSSIAN_DISTRIBUTION_HPP
   #define MLPACK_CORE_DISTRIBUTIONS_DIAGONAL_GAUSSIAN_DISTRIBUTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace distribution {
   
   class DiagonalGaussianDistribution
   {
    private:
     arma::vec mean;
     arma::vec covariance;
     arma::vec invCov;
     double logDetCov;
   
     static const constexpr double log2pi = 1.83787706640934533908193770912475883;
   
    public:
     DiagonalGaussianDistribution() : logDetCov(0.0) { /* nothing to do. */ }
   
     DiagonalGaussianDistribution(const size_t dimension) :
         mean(arma::zeros<arma::vec>(dimension)),
         covariance(arma::ones<arma::vec>(dimension)),
         invCov(arma::ones<arma::vec>(dimension)),
         logDetCov(0)
     { /* Nothing to do. */ }
   
     DiagonalGaussianDistribution(const arma::vec& mean,
                                  const arma::vec& covariance);
   
     size_t Dimensionality() const { return mean.n_elem; }
   
     double Probability(const arma::vec& observation) const
     {
       return exp(LogProbability(observation));
     }
   
     double LogProbability(const arma::vec& observation) const;
   
     void Probability(const arma::mat& x, arma::vec& probabilities) const
     {
       arma::vec logProbabilities;
       LogProbability(x, logProbabilities);
       probabilities = arma::exp(logProbabilities);
     }
   
     void LogProbability(const arma::mat& observations,
                         arma::vec& logProbabilities) const;
   
     arma::vec Random() const;
   
     void Train(const arma::mat& observations);
   
     void Train(const arma::mat& observations,
                const arma::vec& probabilities);
   
     const arma::vec& Mean() const { return mean; }
   
     arma::vec& Mean() { return mean; }
   
     const arma::vec& Covariance() const { return covariance; }
   
     void Covariance(const arma::vec& covariance);
   
     void Covariance(arma::vec&& covariance);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       // We just need to serialize each of the members.
       ar(CEREAL_NVP(mean));
       ar(CEREAL_NVP(covariance));
       ar(CEREAL_NVP(invCov));
       ar(CEREAL_NVP(logDetCov));
     }
   };
   
   } // namespace distribution
   } // namespace mlpack
   
   #endif
