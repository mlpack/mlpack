
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_dists_gamma_distribution.hpp:

Program Listing for File gamma_distribution.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_dists_gamma_distribution.hpp>` (``/home/aakash/mlpack/src/mlpack/core/dists/gamma_distribution.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef _MLPACK_CORE_DISTRIBUTIONS_GAMMA_DISTRIBUTION_HPP
   #define _MLPACK_CORE_DISTRIBUTIONS_GAMMA_DISTRIBUTION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/math/random.hpp>
   
   namespace mlpack {
   namespace distribution {
   
   class GammaDistribution
   {
    public:
     GammaDistribution(const size_t dimensionality = 0);
   
     GammaDistribution(const arma::mat& data, const double tol = 1e-8);
   
     GammaDistribution(const arma::vec& alpha, const arma::vec& beta);
   
     ~GammaDistribution() {}
   
     void Train(const arma::mat& rdata, const double tol = 1e-8);
   
     void Train(const arma::mat& observations,
                const arma::vec& probabilities,
                const double tol = 1e-8);
   
     void Train(const arma::vec& logMeanxVec,
                const arma::vec& meanLogxVec,
                const arma::vec& meanxVec,
                const double tol = 1e-8);
   
     void Probability(const arma::mat& observations,
                      arma::vec& probabilities) const;
   
     double Probability(double x, const size_t dim) const;
   
     void LogProbability(const arma::mat& observations,
                         arma::vec& logProbabilities) const;
   
     double LogProbability(double x, const size_t dim) const;
   
     arma::vec Random() const;
   
     // Access to Gamma distribution parameters.
   
     double Alpha(const size_t dim) const { return alpha[dim]; }
     double& Alpha(const size_t dim) { return alpha[dim]; }
   
     double Beta(const size_t dim) const { return beta[dim]; }
     double& Beta(const size_t dim) { return beta[dim]; }
   
     size_t Dimensionality() const { return alpha.n_elem; }
   
    private:
     arma::vec alpha;
     arma::vec beta;
   
     inline bool Converged(const double aOld,
                           const double aNew,
                           const double tol);
   };
   
   } // namespace distribution
   } // namespace mlpack
   
   #endif
