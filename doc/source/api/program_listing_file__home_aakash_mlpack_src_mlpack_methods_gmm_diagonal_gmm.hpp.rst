
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_gmm_diagonal_gmm.hpp:

Program Listing for File diagonal_gmm.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_gmm_diagonal_gmm.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/gmm/diagonal_gmm.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_GMM_DIAGONAL_GMM_HPP
   #define MLPACK_METHODS_GMM_DIAGONAL_GMM_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/dists/diagonal_gaussian_distribution.hpp>
   
   // This is the default fitting method class.
   #include "em_fit.hpp"
   
   // This is the default covariance matrix constraint.
   #include "diagonal_constraint.hpp"
   
   namespace mlpack {
   namespace gmm  {
   
   class DiagonalGMM
   {
    private:
     size_t gaussians;
     size_t dimensionality;
   
     std::vector<distribution::DiagonalGaussianDistribution> dists;
   
     arma::vec weights;
   
    public:
     DiagonalGMM() :
         gaussians(0),
         dimensionality(0)
     {
       // Warn the user.  They probably don't want to do this.  If this
       // constructor is being used (because it is required by some template
       // classes), the user should know that it is potentially dangerous.
       Log::Debug << "DiagonalGMM::DiagonalGMM(): no parameters given;"
           "Estimate() may fail " << "unless parameters are set." << std::endl;
     }
   
     DiagonalGMM(const size_t gaussians, const size_t dimensionality);
   
     DiagonalGMM(const std::vector<distribution::DiagonalGaussianDistribution>&
         dists, const arma::vec& weights) :
         gaussians(dists.size()),
         dimensionality((!dists.empty()) ? dists[0].Mean().n_elem : 0),
         dists(dists),
         weights(weights) { /* Nothing to do. */ }
   
     DiagonalGMM(const DiagonalGMM& other);
   
     DiagonalGMM& operator=(const DiagonalGMM& other);
   
     size_t Gaussians() const { return gaussians; }
     size_t Dimensionality() const { return dimensionality; }
   
     const distribution::DiagonalGaussianDistribution& Component(size_t i) const
     {
       return dists[i];
     }
   
     distribution::DiagonalGaussianDistribution& Component(size_t i)
     {
       return dists[i];
     }
   
     const arma::vec& Weights() const { return weights; }
     arma::vec& Weights() { return weights; }
   
     double Probability(const arma::vec& observation) const;
   
     void Probability(const arma::mat& observation, arma::vec& probs) const;
   
     double LogProbability(const arma::vec& observation) const;
   
     void LogProbability(const arma::mat& observation, arma::vec& logProbs) const;
   
     double Probability(const arma::vec& observation,
                        const size_t component) const;
   
     double LogProbability(const arma::vec& observation,
                           const size_t component) const;
     arma::vec Random() const;
   
     template<typename FittingType = EMFit<kmeans::KMeans<>, DiagonalConstraint,
         distribution::DiagonalGaussianDistribution>>
     double Train(const arma::mat& observations,
                  const size_t trials = 1,
                  const bool useExistingModel = false,
                  FittingType fitter = FittingType());
   
     template<typename FittingType = EMFit<kmeans::KMeans<>, DiagonalConstraint,
         distribution::DiagonalGaussianDistribution>>
     double Train(const arma::mat& observations,
                  const arma::vec& probabilities,
                  const size_t trials = 1,
                  const bool useExistingModel = false,
                  FittingType fitter = FittingType());
   
     void Classify(const arma::mat& observations,
                   arma::Row<size_t>& labels) const;
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     double LogLikelihood(
         const arma::mat& observations,
         const std::vector<distribution::DiagonalGaussianDistribution>& dists,
         const arma::vec& weights) const;
   };
   
   } // namespace gmm
   } // namespace mlpack
   
   // Include implementation.
   #include "diagonal_gmm_impl.hpp"
   
   #endif // MLPACK_METHODS_GMM_DIAGONAL_GMM_HPP
