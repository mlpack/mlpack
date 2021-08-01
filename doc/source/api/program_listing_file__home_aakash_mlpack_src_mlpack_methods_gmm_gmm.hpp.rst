
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_gmm_gmm.hpp:

Program Listing for File gmm.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_gmm_gmm.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/gmm/gmm.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_MOG_MOG_EM_HPP
   #define MLPACK_METHODS_MOG_MOG_EM_HPP
   
   #include <mlpack/prereqs.hpp>
   
   // This is the default fitting method class.
   #include "em_fit.hpp"
   
   namespace mlpack {
   namespace gmm  {
   
   class GMM
   {
    private:
     size_t gaussians;
     size_t dimensionality;
   
     std::vector<distribution::GaussianDistribution> dists;
   
     arma::vec weights;
   
    public:
     GMM() :
         gaussians(0),
         dimensionality(0)
     {
       // Warn the user.  They probably don't want to do this.  If this constructor
       // is being used (because it is required by some template classes), the user
       // should know that it is potentially dangerous.
       Log::Debug << "GMM::GMM(): no parameters given; Estimate() may fail "
           << "unless parameters are set." << std::endl;
     }
   
     GMM(const size_t gaussians, const size_t dimensionality);
   
     GMM(const std::vector<distribution::GaussianDistribution> & dists,
         const arma::vec& weights) :
         gaussians(dists.size()),
         dimensionality((!dists.empty()) ? dists[0].Mean().n_elem : 0),
         dists(dists),
         weights(weights) { /* Nothing to do. */ }
   
     GMM(const GMM& other);
   
     GMM& operator=(const GMM& other);
   
     size_t Gaussians() const { return gaussians; }
     size_t Dimensionality() const { return dimensionality; }
   
     const distribution::GaussianDistribution& Component(size_t i) const {
         return dists[i]; }
     distribution::GaussianDistribution& Component(size_t i) { return dists[i]; }
   
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
   
     template<typename FittingType = EMFit<>>
     double Train(const arma::mat& observations,
                  const size_t trials = 1,
                  const bool useExistingModel = false,
                  FittingType fitter = FittingType());
   
     template<typename FittingType = EMFit<>>
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
         const arma::mat& dataPoints,
         const std::vector<distribution::GaussianDistribution>& distsL,
         const arma::vec& weights) const;
   };
   
   } // namespace gmm
   } // namespace mlpack
   
   // Include implementation.
   #include "gmm_impl.hpp"
   
   #endif
