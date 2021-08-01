
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_gmm_em_fit.hpp:

Program Listing for File em_fit.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_gmm_em_fit.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/gmm/em_fit.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_GMM_EM_FIT_HPP
   #define MLPACK_METHODS_GMM_EM_FIT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/dists/gaussian_distribution.hpp>
   #include <mlpack/core/dists/diagonal_gaussian_distribution.hpp>
   
   // Default clustering mechanism.
   #include <mlpack/methods/kmeans/kmeans.hpp>
   // Default covariance matrix constraint.
   #include "positive_definite_constraint.hpp"
   
   namespace mlpack {
   namespace gmm {
   
   template<typename InitialClusteringType = kmeans::KMeans<>,
            typename CovarianceConstraintPolicy = PositiveDefiniteConstraint,
            typename Distribution = distribution::GaussianDistribution>
   class EMFit
   {
    public:
     EMFit(const size_t maxIterations = 300,
           const double tolerance = 1e-10,
           InitialClusteringType clusterer = InitialClusteringType(),
           CovarianceConstraintPolicy constraint = CovarianceConstraintPolicy());
   
     void Estimate(const arma::mat& observations,
                   std::vector<Distribution>& dists,
                   arma::vec& weights,
                   const bool useInitialModel = false);
   
     void Estimate(const arma::mat& observations,
                   const arma::vec& probabilities,
                   std::vector<Distribution>& dists,
                   arma::vec& weights,
                   const bool useInitialModel = false);
   
     const InitialClusteringType& Clusterer() const { return clusterer; }
     InitialClusteringType& Clusterer() { return clusterer; }
   
     const CovarianceConstraintPolicy& Constraint() const { return constraint; }
     CovarianceConstraintPolicy& Constraint() { return constraint; }
   
     size_t MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
     double Tolerance() const { return tolerance; }
     double& Tolerance() { return tolerance; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
    private:
     void InitialClustering(
         const arma::mat& observations,
         std::vector<Distribution>& dists,
         arma::vec& weights);
   
     double LogLikelihood(
         const arma::mat& data,
         const std::vector<Distribution>& dists,
         const arma::vec& weights) const;
   
     void ArmadilloGMMWrapper(
         const arma::mat& observations,
         std::vector<Distribution>& dists,
         arma::vec& weights,
         const bool useInitialModel);
   
     size_t maxIterations;
     double tolerance;
     InitialClusteringType clusterer;
     CovarianceConstraintPolicy constraint;
   };
   
   } // namespace gmm
   } // namespace mlpack
   
   // Include implementation.
   #include "em_fit_impl.hpp"
   
   #endif
