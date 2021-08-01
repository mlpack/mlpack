
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kde_kde_rules.hpp:

Program Listing for File kde_rules.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kde_kde_rules.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kde/kde_rules.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KDE_RULES_HPP
   #define MLPACK_METHODS_KDE_RULES_HPP
   
   #include <mlpack/core/tree/traversal_info.hpp>
   
   namespace mlpack {
   namespace kde {
   
   template<typename MetricType, typename KernelType, typename TreeType>
   class KDERules
   {
    public:
     KDERules(const arma::mat& referenceSet,
              const arma::mat& querySet,
              arma::vec& densities,
              const double relError,
              const double absError,
              const double mcProb,
              const size_t initialSampleSize,
              const double mcAccessCoef,
              const double mcBreakCoef,
              MetricType& metric,
              KernelType& kernel,
              const bool monteCarlo,
              const bool sameSet);
   
     double BaseCase(const size_t queryIndex, const size_t referenceIndex);
   
     double Score(const size_t queryIndex, TreeType& referenceNode);
   
     double Rescore(const size_t queryIndex,
                    TreeType& referenceNode,
                    const double oldScore) const;
   
     double Score(TreeType& queryNode, TreeType& referenceNode);
   
     double Rescore(TreeType& queryNode,
                    TreeType& referenceNode,
                    const double oldScore) const;
   
     typedef typename tree::TraversalInfo<TreeType> TraversalInfoType;
   
     const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
   
     TraversalInfoType& TraversalInfo() { return traversalInfo; }
   
     size_t BaseCases() const { return baseCases; }
   
     size_t Scores() const { return scores; }
   
     size_t MinimumBaseCases() const { return 0; }
   
    private:
     double EvaluateKernel(const size_t queryIndex,
                           const size_t referenceIndex) const;
   
     double EvaluateKernel(const arma::vec& query,
                           const arma::vec& reference) const;
   
     double CalculateAlpha(TreeType* node);
   
     const arma::mat& referenceSet;
   
     const arma::mat& querySet;
   
     arma::vec& densities;
   
     const double absError;
   
     const double relError;
   
     const double mcBeta;
   
     const size_t initialSampleSize;
   
     const double mcAccessCoef;
   
     const double mcBreakCoef;
   
     MetricType& metric;
   
     KernelType& kernel;
   
     const bool monteCarlo;
   
     arma::vec accumMCAlpha;
   
     arma::vec accumError;
   
     const bool sameSet;
   
     constexpr static bool kernelIsGaussian =
         std::is_same<KernelType, kernel::GaussianKernel>::value;
   
     const double absErrorTol;
   
     size_t lastQueryIndex;
   
     size_t lastReferenceIndex;
   
     TraversalInfoType traversalInfo;
   
     size_t baseCases;
   
     size_t scores;
   };
   
   template<typename TreeType>
   class KDECleanRules
   {
    public:
     KDECleanRules() { /* Nothing to do. */ }
   
     double BaseCase(const size_t /* queryIndex */, const size_t /* refIndex */);
   
     double Score(const size_t /* queryIndex */, TreeType& referenceNode);
   
     double Rescore(const size_t /* queryIndex */,
                    TreeType& /* referenceNode */,
                    const double oldScore) const { return oldScore; }
   
     double Score(TreeType& queryNode, TreeType& referenceNode);
   
     double Rescore(TreeType& /* queryNode */,
                    TreeType& /* referenceNode*/ ,
                    const double oldScore) const { return oldScore; }
   
     typedef typename tree::TraversalInfo<TreeType> TraversalInfoType;
   
     const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
   
     TraversalInfoType& TraversalInfo() { return traversalInfo; }
   
    private:
     TraversalInfoType traversalInfo;
   };
   
   } // namespace kde
   } // namespace mlpack
   
   // Include implementation.
   #include "kde_rules_impl.hpp"
   
   #endif
