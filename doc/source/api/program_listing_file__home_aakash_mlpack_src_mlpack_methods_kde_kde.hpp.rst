
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kde_kde.hpp:

Program Listing for File kde.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kde_kde.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kde/kde.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KDE_KDE_HPP
   #define MLPACK_METHODS_KDE_KDE_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/tree/binary_space_tree.hpp>
   
   #include "kde_stat.hpp"
   
   namespace mlpack {
   namespace kde  {
   
   enum KDEMode
   {
     DUAL_TREE_MODE,
     SINGLE_TREE_MODE
   };
   
   struct KDEDefaultParams
   {
     static constexpr double relError = 0.05;
   
     static constexpr double absError = 0;
   
     static constexpr KDEMode mode = KDEMode::DUAL_TREE_MODE;
   
     static constexpr bool monteCarlo = false;
   
     static constexpr double mcProb = 0.95;
   
     static constexpr size_t initialSampleSize = 100;
   
     static constexpr double mcEntryCoef = 3;
   
     static constexpr double mcBreakCoef = 0.4;
   };
   
   template<typename KernelType = kernel::GaussianKernel,
            typename MetricType = mlpack::metric::EuclideanDistance,
            typename MatType = arma::mat,
            template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType = tree::KDTree,
            template<typename RuleType> class DualTreeTraversalType =
                TreeType<MetricType,
                         kde::KDEStat,
                         MatType>::template DualTreeTraverser,
            template<typename RuleType> class SingleTreeTraversalType =
                TreeType<MetricType,
                         kde::KDEStat,
                         MatType>::template SingleTreeTraverser>
   class KDE
   {
    public:
     typedef TreeType<MetricType, kde::KDEStat, MatType> Tree;
   
     KDE(const double relError = KDEDefaultParams::relError,
         const double absError = KDEDefaultParams::absError,
         KernelType kernel = KernelType(),
         const KDEMode mode = KDEDefaultParams::mode,
         MetricType metric = MetricType(),
         const bool monteCarlo = KDEDefaultParams::monteCarlo,
         const double mcProb = KDEDefaultParams::mcProb,
         const size_t initialSampleSize = KDEDefaultParams::initialSampleSize,
         const double mcEntryCoef = KDEDefaultParams::mcEntryCoef,
         const double mcBreakCoef = KDEDefaultParams::mcBreakCoef);
   
     KDE(const KDE& other);
   
     KDE(KDE&& other);
   
     KDE& operator=(const KDE& other);
   
     KDE& operator=(KDE&& other);
   
     ~KDE();
   
     void Train(MatType referenceSet);
   
     void Train(Tree* referenceTree, std::vector<size_t>* oldFromNewReferences);
   
     void Evaluate(MatType querySet, arma::vec& estimations);
   
     void Evaluate(Tree* queryTree,
                   const std::vector<size_t>& oldFromNewQueries,
                   arma::vec& estimations);
   
     void Evaluate(arma::vec& estimations);
   
     const KernelType& Kernel() const { return kernel; }
   
     KernelType& Kernel() { return kernel; }
   
     const MetricType& Metric() const { return metric; }
   
     MetricType& Metric() { return metric; }
   
     Tree* ReferenceTree() { return referenceTree; }
   
     double RelativeError() const { return relError; }
   
     void RelativeError(const double newError);
   
     double AbsoluteError() const { return absError; }
   
     void AbsoluteError(const double newError);
   
     bool OwnsReferenceTree() const { return ownsReferenceTree; }
   
     bool IsTrained() const { return trained; }
   
     KDEMode Mode() const { return mode; }
   
     KDEMode& Mode() { return mode; }
   
     bool MonteCarlo() const { return monteCarlo; }
   
     bool& MonteCarlo() { return monteCarlo; }
   
     double MCProb() const { return mcProb; }
   
     void MCProb(const double newProb);
   
     size_t MCInitialSampleSize() const { return initialSampleSize; }
   
     size_t& MCInitialSampleSize() { return initialSampleSize; }
   
     double MCEntryCoef() const { return mcEntryCoef; }
   
     void MCEntryCoef(const double newCoef);
   
     double MCBreakCoef() const { return mcBreakCoef; }
   
     void MCBreakCoef(const double newCoef);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
    private:
     KernelType kernel;
   
     MetricType metric;
   
     Tree* referenceTree;
   
     std::vector<size_t>* oldFromNewReferences;
   
     double relError;
   
     double absError;
   
     bool ownsReferenceTree;
   
     bool trained;
   
     KDEMode mode;
   
     bool monteCarlo;
   
     double mcProb;
   
     size_t initialSampleSize;
   
     double mcEntryCoef;
   
     double mcBreakCoef;
   
     static void CheckErrorValues(const double relError, const double absError);
   
     static void RearrangeEstimations(const std::vector<size_t>& oldFromNew,
                                      arma::vec& estimations);
   };
   
   } // namespace kde
   } // namespace mlpack
   
   // Include implementation.
   #include "kde_impl.hpp"
   
   #endif // MLPACK_METHODS_KDE_KDE_HPP
