
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kde_kde_model.hpp:

Program Listing for File kde_model.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kde_kde_model.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kde/kde_model.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KDE_MODEL_HPP
   #define MLPACK_METHODS_KDE_MODEL_HPP
   
   // Include trees.
   #include <mlpack/core/tree/binary_space_tree.hpp>
   #include <mlpack/core/tree/octree.hpp>
   #include <mlpack/core/tree/cover_tree.hpp>
   #include <mlpack/core/tree/rectangle_tree.hpp>
   
   // Include core.
   #include <mlpack/core.hpp>
   
   // Remaining includes.
   #include "kde.hpp"
   
   namespace mlpack {
   namespace kde {
   
   class KernelNormalizer
   {
    private:
     // SFINAE check if Normalizer function is present.
     HAS_MEM_FUNC(Normalizer, HasNormalizer);
   
    public:
     template<typename KernelType>
     static void ApplyNormalizer(
         KernelType& /* kernel */,
         const size_t /* dimension */,
         arma::vec& /* estimations */,
         const typename std::enable_if<
             !HasNormalizer<KernelType, double(KernelType::*)(size_t)>::value>::
             type* = 0)
     { return; }
   
     template<typename KernelType>
     static void ApplyNormalizer(
         KernelType& kernel,
         const size_t dimension,
         arma::vec& estimations,
         const typename std::enable_if<
             HasNormalizer<KernelType, double(KernelType::*)(size_t)>::value>::
             type* = 0)
     {
       estimations /= kernel.Normalizer(dimension);
     }
   };
   
   class KDEWrapperBase
   {
    public:
     KDEWrapperBase() { }
   
     virtual KDEWrapperBase* Clone() const = 0;
   
     virtual ~KDEWrapperBase() { }
   
     virtual void Bandwidth(const double bw) = 0;
   
     virtual void RelativeError(const double relError) = 0;
   
     virtual void AbsoluteError(const double absError) = 0;
   
     virtual bool MonteCarlo() const = 0;
     virtual bool& MonteCarlo() = 0;
   
     virtual void MCProb(const double mcProb) = 0;
   
     virtual size_t MCInitialSampleSize() const = 0;
     virtual size_t& MCInitialSampleSize() = 0;
   
     virtual void MCEntryCoef(const double entryCoef) = 0;
   
     virtual void MCBreakCoef(const double breakCoef) = 0;
   
     virtual KDEMode Mode() const = 0;
     virtual KDEMode& Mode() = 0;
   
     virtual void Train(arma::mat&& referenceSet) = 0;
   
     virtual void Evaluate(arma::mat&& querySet,
                           arma::vec& estimates) = 0;
   
     virtual void Evaluate(arma::vec& estimates) = 0;
   };
   
   template<typename KernelType,
            template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType>
   class KDEWrapper : public KDEWrapperBase
   {
    public:
     KDEWrapper(const double relError,
                const double absError,
                const KernelType& kernel) :
         kde(relError, absError, kernel)
     {
       // Nothing left to do.
     }
   
     virtual KDEWrapper* Clone() const { return new KDEWrapper(*this); }
   
     virtual ~KDEWrapper() { }
   
     virtual void Bandwidth(const double bw) { kde.Kernel() = KernelType(bw); }
   
     virtual void RelativeError(const double eps) { kde.RelativeError(eps); }
   
     virtual void AbsoluteError(const double eps) { kde.AbsoluteError(eps); }
   
     virtual bool MonteCarlo() const { return kde.MonteCarlo(); }
     virtual bool& MonteCarlo() { return kde.MonteCarlo(); }
   
     virtual void MCProb(const double mcProb) { kde.MCProb(mcProb); }
   
     virtual size_t MCInitialSampleSize() const
     {
       return kde.MCInitialSampleSize();
     }
     virtual size_t& MCInitialSampleSize()
     {
       return kde.MCInitialSampleSize();
     }
   
     virtual void MCEntryCoef(const double e) { kde.MCEntryCoef(e); }
   
     virtual void MCBreakCoef(const double b) { kde.MCBreakCoef(b); }
   
     virtual KDEMode Mode() const { return kde.Mode(); }
     virtual KDEMode& Mode() { return kde.Mode(); }
   
     virtual void Train(arma::mat&& referenceSet);
   
     virtual void Evaluate(arma::mat&& querySet,
                           arma::vec& estimates);
   
     virtual void Evaluate(arma::vec& estimates);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(kde));
     }
   
    protected:
     typedef KDE<KernelType,
                 metric::EuclideanDistance,
                 arma::mat,
                 TreeType> KDEType;
   
     KDEType kde;
   };
   
   class KDEModel
   {
    public:
     enum TreeTypes
     {
       KD_TREE,
       BALL_TREE,
       COVER_TREE,
       OCTREE,
       R_TREE
     };
   
     enum KernelTypes
     {
       GAUSSIAN_KERNEL,
       EPANECHNIKOV_KERNEL,
       LAPLACIAN_KERNEL,
       SPHERICAL_KERNEL,
       TRIANGULAR_KERNEL
     };
   
    private:
     double bandwidth;
   
     double relError;
   
     double absError;
   
     KernelTypes kernelType;
   
     TreeTypes treeType;
   
     bool monteCarlo;
   
     double mcProb;
   
     size_t initialSampleSize;
   
     double mcEntryCoef;
   
     double mcBreakCoef;
   
     KDEWrapperBase* kdeModel;
   
    public:
     KDEModel(const double bandwidth = 1.0,
              const double relError = KDEDefaultParams::relError,
              const double absError = KDEDefaultParams::absError,
              const KernelTypes kernelType = KernelTypes::GAUSSIAN_KERNEL,
              const TreeTypes treeType = TreeTypes::KD_TREE,
              const bool monteCarlo = KDEDefaultParams::mode,
              const double mcProb = KDEDefaultParams::mcProb,
              const size_t initialSampleSize = KDEDefaultParams::initialSampleSize,
              const double mcEntryCoef = KDEDefaultParams::mcEntryCoef,
              const double mcBreakCoef = KDEDefaultParams::mcBreakCoef);
   
     KDEModel(const KDEModel& other);
   
     KDEModel(KDEModel&& other);
   
     KDEModel& operator=(const KDEModel& other);
   
     KDEModel& operator=(KDEModel&& other);
   
     ~KDEModel();
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
     double Bandwidth() const { return bandwidth; }
   
     void Bandwidth(const double newBandwidth);
   
     double RelativeError() const { return relError; }
   
     void RelativeError(const double newRelError);
   
     double AbsoluteError() const { return absError; }
   
     void AbsoluteError(const double newAbsError);
   
     TreeTypes TreeType() const { return treeType; }
   
     TreeTypes& TreeType() { return treeType; }
   
     KernelTypes KernelType() const { return kernelType; }
   
     KernelTypes& KernelType() { return kernelType; }
   
     bool MonteCarlo() const { return monteCarlo; }
   
     void MonteCarlo(const bool newMonteCarlo);
   
     double MCProbability() const { return mcProb; }
   
     void MCProbability(const double newMCProb);
   
     size_t MCInitialSampleSize() const { return initialSampleSize; }
   
     void MCInitialSampleSize(const size_t newSampleSize);
   
     double MCEntryCoefficient() const { return mcEntryCoef; }
   
     void MCEntryCoefficient(const double newEntryCoef);
   
     double MCBreakCoefficient() const { return mcBreakCoef; }
   
     void MCBreakCoefficient(const double newBreakCoef);
   
     KDEMode Mode() const { return kdeModel->Mode(); }
   
     KDEMode& Mode() { return kdeModel->Mode(); }
   
     void InitializeModel();
   
     void BuildModel(arma::mat&& referenceSet);
   
     void Evaluate(arma::mat&& querySet, arma::vec& estimations);
   
     void Evaluate(arma::vec& estimations);
   
   
    private:
     void CleanMemory();
   };
   
   } // namespace kde
   } // namespace mlpack
   
   #include "kde_model_impl.hpp"
   
   #endif
