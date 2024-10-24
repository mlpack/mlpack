/**
 * @file methods/kde/kde_model.hpp
 * @author Roberto Hueso
 *
 * Model for KDE. It abstracts different types of tree, kernels, etc.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
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

/**
 * KernelNormalizer holds a set of methods to normalize estimations applying
 * in each case the appropiate kernel normalizer function.
 */
class KernelNormalizer
{
 private:
  // SFINAE check if Normalizer function is present.
  HAS_MEM_FUNC(Normalizer, HasNormalizer);

 public:
  //! Normalization not needed.
  template<typename KernelType>
  static void ApplyNormalizer(
      KernelType& /* kernel */,
      const size_t /* dimension */,
      arma::vec& /* estimations */,
      const std::enable_if_t<
          !HasNormalizer<KernelType, double(KernelType::*)(size_t)>::value>*
          = 0)
  { return; }

  //! Normalize kernels that have normalizer.
  template<typename KernelType>
  static void ApplyNormalizer(
      KernelType& kernel,
      const size_t dimension,
      arma::vec& estimations,
      const std::enable_if_t<
          HasNormalizer<KernelType, double(KernelType::*)(size_t)>::value>*
          = 0)
  {
    estimations /= kernel.Normalizer(dimension);
  }
};

/**
 * KDEWrapperBase is a base wrapper class for holding all KDE types supported by
 * KDEModel.  All KDE type wrappers inheirt from this class, allowing a simple
 * interface via inheritance for all the different types we want to support.
 */
class KDEWrapperBase
{
 public:
  //! Create the KDEWrapperBase object.  The base class does not hold anything,
  //! so this constructor does nothing.
  KDEWrapperBase() { }

  //! Create a new KDEWrapperBase that is the same as this one.  This function
  //! will properly handle polymorphism.
  virtual KDEWrapperBase* Clone() const = 0;

  //! Destruct the KDEWrapperBase (nothing to do).
  virtual ~KDEWrapperBase() { }

  //! Modify the bandwidth of the kernel.
  virtual void Bandwidth(const double bw) = 0;

  //! Modify the relative error tolerance.
  virtual void RelativeError(const double relError) = 0;

  //! Modify the absolute error tolerance.
  virtual void AbsoluteError(const double absError) = 0;

  //! Get whether Monte Carlo search is being used.
  virtual bool MonteCarlo() const = 0;
  //! Modify whether Monte Carlo search is being used.
  virtual bool& MonteCarlo() = 0;

  //! Modify the Monte Carlo probability.
  virtual void MCProb(const double mcProb) = 0;

  //! Get the Monte Carlo sample size.
  virtual size_t MCInitialSampleSize() const = 0;
  //! Modify the Monte Carlo sample size.
  virtual size_t& MCInitialSampleSize() = 0;

  //! Modify the Monte Carlo entry coefficient.
  virtual void MCEntryCoef(const double entryCoef) = 0;

  //! Modify the Monte Carlo break coefficient.
  virtual void MCBreakCoef(const double breakCoef) = 0;

  //! Get the search mode.
  virtual KDEMode Mode() const = 0;
  //! Modify the search mode.
  virtual KDEMode& Mode() = 0;

  //! Train the model (build the tree).
  virtual void Train(util::Timers& timers, arma::mat&& referenceSet) = 0;

  //! Perform bichromatic KDE (i.e. KDE with a separate query set).
  virtual void Evaluate(util::Timers& timers,
                        arma::mat&& querySet,
                        arma::vec& estimates) = 0;

  //! Perform monochromatic KDE (i.e. with the reference set as the query set).
  virtual void Evaluate(util::Timers& timers, arma::vec& estimates) = 0;
};

/**
 * KDEWrapper is a wrapper class for all KDE types supported by KDEModel.  It
 * can be extended with new child classes if new functionality for certain types
 * is needed.
 */
template<typename KernelType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
class KDEWrapper : public KDEWrapperBase
{
 public:
  //! Create the KDEWrapper object, initializing the internally-held KDE object.
  KDEWrapper(const double relError,
             const double absError,
             const KernelType& kernel) :
      kde(relError, absError, kernel)
  {
    // Nothing left to do.
  }

  //! Create a new KDEWrapper that is the same as this one.  This function
  //! will properly handle polymorphism.
  virtual KDEWrapper* Clone() const { return new KDEWrapper(*this); }

  //! Destruct the KDEWrapper (nothing to do).
  virtual ~KDEWrapper() { }

  //! Modify the bandwidth of the kernel.
  virtual void Bandwidth(const double bw) { kde.Kernel() = KernelType(bw); }

  //! Modify the relative error tolerance.
  virtual void RelativeError(const double eps) { kde.RelativeError(eps); }

  //! Modify the absolute error tolerance.
  virtual void AbsoluteError(const double eps) { kde.AbsoluteError(eps); }

  //! Get whether Monte Carlo search is being used.
  virtual bool MonteCarlo() const { return kde.MonteCarlo(); }
  //! Modify whether Monte Carlo search is being used.
  virtual bool& MonteCarlo() { return kde.MonteCarlo(); }

  //! Modify the Monte Carlo probability.
  virtual void MCProb(const double mcProb) { kde.MCProb(mcProb); }

  //! Get the Monte Carlo sample size.
  virtual size_t MCInitialSampleSize() const
  {
    return kde.MCInitialSampleSize();
  }
  //! Modify the Monte Carlo sample size.
  virtual size_t& MCInitialSampleSize()
  {
    return kde.MCInitialSampleSize();
  }

  //! Modify the Monte Carlo entry coefficient.
  virtual void MCEntryCoef(const double e) { kde.MCEntryCoef(e); }

  //! Modify the Monte Carlo break coefficient.
  virtual void MCBreakCoef(const double b) { kde.MCBreakCoef(b); }

  //! Get the search mode.
  virtual KDEMode Mode() const { return kde.Mode(); }
  //! Modify the search mode.
  virtual KDEMode& Mode() { return kde.Mode(); }

  //! Train the model (build the tree).
  virtual void Train(util::Timers& timers, arma::mat&& referenceSet);

  //! Perform bichromatic KDE (i.e. KDE with a separate query set).
  virtual void Evaluate(util::Timers& timers,
                        arma::mat&& querySet,
                        arma::vec& estimates);

  //! Perform monochromatic KDE (i.e. with the reference set as the query set).
  virtual void Evaluate(util::Timers& timers, arma::vec& estimates);

  //! Serialize the KDE model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(kde));
  }

 protected:
  using KDEType = KDE<KernelType, EuclideanDistance, arma::mat, TreeType>;

  //! The instantiated KDE object that we are wrapping.
  KDEType kde;
};

/**
 * The KDEModel provides an abstraction for the KDE class, abstracting away the
 * KernelType and TreeType parameters and allowing those to be specified at
 * runtime.  This class is written for the sake of the `kde` binding, but it is
 * not necessarily restricted to that usage.
 */
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
  //! Bandwidth of the kernel.
  double bandwidth;

  //! Relative error tolerance.
  double relError;

  //! Absolute error tolerance.
  double absError;

  //! Type of kernel.
  KernelTypes kernelType;

  //! Type of tree.
  TreeTypes treeType;

  //! Whether Monte Carlo estimations will be used.
  bool monteCarlo;

  //! Probability of estimation being bounded by relative error when using
  //! Monte Carlo estimations.
  double mcProb;

  //! Size of the initial sample for Monte Carlo estimations.
  size_t initialSampleSize;

  //! Entry coefficient for Monte Carlo estimations.
  double mcEntryCoef;

  //! Break coefficient for Monte Carlo estimations.
  double mcBreakCoef;

  /**
   * kdeModel holds whatever KDE type we are using.  It is initialized using the
   * `BuildModel()` method.
   */
  KDEWrapperBase* kdeModel;

 public:
  /**
   * Initialize KDEModel.
   *
   * @param bandwidth Bandwidth to use for the kernel.
   * @param relError Maximum relative error tolerance for each point in the
   *                 model. For example, 0.05 means that each value must be
   *                 within 5% of the true KDE value.
   * @param absError Maximum absolute error tolerance for each point in the
   *                 model. For example, 0.1 means that for each point the
   *                 value can have a maximum error of 0.1 units.
   * @param kernelType Type of kernel to use.
   * @param treeType Type of tree to use.
   * @param monteCarlo Whether to use Monte Carlo estimations when possible.
   * @param mcProb of a Monte Carlo estimation to be bounded by relative
   *        error tolerance.
   * @param initialSampleSize Initial sample size for Monte Carlo estimations.
   * @param mcEntryCoef Coefficient to control how much larger does the amount
   *                    of node descendants has to be compared to the initial
   *                    sample size in order for it to be a candidate for Monte
   *                    Carlo estimations.
   * @param mcBreakCoef Coefficient to control what fraction of the node's
   *                    descendants evaluated is the limit before Monte Carlo
   *                    estimation recurses.
   */
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

  //! Copy constructor of the given model.
  KDEModel(const KDEModel& other);

  //! Move constructor of the given model. Takes ownership of the model.
  KDEModel(KDEModel&& other);

  /**
   * Copy the given model.
   *
   * @param other KDEModel to copy.
   */
  KDEModel& operator=(const KDEModel& other);

  /**
   * Take ownership of the contents of the given model.
   *
   * @param other KDEModel to take ownership of.
   */
  KDEModel& operator=(KDEModel&& other);

  //! Destroy the KDEModel object.
  ~KDEModel();

  //! Serialize the KDE model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

  //! Get the bandwidth of the kernel.
  double Bandwidth() const { return bandwidth; }

  //! Modify the bandwidth of the kernel.
  void Bandwidth(const double newBandwidth);

  //! Get the relative error tolerance.
  double RelativeError() const { return relError; }

  //! Modify the relative error tolerance.
  void RelativeError(const double newRelError);

  //! Get the absolute error tolerance.
  double AbsoluteError() const { return absError; }

  //! Modify the absolute error tolerance.
  void AbsoluteError(const double newAbsError);

  //! Get the tree type of the model.
  TreeTypes TreeType() const { return treeType; }

  //! Modify the tree type of the model.
  TreeTypes& TreeType() { return treeType; }

  //! Get the kernel type of the model.
  KernelTypes KernelType() const { return kernelType; }

  //! Modify the kernel type of the model.
  KernelTypes& KernelType() { return kernelType; }

  //! Get whether the model is using Monte Carlo estimations or not.
  bool MonteCarlo() const { return monteCarlo; }

  //! Modify whether the model is using Monte Carlo estimations or not.
  void MonteCarlo(const bool newMonteCarlo);

  //! Get Monte Carlo probability of error being bounded by relative error.
  double MCProbability() const { return mcProb; }

  //! Modify Monte Carlo probability of error being bounded by relative error.
  void MCProbability(const double newMCProb);

  //! Get the initial sample size for Monte Carlo estimations.
  size_t MCInitialSampleSize() const { return initialSampleSize; }

  //! Modify the initial sample size for Monte Carlo estimations.
  void MCInitialSampleSize(const size_t newSampleSize);

  //! Get Monte Carlo entry coefficient.
  double MCEntryCoefficient() const { return mcEntryCoef; }

  //! Modify Monte Carlo entry coefficient.
  void MCEntryCoefficient(const double newEntryCoef);

  //! Get Monte Carlo break coefficient.
  double MCBreakCoefficient() const { return mcBreakCoef; }

  //! Modify Monte Carlo break coefficient.
  void MCBreakCoefficient(const double newBreakCoef);

  //! Get the mode of the model.
  KDEMode Mode() const { return kdeModel->Mode(); }

  //! Modify the mode of the model.
  KDEMode& Mode() { return kdeModel->Mode(); }

  /**
   * Initialize the KDE model.
   */
  void InitializeModel();

  /**
   * Build the KDE model with the given parameters and then trains it with the
   * given reference data.
   * Takes possession of the reference set to avoid a copy, so the reference set
   * will not be usable after this.
   *
   * @param timers Object to hold timing information in.
   * @param referenceSet Set of reference points.
   */
  void BuildModel(util::Timers& timers, arma::mat&& referenceSet);

  /**
   * Perform kernel density estimation on the given query set.
   * Takes possession of the query set to avoid a copy, so the query set
   * will not be usable after this. If possible, it returns normalized
   * estimations.
   *
   * @pre The model has to be previously created with BuildModel.
   * @param timers Object to hold timing information in.
   * @param querySet Set of query points.
   * @param estimations Vector where the results will be stored in the same
   *                    order as the query points.
   */
  void Evaluate(util::Timers& timers,
                arma::mat&& querySet,
                arma::vec& estimations);

  /**
   * Perform kernel density estimation on the reference set.
   * If possible, it returns normalized estimations.
   *
   * @pre The model has to be previously created with BuildModel.
   * @param timers Object to hold timing information in.
   * @param estimations Vector where the results will be stored in the same
   *                    order as the query points.
   */
  void Evaluate(util::Timers& timers, arma::vec& estimations);


 private:
  //! Clean memory.
  void CleanMemory();
};

} // namespace mlpack

#include "kde_model_impl.hpp"

#endif
