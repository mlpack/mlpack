/**
 * @file methods/kde/kde_model_impl.hpp
 * @author Roberto Hueso
 *
 * Implementation of KDE Model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KDE_MODEL_IMPL_HPP
#define MLPACK_METHODS_KDE_MODEL_IMPL_HPP

// In case it hasn't been included yet.
#include "kde_model.hpp"

namespace mlpack {

//! Initialize the KDEModel with the given parameters.
inline KDEModel::KDEModel(
    const double bandwidth,
    const double relError,
    const double absError,
    const KernelTypes kernelType,
    const TreeTypes treeType,
    const bool monteCarlo,
    const double mcProb,
    const size_t initialSampleSize,
    const double mcEntryCoef,
    const double mcBreakCoef) :
    bandwidth(bandwidth),
    relError(relError),
    absError(absError),
    kernelType(kernelType),
    treeType(treeType),
    monteCarlo(monteCarlo),
    mcProb(mcProb),
    initialSampleSize(initialSampleSize),
    mcEntryCoef(mcEntryCoef),
    mcBreakCoef(mcBreakCoef),
    kdeModel(NULL)
{
  // Nothing to do.
}

// Copy constructor.
inline KDEModel::KDEModel(const KDEModel& other) :
    bandwidth(other.bandwidth),
    relError(other.relError),
    absError(other.absError),
    kernelType(other.kernelType),
    treeType(other.treeType),
    monteCarlo(other.monteCarlo),
    mcProb(other.mcProb),
    initialSampleSize(other.initialSampleSize),
    mcEntryCoef(other.mcEntryCoef),
    mcBreakCoef(other.mcBreakCoef),
    kdeModel(other.kdeModel->Clone())
{
  // Nothing to do.
}

// Move constructor.
inline KDEModel::KDEModel(KDEModel&& other) :
    bandwidth(other.bandwidth),
    relError(other.relError),
    absError(other.absError),
    kernelType(other.kernelType),
    treeType(other.treeType),
    monteCarlo(other.monteCarlo),
    mcProb(other.mcProb),
    initialSampleSize(other.initialSampleSize),
    mcEntryCoef(other.mcEntryCoef),
    mcBreakCoef(other.mcBreakCoef),
    kdeModel(std::move(other.kdeModel))
{
  // Reset other model.
  other.bandwidth = 1.0;
  other.relError = KDEDefaultParams::relError;
  other.absError = KDEDefaultParams::absError;
  other.kernelType = KernelTypes::GAUSSIAN_KERNEL;
  other.treeType = TreeTypes::KD_TREE;
  other.monteCarlo = KDEDefaultParams::monteCarlo;
  other.mcProb = KDEDefaultParams::mcProb;
  other.initialSampleSize = KDEDefaultParams::initialSampleSize;
  other.mcEntryCoef = KDEDefaultParams::mcEntryCoef;
  other.mcBreakCoef = KDEDefaultParams::mcBreakCoef;
}

inline KDEModel& KDEModel::operator=(const KDEModel& other)
{
  if (this != &other)
  {
    delete kdeModel;

    bandwidth = other.bandwidth;
    relError = other.relError;
    absError = other.absError;
    kernelType = other.kernelType;
    treeType = other.treeType;
    monteCarlo = other.monteCarlo;
    mcProb = other.mcProb;
    initialSampleSize = other.initialSampleSize;
    mcEntryCoef = other.mcEntryCoef;
    mcBreakCoef = other.mcBreakCoef;
    kdeModel = other.kdeModel->Clone();
  }

  return *this;
}

inline KDEModel& KDEModel::operator=(KDEModel&& other)
{
  if (this != &other)
  {
    delete kdeModel;

    bandwidth = other.bandwidth;
    relError = other.relError;
    absError = other.absError;
    kernelType = other.kernelType;
    treeType = other.treeType;
    monteCarlo = other.monteCarlo;
    mcProb = other.mcProb;
    initialSampleSize = other.initialSampleSize;
    mcEntryCoef = other.mcEntryCoef;
    mcBreakCoef = other.mcBreakCoef;
    kdeModel = std::move(other.kdeModel);

    // Reset other model.
    other.bandwidth = 1.0;
    other.relError = KDEDefaultParams::relError;
    other.absError = KDEDefaultParams::absError;
    other.kernelType = KernelTypes::GAUSSIAN_KERNEL;
    other.treeType = TreeTypes::KD_TREE;
    other.monteCarlo = KDEDefaultParams::monteCarlo;
    other.mcProb = KDEDefaultParams::mcProb;
    other.initialSampleSize = KDEDefaultParams::initialSampleSize;
    other.mcEntryCoef = KDEDefaultParams::mcEntryCoef;
    other.mcBreakCoef = KDEDefaultParams::mcBreakCoef;
  }

  return *this;
}

// Clean memory.
inline KDEModel::~KDEModel()
{
  delete kdeModel;
}

template<template<typename TreeDistanceType,
                  typename TreeMatType,
                  typename TreeStatType> class TreeType>
KDEWrapperBase* InitializeModelHelper(const KDEModel::KernelTypes kernelType,
                                      const double relError,
                                      const double absError,
                                      const double bandwidth)
{
  switch (kernelType)
  {
    case KDEModel::GAUSSIAN_KERNEL:
      return new KDEWrapper<GaussianKernel, TreeType>(relError, absError,
          GaussianKernel(bandwidth));

    case KDEModel::EPANECHNIKOV_KERNEL:
      return new KDEWrapper<EpanechnikovKernel, TreeType>(relError, absError,
          EpanechnikovKernel(bandwidth));

    case KDEModel::LAPLACIAN_KERNEL:
      return new KDEWrapper<LaplacianKernel, TreeType>(relError, absError,
          LaplacianKernel(bandwidth));

    case KDEModel::SPHERICAL_KERNEL:
      return new KDEWrapper<SphericalKernel, TreeType>(relError, absError,
          SphericalKernel(bandwidth));

    case KDEModel::TRIANGULAR_KERNEL:
      return new KDEWrapper<TriangularKernel, TreeType>(relError, absError,
          TriangularKernel(bandwidth));
  }

  // This should never happen.
  return NULL;
}

inline void KDEModel::InitializeModel()
{
  // Clean memory, if necessary.
  delete kdeModel;

  // Build the actual model.
  switch (treeType)
  {
    case KD_TREE:
      kdeModel = InitializeModelHelper<KDTree>(kernelType, relError, absError,
          bandwidth);
      break;

    case BALL_TREE:
      kdeModel = InitializeModelHelper<BallTree>(kernelType, relError, absError,
          bandwidth);
      break;

    case COVER_TREE:
      kdeModel = InitializeModelHelper<StandardCoverTree>(kernelType, relError,
          absError, bandwidth);
      break;

    case OCTREE:
      kdeModel = InitializeModelHelper<Octree>(kernelType, relError, absError,
          bandwidth);
      break;

    case R_TREE:
      kdeModel = InitializeModelHelper<RTree>(kernelType, relError, absError,
          bandwidth);
      break;
  }
}

inline void KDEModel::BuildModel(util::Timers& timers,
                                 arma::mat&& referenceSet)
{
  InitializeModel();

  // Set whether to use Monte Carlo estimations or not.
  kdeModel->MonteCarlo() = monteCarlo;

  // Set Monte Carlo probability.
  kdeModel->MCProb(mcProb);

  // Set Monte Carlo initial sample size.
  kdeModel->MCInitialSampleSize() = initialSampleSize;

  // Set Monte Carlo entry coefficient.
  kdeModel->MCEntryCoef(mcEntryCoef);

  // Set Monte Carlo break coefficient.
  kdeModel->MCBreakCoef(mcBreakCoef);

  // Train the model.
  kdeModel->Train(timers, std::move(referenceSet));
}

// Perform bichromatic evaluation.
inline void KDEModel::Evaluate(util::Timers& timers,
                               arma::mat&& querySet,
                               arma::vec& estimates)
{
  kdeModel->Evaluate(timers, std::move(querySet), estimates);
}

// Perform monochromatic evaluation.
inline void KDEModel::Evaluate(util::Timers& timers,
                               arma::vec& estimates)
{
  kdeModel->Evaluate(timers, estimates);
}

// Clean memory.
inline void KDEModel::CleanMemory()
{
  delete kdeModel;
}

// Modify model kernel bandwidth.
inline void KDEModel::Bandwidth(const double newBandwidth)
{
  bandwidth = newBandwidth;
  kdeModel->Bandwidth(bandwidth);
}

// Modify model relative error tolerance.
inline void KDEModel::RelativeError(const double newRelError)
{
  relError = newRelError;
  kdeModel->RelativeError(relError);
}

// Modify model absolute error tolerance.
inline void KDEModel::AbsoluteError(const double newAbsError)
{
  absError = newAbsError;
  kdeModel->AbsoluteError(absError);
}

// Modify whether Monte Carlo estimations will be used.
inline void KDEModel::MonteCarlo(const bool newMonteCarlo)
{
  monteCarlo = newMonteCarlo;
  kdeModel->MonteCarlo() = monteCarlo;
}

// Modify model Monte Carlo probability.
inline void KDEModel::MCProbability(const double newMCProb)
{
  mcProb = newMCProb;
  kdeModel->MCProb(mcProb);
}

// Modify model Monte Carlo initial sample size.
inline void KDEModel::MCInitialSampleSize(const size_t newSampleSize)
{
  initialSampleSize = newSampleSize;
  kdeModel->MCInitialSampleSize() = initialSampleSize;
}

// Modify model Monte Carlo entry coefficient.
inline void KDEModel::MCEntryCoefficient(const double newEntryCoef)
{
  mcEntryCoef = newEntryCoef;
  kdeModel->MCEntryCoef(mcEntryCoef);
}

// Modify model Monte Carlo break coefficient.
inline void KDEModel::MCBreakCoefficient(const double newBreakCoef)
{
  mcBreakCoef = newBreakCoef;
  kdeModel->MCBreakCoef(mcBreakCoef);
}

//! Train the model (build the tree).
template<typename KernelType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void KDEWrapper<KernelType, TreeType>::Train(util::Timers& timers,
                                             arma::mat&& referenceSet)
{
  timers.Start("tree_building");
  kde.Train(std::move(referenceSet));
  timers.Stop("tree_building");
}

//! Perform bichromatic KDE (i.e. KDE with a separate query set).
template<typename KernelType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void KDEWrapper<KernelType, TreeType>::Evaluate(util::Timers& timers,
                                                arma::mat&& querySet,
                                                arma::vec& estimates)
{
  const size_t dimension = querySet.n_rows;
  if (kde.Mode() == KDE_DUAL_TREE_MODE)
  {
    // Build the query tree separately, so that we can time it.
    timers.Start("tree_building");
    std::vector<size_t> oldFromNewQueries;
    typename decltype(kde)::Tree* queryTree = BuildTree<
        typename decltype(kde)::Tree>(std::move(querySet), oldFromNewQueries);
    timers.Stop("tree_building");

    timers.Start("computing_kde");
    kde.Evaluate(queryTree, oldFromNewQueries, estimates);
    timers.Stop("computing_kde");

    delete queryTree;
  }
  else
  {
    timers.Start("computing_kde");
    kde.Evaluate(std::move(querySet), estimates);
    timers.Stop("computing_kde");
  }

  timers.Start("applying_normalizer");
  KernelNormalizer::ApplyNormalizer<KernelType>(kde.Kernel(),
                                                dimension,
                                                estimates);
  timers.Stop("applying_normalizer");
}

//! Perform monochromatic KDE (i.e. with the reference set as the query set).
template<typename KernelType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void KDEWrapper<KernelType, TreeType>::Evaluate(util::Timers& timers,
                                                arma::vec& estimates)
{
  timers.Start("computing_kde");
  kde.Evaluate(estimates);
  timers.Stop("computing_kde");

  timers.Start("applying_normalizer");
  const size_t dimension = kde.ReferenceTree()->Dataset().n_rows;
  KernelNormalizer::ApplyNormalizer<KernelType>(kde.Kernel(),
                                                dimension,
                                                estimates);
  timers.Stop("applying_normalizer");
}

template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         typename Archive>
void SerializationHelper(Archive& ar,
                         KDEWrapperBase* kdeModel,
                         const KDEModel::KernelTypes kernelType)
{
  switch (kernelType)
  {
    case KDEModel::GAUSSIAN_KERNEL:
      {
        KDEWrapper<GaussianKernel, TreeType>& typedModel =
            dynamic_cast<KDEWrapper<GaussianKernel, TreeType>&>(*kdeModel);
        ar(CEREAL_NVP(typedModel));
        break;
      }
    case KDEModel::EPANECHNIKOV_KERNEL:
      {
        KDEWrapper<EpanechnikovKernel, TreeType>& typedModel =
            dynamic_cast<KDEWrapper<EpanechnikovKernel, TreeType>&>(*kdeModel);
        ar(CEREAL_NVP(typedModel));
        break;
      }
    case KDEModel::LAPLACIAN_KERNEL:
      {
        KDEWrapper<LaplacianKernel, TreeType>& typedModel =
            dynamic_cast<KDEWrapper<LaplacianKernel, TreeType>&>(*kdeModel);
        ar(CEREAL_NVP(typedModel));
        break;
      }
    case KDEModel::SPHERICAL_KERNEL:
      {
        KDEWrapper<SphericalKernel, TreeType>& typedModel =
            dynamic_cast<KDEWrapper<SphericalKernel, TreeType>&>(*kdeModel);
        ar(CEREAL_NVP(typedModel));
        break;
      }
    case KDEModel::TRIANGULAR_KERNEL:
      {
        KDEWrapper<TriangularKernel, TreeType>& typedModel =
            dynamic_cast<KDEWrapper<TriangularKernel, TreeType>&>(*kdeModel);
        ar(CEREAL_NVP(typedModel));
        break;
      }
  }
}

// Serialize the model.
template<typename Archive>
void KDEModel::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(bandwidth));
  ar(CEREAL_NVP(relError));
  ar(CEREAL_NVP(absError));
  ar(CEREAL_NVP(kernelType));
  ar(CEREAL_NVP(treeType));
  ar(CEREAL_NVP(monteCarlo));
  ar(CEREAL_NVP(mcProb));
  ar(CEREAL_NVP(initialSampleSize));
  ar(CEREAL_NVP(mcEntryCoef));
  ar(CEREAL_NVP(mcBreakCoef));

  if (cereal::is_loading<Archive>())
  {
    monteCarlo = KDEDefaultParams::monteCarlo;
    mcProb = KDEDefaultParams::mcProb;
    initialSampleSize = KDEDefaultParams::initialSampleSize;
    mcEntryCoef = KDEDefaultParams::mcEntryCoef;
    mcBreakCoef = KDEDefaultParams::mcBreakCoef;
  }

  if (cereal::is_loading<Archive>())
    InitializeModel(); // Values will be overwritten.

  // Avoid polymorphism in serialization by serializing directly by the type.
  switch (treeType)
  {
    case KD_TREE:
      SerializationHelper<KDTree>(ar, kdeModel, kernelType);
      break;

    case BALL_TREE:
      SerializationHelper<BallTree>(ar, kdeModel, kernelType);
      break;

    case COVER_TREE:
      SerializationHelper<StandardCoverTree>(ar, kdeModel, kernelType);
      break;

    case OCTREE:
      SerializationHelper<Octree>(ar, kdeModel, kernelType);
      break;

    case R_TREE:
      SerializationHelper<RTree>(ar, kdeModel, kernelType);
      break;
  }
}

} // namespace mlpack

#endif
