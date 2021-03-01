/**
 * @file methods/kde/kde_model.cpp
 * @author Roberto Hueso
 *
 * Implementation of KDE Model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "kde_model.hpp"

namespace mlpack {
namespace kde {

//! Initialize the KDEModel with the given parameters.
KDEModel::KDEModel(const double bandwidth,
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
KDEModel::KDEModel(const KDEModel& other) :
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
KDEModel::KDEModel(KDEModel&& other) :
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

KDEModel& KDEModel::operator=(const KDEModel& other)
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

KDEModel& KDEModel::operator=(KDEModel&& other)
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
KDEModel::~KDEModel()
{
  delete kdeModel;
}

template<template<typename TreeMetricType,
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
      return new KDEWrapper<kernel::GaussianKernel, TreeType>(
          relError, absError, kernel::GaussianKernel(bandwidth));

    case KDEModel::EPANECHNIKOV_KERNEL:
      return new KDEWrapper<kernel::EpanechnikovKernel, TreeType>(
          relError, absError, kernel::EpanechnikovKernel(bandwidth));

    case KDEModel::LAPLACIAN_KERNEL:
      return new KDEWrapper<kernel::LaplacianKernel, TreeType>(
          relError, absError, kernel::LaplacianKernel(bandwidth));

    case KDEModel::SPHERICAL_KERNEL:
      return new KDEWrapper<kernel::SphericalKernel, TreeType>(
          relError, absError, kernel::SphericalKernel(bandwidth));

    case KDEModel::TRIANGULAR_KERNEL:
      return new KDEWrapper<kernel::TriangularKernel, TreeType>(
          relError, absError, kernel::TriangularKernel(bandwidth));
  }

  // This should never happen.
  return NULL;
}

void KDEModel::InitializeModel()
{
  // Clean memory, if necessary.
  delete kdeModel;

  // Build the actual model.
  switch (treeType)
  {
    case KD_TREE:
      kdeModel = InitializeModelHelper<tree::KDTree>(kernelType, relError,
          absError, bandwidth);
      break;

    case BALL_TREE:
      kdeModel = InitializeModelHelper<tree::BallTree>(kernelType, relError,
          absError, bandwidth);
      break;

    case COVER_TREE:
      kdeModel = InitializeModelHelper<tree::StandardCoverTree>(kernelType,
          relError, absError, bandwidth);
      break;

    case OCTREE:
      kdeModel = InitializeModelHelper<tree::Octree>(kernelType, relError,
          absError, bandwidth);
      break;

    case R_TREE:
      kdeModel = InitializeModelHelper<tree::RTree>(kernelType, relError,
          absError, bandwidth);
      break;
  }
}

void KDEModel::BuildModel(arma::mat&& referenceSet)
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
  kdeModel->Train(std::move(referenceSet));
}

// Perform bichromatic evaluation.
void KDEModel::Evaluate(arma::mat&& querySet, arma::vec& estimates)
{
  kdeModel->Evaluate(std::move(querySet), estimates);
}

// Perform monochromatic evaluation.
void KDEModel::Evaluate(arma::vec& estimates)
{
  kdeModel->Evaluate(estimates);
}

// Clean memory.
void KDEModel::CleanMemory()
{
  delete kdeModel;
}

// Modify model kernel bandwidth.
void KDEModel::Bandwidth(const double newBandwidth)
{
  bandwidth = newBandwidth;
  kdeModel->Bandwidth(bandwidth);
}

// Modify model relative error tolerance.
void KDEModel::RelativeError(const double newRelError)
{
  relError = newRelError;
  kdeModel->RelativeError(relError);
}

// Modify model absolute error tolerance.
void KDEModel::AbsoluteError(const double newAbsError)
{
  absError = newAbsError;
  kdeModel->AbsoluteError(absError);
}

// Modify whether Monte Carlo estimations will be used.
void KDEModel::MonteCarlo(const bool newMonteCarlo)
{
  monteCarlo = newMonteCarlo;
  kdeModel->MonteCarlo() = monteCarlo;
}

// Modify model Monte Carlo probability.
void KDEModel::MCProbability(const double newMCProb)
{
  mcProb = newMCProb;
  kdeModel->MCProb(mcProb);
}

// Modify model Monte Carlo initial sample size.
void KDEModel::MCInitialSampleSize(const size_t newSampleSize)
{
  initialSampleSize = newSampleSize;
  kdeModel->MCInitialSampleSize() = initialSampleSize;
}

// Modify model Monte Carlo entry coefficient.
void KDEModel::MCEntryCoefficient(const double newEntryCoef)
{
  mcEntryCoef = newEntryCoef;
  kdeModel->MCEntryCoef(mcEntryCoef);
}

// Modify model Monte Carlo break coefficient.
void KDEModel::MCBreakCoefficient(const double newBreakCoef)
{
  mcBreakCoef = newBreakCoef;
  kdeModel->MCBreakCoef(mcBreakCoef);
}

} // namespace kde
} // namespace mlpack
