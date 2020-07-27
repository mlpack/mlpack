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

#include <boost/serialization/variant.hpp>

namespace mlpack {
namespace kde {

//! Initialize the KDEModel with the given parameters.
inline KDEModel::KDEModel(const double bandwidth,
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
  mcBreakCoef(mcBreakCoef)
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
  mcBreakCoef(other.mcBreakCoef)
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
  other.kdeModel = decltype(other.kdeModel)();
}

inline KDEModel& KDEModel::operator=(KDEModel other)
{
  boost::apply_visitor(DeleteVisitor(), kdeModel);
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
  return *this;
}

// Clean memory.
inline KDEModel::~KDEModel()
{
  boost::apply_visitor(DeleteVisitor(), kdeModel);
}

inline void KDEModel::BuildModel(arma::mat&& referenceSet)
{
  // Clean memory, if necessary.
  boost::apply_visitor(DeleteVisitor(), kdeModel);

  // Build the actual model.
  if (kernelType == GAUSSIAN_KERNEL && treeType == KD_TREE)
  {
    kdeModel = new KDEType<kernel::GaussianKernel, tree::KDTree>
        (relError, absError, kernel::GaussianKernel(bandwidth));
  }
  else if (kernelType == GAUSSIAN_KERNEL && treeType == BALL_TREE)
  {
    kdeModel = new KDEType<kernel::GaussianKernel, tree::BallTree>
        (relError, absError, kernel::GaussianKernel(bandwidth));
  }
  else if (kernelType == GAUSSIAN_KERNEL && treeType == COVER_TREE)
  {
    kdeModel = new KDEType<kernel::GaussianKernel, tree::StandardCoverTree>
        (relError, absError, kernel::GaussianKernel(bandwidth));
  }
  else if (kernelType == GAUSSIAN_KERNEL && treeType == OCTREE)
  {
    kdeModel = new KDEType<kernel::GaussianKernel, tree::Octree>
        (relError, absError, kernel::GaussianKernel(bandwidth));
  }
  else if (kernelType == GAUSSIAN_KERNEL && treeType == R_TREE)
  {
    kdeModel = new KDEType<kernel::GaussianKernel, tree::RTree>
        (relError, absError, kernel::GaussianKernel(bandwidth));
  }
  else if (kernelType == EPANECHNIKOV_KERNEL && treeType == KD_TREE)
  {
    kdeModel = new KDEType<kernel::EpanechnikovKernel, tree::KDTree>
        (relError, absError, kernel::EpanechnikovKernel(bandwidth));
  }
  else if (kernelType == EPANECHNIKOV_KERNEL && treeType == BALL_TREE)
  {
    kdeModel = new KDEType<kernel::EpanechnikovKernel, tree::BallTree>
        (relError, absError, kernel::EpanechnikovKernel(bandwidth));
  }
  else if (kernelType == EPANECHNIKOV_KERNEL && treeType == COVER_TREE)
  {
    kdeModel = new KDEType<kernel::EpanechnikovKernel, tree::StandardCoverTree>
        (relError, absError, kernel::EpanechnikovKernel(bandwidth));
  }
  else if (kernelType == EPANECHNIKOV_KERNEL && treeType == OCTREE)
  {
    kdeModel = new KDEType<kernel::EpanechnikovKernel, tree::Octree>
        (relError, absError, kernel::EpanechnikovKernel(bandwidth));
  }
  else if (kernelType == EPANECHNIKOV_KERNEL && treeType == R_TREE)
  {
    kdeModel = new KDEType<kernel::EpanechnikovKernel, tree::RTree>
        (relError, absError, kernel::EpanechnikovKernel(bandwidth));
  }
  else if (kernelType == LAPLACIAN_KERNEL && treeType == KD_TREE)
  {
    kdeModel = new KDEType<kernel::LaplacianKernel, tree::KDTree>
        (relError, absError, kernel::LaplacianKernel(bandwidth));
  }
  else if (kernelType == LAPLACIAN_KERNEL && treeType == BALL_TREE)
  {
    kdeModel = new KDEType<kernel::LaplacianKernel, tree::BallTree>
        (relError, absError, kernel::LaplacianKernel(bandwidth));
  }
  else if (kernelType == LAPLACIAN_KERNEL && treeType == COVER_TREE)
  {
    kdeModel = new KDEType<kernel::LaplacianKernel, tree::StandardCoverTree>
        (relError, absError, kernel::LaplacianKernel(bandwidth));
  }
  else if (kernelType == LAPLACIAN_KERNEL && treeType == OCTREE)
  {
    kdeModel = new KDEType<kernel::LaplacianKernel, tree::Octree>
        (relError, absError, kernel::LaplacianKernel(bandwidth));
  }
  else if (kernelType == LAPLACIAN_KERNEL && treeType == R_TREE)
  {
    kdeModel = new KDEType<kernel::LaplacianKernel, tree::RTree>
        (relError, absError, kernel::LaplacianKernel(bandwidth));
  }
  else if (kernelType == SPHERICAL_KERNEL && treeType == KD_TREE)
  {
    kdeModel = new KDEType<kernel::SphericalKernel, tree::KDTree>
        (relError, absError, kernel::SphericalKernel(bandwidth));
  }
  else if (kernelType == SPHERICAL_KERNEL && treeType == BALL_TREE)
  {
    kdeModel = new KDEType<kernel::SphericalKernel, tree::BallTree>
        (relError, absError, kernel::SphericalKernel(bandwidth));
  }
  else if (kernelType == SPHERICAL_KERNEL && treeType == COVER_TREE)
  {
    kdeModel = new KDEType<kernel::SphericalKernel, tree::StandardCoverTree>
        (relError, absError, kernel::SphericalKernel(bandwidth));
  }
  else if (kernelType == SPHERICAL_KERNEL && treeType == OCTREE)
  {
    kdeModel = new KDEType<kernel::SphericalKernel, tree::Octree>
        (relError, absError, kernel::SphericalKernel(bandwidth));
  }
  else if (kernelType == SPHERICAL_KERNEL && treeType == R_TREE)
  {
    kdeModel = new KDEType<kernel::SphericalKernel, tree::RTree>
        (relError, absError, kernel::SphericalKernel(bandwidth));
  }
  else if (kernelType == TRIANGULAR_KERNEL && treeType == KD_TREE)
  {
    kdeModel = new KDEType<kernel::TriangularKernel, tree::KDTree>
        (relError, absError, kernel::TriangularKernel(bandwidth));
  }
  else if (kernelType == TRIANGULAR_KERNEL && treeType == BALL_TREE)
  {
    kdeModel = new KDEType<kernel::TriangularKernel, tree::BallTree>
        (relError, absError, kernel::TriangularKernel(bandwidth));
  }
  else if (kernelType == TRIANGULAR_KERNEL && treeType == COVER_TREE)
  {
    kdeModel = new KDEType<kernel::TriangularKernel, tree::StandardCoverTree>
        (relError, absError, kernel::TriangularKernel(bandwidth));
  }
  else if (kernelType == TRIANGULAR_KERNEL && treeType == OCTREE)
  {
    kdeModel = new KDEType<kernel::TriangularKernel, tree::Octree>
        (relError, absError, kernel::TriangularKernel(bandwidth));
  }
  else if (kernelType == TRIANGULAR_KERNEL && treeType == R_TREE)
  {
    kdeModel = new KDEType<kernel::TriangularKernel, tree::RTree>
        (relError, absError, kernel::TriangularKernel(bandwidth));
  }

  // Set whether to use Monte Carlo estimations or not.
  MonteCarloVisitor MCVisitor(monteCarlo);
  boost::apply_visitor(MCVisitor, kdeModel);

  // Set Monte Carlo probability.
  MCProbabilityVisitor probabilityVisitor(mcProb);
  boost::apply_visitor(probabilityVisitor, kdeModel);

  // Set Monte Carlo initial sample size.
  MCSampleSizeVisitor sampleSizeVisitor(initialSampleSize);
  boost::apply_visitor(sampleSizeVisitor, kdeModel);

  // Set Monte Carlo entry coefficient.
  MCEntryCoefVisitor entryCoefficientVisitor(mcEntryCoef);
  boost::apply_visitor(entryCoefficientVisitor, kdeModel);

  // Set Monte Carlo break coefficient.
  MCBreakCoefVisitor breakCoefficientVisitor(mcBreakCoef);
  boost::apply_visitor(breakCoefficientVisitor, kdeModel);

  // Train the model.
  TrainVisitor train(std::move(referenceSet));
  boost::apply_visitor(train, kdeModel);
}

// Perform bichromatic evaluation.
inline void KDEModel::Evaluate(arma::mat&& querySet, arma::vec& estimations)
{
  Log::Info << "Evaluating KDE..." << std::endl;
  DualBiKDE eval(std::move(querySet), estimations);
  boost::apply_visitor(eval, kdeModel);
}

// Perform monochromatic evaluation.
inline void KDEModel::Evaluate(arma::vec& estimations)
{
  Log::Info << "Evaluating KDE..." << std::endl;
  DualMonoKDE eval(estimations);
  boost::apply_visitor(eval, kdeModel);
}

// Clean memory.
inline void KDEModel::CleanMemory()
{
  boost::apply_visitor(DeleteVisitor(), kdeModel);
}

// Parameters for KDE evaluation.
DualMonoKDE::DualMonoKDE(arma::vec& estimations):
    estimations(estimations)
{}

// Default KDE evaluation.
template<typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void DualMonoKDE::operator()(KDETypeT<KernelType, TreeType>* kde) const
{
  if (kde)
  {
    kde->Evaluate(estimations);
    const size_t dimension = (kde->ReferenceTree())->Dataset().n_rows;
    KernelNormalizer::ApplyNormalizer<KernelType>(kde->Kernel(),
                                                  dimension,
                                                  estimations);
  }
  else
  {
    throw std::runtime_error("no KDE model initialized");
  }
}

// Parameters for KDE evaluation.
DualBiKDE::DualBiKDE(arma::mat&& querySet, arma::vec& estimations):
    dimension(querySet.n_rows),
    querySet(std::move(querySet)),
    estimations(estimations)
{}

// Default KDE evaluation.
template<typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void DualBiKDE::operator()(KDETypeT<KernelType, TreeType>* kde) const
{
  if (kde)
  {
    kde->Evaluate(std::move(querySet), estimations);
    KernelNormalizer::ApplyNormalizer<KernelType>(kde->Kernel(),
                                                  dimension,
                                                  estimations);
  }
  else
  {
    throw std::runtime_error("no KDE model initialized");
  }
}

// Parameters for Train.
TrainVisitor::TrainVisitor(arma::mat&& referenceSet) :
    referenceSet(std::move(referenceSet))
{}

// Default Train.
template<typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void TrainVisitor::operator()(KDEType<KernelType, TreeType>* kde) const
{
  Log::Info << "Training KDE model..." << std::endl;
  if (kde)
    kde->Train(std::move(referenceSet));
  else
    throw std::runtime_error("no KDE model initialized");
}

// Modify kernel bandwidth.
BandwidthVisitor::BandwidthVisitor(const double bandwidth) :
    bandwidth(bandwidth)
{}

// Default modify kernel bandwidth.
template<typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void BandwidthVisitor::operator()(KDEType<KernelType, TreeType>* kde) const
{
  if (kde)
    kde->Kernel() = KernelType(bandwidth);
  else
    throw std::runtime_error("no KDE model initialized");
}

// Modify relative error tolerance.
RelErrorVisitor::RelErrorVisitor(const double relError) :
    relError(relError)
{}

// Default modify relative error tolerance.
template<typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RelErrorVisitor::operator()(KDEType<KernelType, TreeType>* kde) const
{
  if (kde)
    kde->RelativeError(relError);
  else
    throw std::runtime_error("no KDE model initialized");
}

// Modify absolute error tolerance.
AbsErrorVisitor::AbsErrorVisitor(const double absError) :
    absError(absError)
{}

// Default modify absolute error tolerance.
template<typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void AbsErrorVisitor::operator()(KDEType<KernelType, TreeType>* kde) const
{
  if (kde)
    kde->AbsoluteError(absError);
  else
    throw std::runtime_error("no KDE model initialized");
}

// Activate or deactivate Monte Carlo.
MonteCarloVisitor::MonteCarloVisitor(const bool monteCarlo) :
    monteCarlo(monteCarlo)
{}

// Default activate or deactivate Monte Carlo.
template<typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void MonteCarloVisitor::operator()(KDEType<KernelType, TreeType>* kde) const
{
  if (kde)
    kde->MonteCarlo() = monteCarlo;
  else
    throw std::runtime_error("no KDE model initialized");
}

// Set Monte Carlo probability.
MCProbabilityVisitor::MCProbabilityVisitor(const double probability) :
    probability(probability)
{}

// Default probability for Monte Carlo.
template<typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void MCProbabilityVisitor::operator()(KDEType<KernelType, TreeType>* kde) const
{
  if (kde)
    kde->MCProb(probability);
  else
    throw std::runtime_error("no KDE model initialized");
}

// Set Monte Carlo sample size.
MCSampleSizeVisitor::MCSampleSizeVisitor(const size_t sampleSize) :
    sampleSize(sampleSize)
{}

// Default sample size for Monte Carlo.
template<typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void MCSampleSizeVisitor::operator()(KDEType<KernelType, TreeType>* kde) const
{
  if (kde)
    kde->MCInitialSampleSize() = sampleSize;
  else
    throw std::runtime_error("no KDE model initialized");
}

// Set Monte Carlo entry coefficient.
MCEntryCoefVisitor::MCEntryCoefVisitor(const double entryCoef) :
    entryCoef(entryCoef)
{}

// Default entry coefficient for Monte Carlo.
template<typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void MCEntryCoefVisitor::operator()(KDEType<KernelType, TreeType>* kde) const
{
  if (kde)
    kde->MCEntryCoef(entryCoef);
  else
    throw std::runtime_error("no KDE model initialized");
}

// Set Monte Carlo break coefficient.
MCBreakCoefVisitor::MCBreakCoefVisitor(const double breakCoef) :
    breakCoef(breakCoef)
{}

// Default break coefficient for Monte Carlo.
template<typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void MCBreakCoefVisitor::operator()(KDEType<KernelType, TreeType>* kde) const
{
  if (kde)
    kde->MCBreakCoef(breakCoef);
  else
    throw std::runtime_error("no KDE model initialized");
}

// Delete model.
template<typename KDEType>
void DeleteVisitor::operator()(KDEType* kde) const
{
  if (kde)
    delete kde;
}

// Mode of model.
template<typename KDEType>
KDEMode& ModeVisitor::operator()(KDEType* kde) const
{
  if (kde)
    return kde->Mode();
  else
    throw std::runtime_error("no KDE model initialized");
}

// Get mode of model.
KDEMode KDEModel::Mode() const
{
  return boost::apply_visitor(ModeVisitor(), kdeModel);
}

// Modify mode of model.
KDEMode& KDEModel::Mode()
{
  return boost::apply_visitor(ModeVisitor(), kdeModel);
}

// Serialize the model.
template<typename Archive>
void KDEModel::serialize(Archive& ar, const unsigned int version)
{
  ar & BOOST_SERIALIZATION_NVP(bandwidth);
  ar & BOOST_SERIALIZATION_NVP(relError);
  ar & BOOST_SERIALIZATION_NVP(absError);
  ar & BOOST_SERIALIZATION_NVP(kernelType);
  ar & BOOST_SERIALIZATION_NVP(treeType);

  // Backward compatibility: Old versions of KDEModel did not need to handle
  // Monte Carlo parameters.
  if (version > 0)
  {
    ar & BOOST_SERIALIZATION_NVP(monteCarlo);
    ar & BOOST_SERIALIZATION_NVP(mcProb);
    ar & BOOST_SERIALIZATION_NVP(initialSampleSize);
    ar & BOOST_SERIALIZATION_NVP(mcEntryCoef);
    ar & BOOST_SERIALIZATION_NVP(mcBreakCoef);
  }
  else if (Archive::is_loading::value)
  {
    monteCarlo = KDEDefaultParams::monteCarlo;
    mcProb = KDEDefaultParams::mcProb;
    initialSampleSize = KDEDefaultParams::initialSampleSize;
    mcEntryCoef = KDEDefaultParams::mcEntryCoef;
    mcBreakCoef = KDEDefaultParams::mcBreakCoef;
  }

  if (Archive::is_loading::value)
    boost::apply_visitor(DeleteVisitor(), kdeModel);

  ar & BOOST_SERIALIZATION_NVP(kdeModel);
}

// Modify model kernel bandwidth.
void KDEModel::Bandwidth(const double newBandwidth)
{
  bandwidth = newBandwidth;
  BandwidthVisitor bandwidthVisitor(newBandwidth);
  boost::apply_visitor(bandwidthVisitor, kdeModel);
}

// Modify model relative error tolerance.
void KDEModel::RelativeError(const double newRelError)
{
  relError = newRelError;
  RelErrorVisitor relErrorVisitor(newRelError);
  boost::apply_visitor(relErrorVisitor, kdeModel);
}

// Modify model absolute error tolerance.
void KDEModel::AbsoluteError(const double newAbsError)
{
  absError = newAbsError;
  AbsErrorVisitor absErrorVisitor(newAbsError);
  boost::apply_visitor(absErrorVisitor, kdeModel);
}

// Modify whether Monte Carlo estimations will be used.
void KDEModel::MonteCarlo(const bool newMonteCarlo)
{
  monteCarlo = newMonteCarlo;
  MonteCarloVisitor monteCarloVisitor(newMonteCarlo);
  boost::apply_visitor(monteCarloVisitor, kdeModel);
}

// Modify model Monte Carlo probability.
void KDEModel::MCProbability(const double newMCProb)
{
  mcProb = newMCProb;
  MCProbabilityVisitor mcProbVisitor(newMCProb);
  boost::apply_visitor(mcProbVisitor, kdeModel);
}

// Modify model Monte Carlo initial sample size.
void KDEModel::MCInitialSampleSize(const size_t newSampleSize)
{
  initialSampleSize = newSampleSize;
  MCSampleSizeVisitor mcSampleSizeVisitor(newSampleSize);
  boost::apply_visitor(mcSampleSizeVisitor, kdeModel);
}

// Modify model Monte Carlo entry coefficient.
void KDEModel::MCEntryCoefficient(const double newEntryCoef)
{
  mcEntryCoef = newEntryCoef;
  MCEntryCoefVisitor mcEntryCoefVisitor(newEntryCoef);
  boost::apply_visitor(mcEntryCoefVisitor, kdeModel);
}

// Modify model Monte Carlo break coefficient.
void KDEModel::MCBreakCoefficient(const double newBreakCoef)
{
  mcBreakCoef = newBreakCoef;
  MCBreakCoefVisitor mcBreakCoefVisitor(newBreakCoef);
  boost::apply_visitor(mcBreakCoefVisitor, kdeModel);
}

} // namespace kde
} // namespace mlpack

#endif
