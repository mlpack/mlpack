/**
 * @file kde_model_impl.hpp
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
                          const TreeTypes treeType) :
  bandwidth(bandwidth),
  relError(relError),
  absError(absError),
  kernelType(kernelType),
  treeType(treeType)
{
  // Nothing to do
}

// Copy constructor.
inline KDEModel::KDEModel(const KDEModel& other) :
  bandwidth(other.bandwidth),
  relError(other.relError),
  absError(other.absError),
  kernelType(other.kernelType),
  treeType(other.treeType)
{
  // Nothing to do
}

// Move constructor.
inline KDEModel::KDEModel(KDEModel&& other) :
  bandwidth(other.bandwidth),
  relError(other.relError),
  absError(other.absError),
  kernelType(other.kernelType),
  treeType(other.treeType),
  kdeModel(std::move(other.kdeModel))
{
  // Reset other model
  other.bandwidth = 1.0;
  other.relError = 0.05;
  other.absError = 0;
  other.kernelType = KernelTypes::GAUSSIAN_KERNEL;
  other.treeType = TreeTypes::KD_TREE;
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
  kdeModel = std::move(other.kdeModel);
  return *this;
}

// Clean memory
inline KDEModel::~KDEModel()
{
  boost::apply_visitor(DeleteVisitor(), kdeModel);
}

inline void KDEModel::BuildModel(arma::mat&& referenceSet)
{
  // Clean memory, if necessary.
  boost::apply_visitor(DeleteVisitor(), kdeModel);

  if (kernelType == GAUSSIAN_KERNEL && treeType == KD_TREE)
  {
    kdeModel = new KDEType<kernel::GaussianKernel, tree::KDTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == GAUSSIAN_KERNEL && treeType == BALL_TREE)
  {
    kdeModel = new KDEType<kernel::GaussianKernel, tree::BallTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == GAUSSIAN_KERNEL && treeType == COVER_TREE)
  {
    kdeModel = new KDEType<kernel::GaussianKernel, tree::StandardCoverTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == GAUSSIAN_KERNEL && treeType == OCTREE)
  {
    kdeModel = new KDEType<kernel::GaussianKernel, tree::Octree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == GAUSSIAN_KERNEL && treeType == R_TREE)
  {
    kdeModel = new KDEType<kernel::GaussianKernel, tree::RTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == EPANECHNIKOV_KERNEL && treeType == KD_TREE)
  {
    kdeModel = new KDEType<kernel::EpanechnikovKernel, tree::KDTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == EPANECHNIKOV_KERNEL && treeType == BALL_TREE)
  {
    kdeModel = new KDEType<kernel::EpanechnikovKernel, tree::BallTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == EPANECHNIKOV_KERNEL && treeType == COVER_TREE)
  {
    kdeModel = new KDEType<kernel::EpanechnikovKernel, tree::StandardCoverTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == EPANECHNIKOV_KERNEL && treeType == OCTREE)
  {
    kdeModel = new KDEType<kernel::EpanechnikovKernel, tree::Octree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == EPANECHNIKOV_KERNEL && treeType == R_TREE)
  {
    kdeModel = new KDEType<kernel::EpanechnikovKernel, tree::RTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == LAPLACIAN_KERNEL && treeType == KD_TREE)
  {
    kdeModel = new KDEType<kernel::LaplacianKernel, tree::KDTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == LAPLACIAN_KERNEL && treeType == BALL_TREE)
  {
    kdeModel = new KDEType<kernel::LaplacianKernel, tree::BallTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == LAPLACIAN_KERNEL && treeType == COVER_TREE)
  {
    kdeModel = new KDEType<kernel::LaplacianKernel, tree::StandardCoverTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == LAPLACIAN_KERNEL && treeType == OCTREE)
  {
    kdeModel = new KDEType<kernel::LaplacianKernel, tree::Octree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == LAPLACIAN_KERNEL && treeType == R_TREE)
  {
    kdeModel = new KDEType<kernel::LaplacianKernel, tree::RTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == SPHERICAL_KERNEL && treeType == KD_TREE)
  {
    kdeModel = new KDEType<kernel::SphericalKernel, tree::KDTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == SPHERICAL_KERNEL && treeType == BALL_TREE)
  {
    kdeModel = new KDEType<kernel::SphericalKernel, tree::BallTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == SPHERICAL_KERNEL && treeType == COVER_TREE)
  {
    kdeModel = new KDEType<kernel::SphericalKernel, tree::StandardCoverTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == SPHERICAL_KERNEL && treeType == OCTREE)
  {
    kdeModel = new KDEType<kernel::SphericalKernel, tree::Octree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == SPHERICAL_KERNEL && treeType == R_TREE)
  {
    kdeModel = new KDEType<kernel::SphericalKernel, tree::RTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == TRIANGULAR_KERNEL && treeType == KD_TREE)
  {
    kdeModel = new KDEType<kernel::TriangularKernel, tree::KDTree>
        (bandwidth, relError, absError);
  }
  else if (kernelType == TRIANGULAR_KERNEL && treeType == BALL_TREE)
  {
    kdeModel = new KDEType<kernel::TriangularKernel, tree::BallTree>
        (bandwidth, relError, absError);
  }
   else if (kernelType == TRIANGULAR_KERNEL && treeType == COVER_TREE)
  {
    kdeModel = new KDEType<kernel::TriangularKernel, tree::StandardCoverTree>
        (bandwidth, relError, absError);
  }
   else if (kernelType == TRIANGULAR_KERNEL && treeType == OCTREE)
  {
    kdeModel = new KDEType<kernel::TriangularKernel, tree::Octree>
        (bandwidth, relError, absError);
  }
   else if (kernelType == TRIANGULAR_KERNEL && treeType == R_TREE)
  {
    kdeModel = new KDEType<kernel::TriangularKernel, tree::RTree>
        (bandwidth, relError, absError);
  }

  TrainVisitor train(std::move(referenceSet));
  boost::apply_visitor(train, kdeModel);
}

// Perform bichromatic evaluation
inline void KDEModel::Evaluate(arma::mat&& querySet, arma::vec& estimations)
{
  DualBiKDE eval(std::move(querySet), estimations);
  boost::apply_visitor(eval, kdeModel);
}

// Perform monochromatic evaluation
inline void KDEModel::Evaluate(arma::vec& estimations)
{
  DualMonoKDE eval(estimations);
  boost::apply_visitor(eval, kdeModel);
}

// Clean memory
inline void KDEModel::CleanMemory()
{
  boost::apply_visitor(DeleteVisitor(), kdeModel);
}

// Gaussian KDE normalization
template<typename KernelType>
void KernelNormalizer::ApplyNormalizer(kernel::GaussianKernel& kernel,
                                       const size_t dimension,
                                       arma::vec& estimations)
{
  estimations /= kernel.Normalizer(dimension);
}

// Epanechnikov KDE normalization
template<typename KernelType>
void KernelNormalizer::ApplyNormalizer(kernel::EpanechnikovKernel& kernel,
                                       const size_t dimension,
                                       arma::vec& estimations)
{
  estimations /= kernel.Normalizer(dimension);
}

// Spherical KDE normalization
template<typename KernelType>
void KernelNormalizer::ApplyNormalizer(kernel::SphericalKernel& kernel,
                                       const size_t dimension,
                                       arma::vec& estimations)
{
  estimations /= kernel.Normalizer(dimension);
}

// Parameters for KDE evaluation
DualMonoKDE::DualMonoKDE(arma::vec& estimations):
    estimations(estimations)
{}

// Default KDE evaluation
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
    throw std::runtime_error("no KDE model initialized");
}

// Parameters for KDE evaluation
DualBiKDE::DualBiKDE(arma::mat&& querySet, arma::vec& estimations):
    dimension(querySet.n_rows),
    querySet(std::move(querySet)),
    estimations(estimations)
{}

// Default KDE evaluation
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
    throw std::runtime_error("no KDE model initialized");
}

// Parameters for Train.
TrainVisitor::TrainVisitor(arma::mat&& referenceSet) :
    referenceSet(std::move(referenceSet))
{}

// Default Train
template<typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void TrainVisitor::operator()(KDEType<KernelType, TreeType>* kde) const
{
  if (kde)
    kde->Train(std::move(referenceSet));
  else
    throw std::runtime_error("no KDE model initialized");
}

// Delete model
template<typename KDEType>
void DeleteVisitor::operator()(KDEType* kde) const
{
  if (kde)
    delete kde;
}

// Serialize the model.
template<typename Archive>
void KDEModel::serialize(Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(bandwidth);
  ar & BOOST_SERIALIZATION_NVP(relError);
  ar & BOOST_SERIALIZATION_NVP(absError);
  ar & BOOST_SERIALIZATION_NVP(kernelType);
  ar & BOOST_SERIALIZATION_NVP(treeType);

  if (Archive::is_loading::value)
    boost::apply_visitor(DeleteVisitor(), kdeModel);

  ar & BOOST_SERIALIZATION_NVP(kdeModel);
}

} // namespace kde
} // namespace mlpack

#endif
