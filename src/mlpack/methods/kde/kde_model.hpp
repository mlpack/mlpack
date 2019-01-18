/**
 * @file kde_model.hpp
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
#include <boost/variant.hpp>
#include "kde.hpp"

namespace mlpack {
namespace kde {

//! Alias template.
template<typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
using KDEType = KDE<KernelType,
                    metric::EuclideanDistance,
                    arma::mat,
                    TreeType,
                    TreeType<metric::EuclideanDistance,
                             kde::KDEStat,
                             arma::mat>::template DualTreeTraverser,
                    TreeType<metric::EuclideanDistance,
                             kde::KDEStat,
                             arma::mat>::template SingleTreeTraverser>;

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
      const typename std::enable_if<
          !HasNormalizer<KernelType, double(KernelType::*)(size_t)>::value>::
          type* = 0)
  { return; }

  //! Normalize kernels that have normalizer.
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

/**
 * DualMonoKDE computes a Kernel Density Estimation on the given KDEType.
 * It performs a monochromatic KDE.
 */
class DualMonoKDE : public boost::static_visitor<void>
{
 private:
  //! Vector to store the KDE results.
  arma::vec& estimations;

 public:
  //! Alias template necessary for Visual C++ compiler.
  template<typename KernelType,
           template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using KDETypeT = KDEType<KernelType, TreeType>;

  //! Default DualMonoKDE on some KDEType.
  template<typename KernelType,
           template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  void operator()(KDETypeT<KernelType, TreeType>* kde) const;

  // TODO Implement specific cases where a leaf size can be selected.

  //! DualMonoKDE constructor.
  DualMonoKDE(arma::vec& estimations);
};

/**
 * DualBiKDE computes a Kernel Density Estimation on the given KDEType.
 * It performs a bichromatic KDE.
 */
class DualBiKDE : public boost::static_visitor<void>
{
 private:
  //! Query set dimensionality.
  const size_t dimension;

  //! The query set for the KDE.
  const arma::mat& querySet;

  //! Vector to store the KDE results.
  arma::vec& estimations;

 public:
  //! Alias template necessary for Visual C++ compiler.
  template<typename KernelType,
           template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using KDETypeT = KDEType<KernelType, TreeType>;

  //! Default DualBiKDE on some KDEType.
  template<typename KernelType,
           template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  void operator()(KDETypeT<KernelType, TreeType>* kde) const;

  // TODO Implement specific cases where a leaf size can be selected.

  //! DualBiKDE constructor. Takes ownership of the given querySet.
  DualBiKDE(arma::mat&& querySet, arma::vec& estimations);
};

/**
 * TrainVisitor trains a given KDEType using a reference set.
 */
class TrainVisitor : public boost::static_visitor<void>
{
 private:
  //! The reference set used for training.
  arma::mat&& referenceSet;

 public:
  //! Default TrainVisitor on some KDEType.
  template<typename KernelType,
           template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  void operator()(KDEType<KernelType, TreeType>* kde) const;

  // TODO Implement specific cases where a leaf size can be selected.

  //! TrainVisitor constructor. Takes ownership of the given referenceSet.
  TrainVisitor(arma::mat&& referenceSet);
};

/**
 * ModeVisitor exposes the Mode() method of the KDEType.
 */
class ModeVisitor : public boost::static_visitor<KDEMode&>
{
 public:
  //! Return mode of KDEType instance.
  template<typename KDEType>
  KDEMode& operator()(KDEType* kde) const;
};

class DeleteVisitor : public boost::static_visitor<void>
{
 public:
  //! Delete KDEType instance.
  template<typename KDEType>
  void operator()(KDEType* kde) const;
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

  /**
   * kdeModel holds an instance of each possible combination of KernelType and
   * TreeType. It is initialized using BuildModel.
   */
  boost::variant<KDEType<kernel::GaussianKernel, tree::KDTree>*,
                 KDEType<kernel::GaussianKernel, tree::BallTree>*,
                 KDEType<kernel::GaussianKernel, tree::StandardCoverTree>*,
                 KDEType<kernel::GaussianKernel, tree::Octree>*,
                 KDEType<kernel::GaussianKernel, tree::RTree>*,
                 KDEType<kernel::EpanechnikovKernel, tree::KDTree>*,
                 KDEType<kernel::EpanechnikovKernel, tree::BallTree>*,
                 KDEType<kernel::EpanechnikovKernel, tree::StandardCoverTree>*,
                 KDEType<kernel::EpanechnikovKernel, tree::Octree>*,
                 KDEType<kernel::EpanechnikovKernel, tree::RTree>*,
                 KDEType<kernel::LaplacianKernel, tree::KDTree>*,
                 KDEType<kernel::LaplacianKernel, tree::BallTree>*,
                 KDEType<kernel::LaplacianKernel, tree::StandardCoverTree>*,
                 KDEType<kernel::LaplacianKernel, tree::Octree>*,
                 KDEType<kernel::LaplacianKernel, tree::RTree>*,
                 KDEType<kernel::SphericalKernel, tree::KDTree>*,
                 KDEType<kernel::SphericalKernel, tree::BallTree>*,
                 KDEType<kernel::SphericalKernel, tree::StandardCoverTree>*,
                 KDEType<kernel::SphericalKernel, tree::Octree>*,
                 KDEType<kernel::SphericalKernel, tree::RTree>*,
                 KDEType<kernel::TriangularKernel, tree::KDTree>*,
                 KDEType<kernel::TriangularKernel, tree::BallTree>*,
                 KDEType<kernel::TriangularKernel, tree::StandardCoverTree>*,
                 KDEType<kernel::TriangularKernel, tree::Octree>*,
                 KDEType<kernel::TriangularKernel, tree::RTree>*> kdeModel;

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
   */
  KDEModel(const double bandwidth = 1.0,
           const double relError = 0.05,
           const double absError = 0,
           const KernelTypes kernelType = KernelTypes::GAUSSIAN_KERNEL,
           const TreeTypes treeType = TreeTypes::KD_TREE);

  //! Copy constructor of the given model.
  KDEModel(const KDEModel& other);

  //! Move constructor of the given model. Takes ownership of the model.
  KDEModel(KDEModel&& other);

  /**
   * Copy the given model.
   *
   * Use std::move if the object to copy is no longer needed.
   *
   * @param other KDEModel to copy.
   */
  KDEModel& operator=(KDEModel other);

  //! Destroy the KDEModel object.
  ~KDEModel();

  //! Serialize the KDE model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

  //! Get the bandwidth of the kernel.
  double Bandwidth() const { return bandwidth; }

  //! Modify the bandwidth of the kernel.
  double& Bandwidth() { return bandwidth; }

  //! Get the relative error tolerance.
  double RelativeError() const { return relError; }

  //! Modify the relative error tolerance.
  double& RelativeError() { return relError; }

  //! Get the absolute error tolerance.
  double AbsoluteError() const { return absError; }

  //! Modify the absolute error tolerance.
  double& AbsoluteError() { return absError; }

  //! Get the tree type of the model.
  TreeTypes TreeType() const { return treeType; }

  //! Modify the tree type of the model.
  TreeTypes& TreeType() { return treeType; }

  //! Get the kernel type of the model.
  KernelTypes KernelType() const { return kernelType; }

  //! Modify the kernel type of the model.
  KernelTypes& KernelType() { return kernelType; }

  //! Get the mode of the model.
  KDEMode Mode() const;

  //! Modify the mode of the model.
  KDEMode& Mode();

  /**
   * Build the KDE model with the given parameters and then trains it with the
   * given reference data.
   * Takes possession of the reference set to avoid a copy, so the reference set
   * will not be usable after this.
   *
   * @param referenceSet Set of reference points.
   */
  void BuildModel(arma::mat&& referenceSet);

  /**
   * Perform kernel density estimation on the given query set.
   * Takes possession of the query set to avoid a copy, so the query set
   * will not be usable after this. If possible, it returns normalized
   * estimations.
   *
   * @pre The model has to be previously created with BuildModel.
   * @param querySet Set of query points.
   * @param estimations Vector where the results will be stored in the same
   *                    order as the query points.
   */
  void Evaluate(arma::mat&& querySet, arma::vec& estimations);

  /**
   * Perform kernel density estimation on the reference set.
   * If possible, it returns normalized estimations.
   *
   * @pre The model has to be previously created with BuildModel.
   * @param estimations Vector where the results will be stored in the same
   *                    order as the query points.
   */
  void Evaluate(arma::vec& estimations);


 private:
  //! Clean memory.
  void CleanMemory();
};

} // namespace kde
} // namespace mlpack

#include "kde_model_impl.hpp"

#endif
