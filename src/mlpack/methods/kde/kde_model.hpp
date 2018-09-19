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

// Include trees
#include <mlpack/core/tree/binary_space_tree.hpp>

// Include kernels
#include <mlpack/core/kernels/kernel_traits.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/core/kernels/epanechnikov_kernel.hpp>

// Remaining includes
#include <boost/variant.hpp>
#include "kde.hpp"

namespace mlpack {
namespace kde {

//! Alias template.
template<typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
using KDEType = KDE<metric::EuclideanDistance, arma::mat, KernelType, TreeType>;

/**
 * DualTreeVisitor computes a Kernel Density Estimation on the given KDEType.
 */
class DualTreeVisitor : public boost::static_visitor<void>
{
 private:
  //! The query set for the KDE.
  const arma::mat& querySet;

  //! Vector to store the KDE results.
  arma::vec& estimations;

 public:
  //! Alias template necessary for visual C++ compiler.
  template<typename KernelType,
           template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using KDETypeT = KDEType<KernelType, TreeType>;

  //! Default DualTreeVisitor on some KDEType.
  template<typename KernelType,
           template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  void operator()(KDETypeT<KernelType, TreeType>* kde) const;

  // TODO Implement specific cases where a leaf size can be selected.

  //! DualTreeVisitor constructor. Takes ownership of the given querySet.
  DualTreeVisitor(arma::mat&& querySet, arma::vec& estimations);
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
  //! Alias template necessary for visual C++ compiler.
  template<typename KernelType,
           template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using KDETypeT = KDEType<KernelType, TreeType>;

  //! Default TrainVisitor on some KDEType.
  template<typename KernelType,
           template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  void operator()(KDETypeT<KernelType, TreeType>* kde) const;

  // TODO Implement specific cases where a leaf size can be selected.

  //! TrainVisitor constructor. Takes ownership of the given referenceSet.
  TrainVisitor(arma::mat&& referenceSet);
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
    BALL_TREE
  };

  enum KernelTypes
  {
    GAUSSIAN_KERNEL,
    EPANECHNIKOV_KERNEL
  };

 private:
  //! Bandwidth of the kernel.
  double bandwidth;

  //! Relative error tolerance.
  double relError;

  //! Absolute error tolerance.
  double absError;

  //! If true, a breadth-first approach is used when evaluating.
  bool breadthFirst;

  KernelTypes kernelType;

  TreeTypes treeType;

  /**
   * kdeModel holds an instance of each possible combination of KernelType and
   * TreeType. It is initialized using BuildModel.
   */
  boost::variant<KDEType<kernel::GaussianKernel, tree::KDTree>*,
                 KDEType<kernel::GaussianKernel, tree::BallTree>*,
                 KDEType<kernel::EpanechnikovKernel, tree::KDTree>*,
                 KDEType<kernel::EpanechnikovKernel, tree::BallTree>*> kdeModel;

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
   * @param breadthFirst Whether the tree should be traversed using a
   *                     breadth-first approach.
   * @param kernelType Type of kernel to use.
   * @param treeType Type of tree to use.
   */
  KDEModel(const double bandwidth = 1.0,
           const double relError = 1e-6,
           const double absError = 0,
           const bool breadthFirst = false,
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

  //! Get whether breadth-first traversal is being used.
  bool BreadthFirst() const { return breadthFirst; }

  //! Modify whether breadth-first traversal is being used.
  bool& BreadthFirst() { return breadthFirst; }

  //! Get the tree type of the model.
  TreeTypes TreeType() const { return treeType; }

  //! Modify the tree type of the model.
  TreeTypes& TreeType() { return treeType; }

  //! Get the kernel type of the model.
  KernelTypes KernelType() const { return kernelType; }

  //! Modify the kernel type of the model.
  KernelTypes& KernelType() { return kernelType; }

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
   * will not be usable after this.
   *
   * @pre The model has to be previously created with BuildModel.
   * @param querySet Set of query points.
   * @param estimations Vector where the results will be stored in the same
   *                    order as the query points.
   */
  void Evaluate(arma::mat&& querySet, arma::vec& estimations);

 private:
  //! Clean memory.
  void CleanMemory();
};

} // namespace kde
} // namespace mlpack

#include "kde_model_impl.hpp"

#endif
