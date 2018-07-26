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

class DualTreeVisitor : public boost::static_visitor<void>
{
 private:
  const arma::mat& querySet;

  arma::vec& estimations;

 public:
  //! Alias template necessary for visual C++ compiler.
  template<typename KernelType,
           template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using KDETypeT = KDEType<KernelType, TreeType>;

  template<typename KernelType,
           template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  void operator()(KDETypeT<KernelType, TreeType>* kde) const;

  // TODO Implement specific cases where a leaf size can be selected.

  DualTreeVisitor(const arma::mat& querySet,
                  arma::vec& estimations);
};

class TrainVisitor : public boost::static_visitor<void>
{
 private:
  arma::mat&& referenceSet;

 public:
  //! Alias template necessary for visual C++ compiler.
  template<typename KernelType,
           template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using KDETypeT = KDEType<KernelType, TreeType>;

  template<typename KernelType,
           template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  void operator()(KDETypeT<KernelType, TreeType>* kde) const;

  // TODO Implement specific cases where a leaf size can be selected.

  TrainVisitor(arma::mat&& referenceSet);
};

class DeleteVisitor : public boost::static_visitor<void>
{
 public:
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

  boost::variant<KDEType<kernel::GaussianKernel, tree::KDTree>*,
                 KDEType<kernel::GaussianKernel, tree::BallTree>*,
                 KDEType<kernel::EpanechnikovKernel, tree::KDTree>*,
                 KDEType<kernel::EpanechnikovKernel, tree::BallTree>*> kdeModel;

 public:
  KDEModel(const double bandwidth = 1.0,
           const double relError = 1e-6,
           const double absError = 0,
           const bool breadthFirst = false,
           const KernelTypes kernelType = KernelTypes::GAUSSIAN_KERNEL,
           const TreeTypes treeType = TreeTypes::KD_TREE);

  KDEModel(const KDEModel& other);

  KDEModel(KDEModel&& other);

  KDEModel& operator=(KDEModel other);

  ~KDEModel();

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

  double Bandwidth() const { return bandwidth; }

  double& Bandwidth() { return bandwidth; }

  double RelativeError() const { return relError; }

  double& RelativeError() { return relError; }

  double AbsoluteError() const { return absError; }

  double& AbsoluteError() { return absError; }

  //! Get whether breadth-first traversal is being used.
  bool BreadthFirst() const { return breadthFirst; }

  //! Modify whether breadth-first traversal is being used.
  bool& BreadthFirst() { return breadthFirst; }

  TreeTypes TreeType() const { return treeType; }

  TreeTypes& TreeType() { return treeType; }

  KernelTypes KernelType() const { return kernelType; }

  KernelTypes& KernelType() { return kernelType; }

  void BuildModel(arma::mat&& referenceSet);

  void Evaluate(arma::mat&& querySet, arma::vec& estimations);

 private:
  void CleanMemory();
};

} // namespace kde
} // namespace mlpack

#include "kde_model_impl.hpp"

#endif
