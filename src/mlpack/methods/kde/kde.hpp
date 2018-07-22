/**
 * @file kde.hpp
 * @author Roberto Hueso (robertohueso96@gmail.com)
 *
 * Kernel Density Estimation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_KDE_KDE_HPP
#define MLPACK_METHODS_KDE_KDE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>

#include "kde_stat.hpp"

namespace mlpack {
namespace kde /** Kernel Density Estimation. */ {

/**
 * The KDE class is a template class for performing Kernel Density Estimations.
 * In statistics, kernel density estimation, is a way to estimate the
 * probability density function of a variable in a non parametric way.
 * This implementation performs this estimation using a tree-independent
 * dual-tree algorithm. Details about this algorithm are available in KDERules.
 *
 * @tparam MetricType Metric to use for KDE calculations.
 * @tparam MatType Type of data to use.
 * @tparam KernelType Kernel function to use for KDE calculations.
 * @tparam TreeType Type of tree to use; must satisfy the TreeType policy API.
 */
template<typename MetricType = mlpack::metric::EuclideanDistance,
         typename MatType = arma::mat,
         typename KernelType = kernel::GaussianKernel,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType = tree::KDTree>
class KDE
{
 public:
  //! Convenience typedef.
  typedef TreeType<MetricType, kde::KDEStat, MatType> Tree;

  /**
   * Initialize KDE object with the default Kernel and Metric parameters.
   * Relative error tolernce is initialized to 1e-6, absolute error tolerance
   * is 0.0 and uses a depth-first approach.
   */
  KDE();

  /**
   * Initialize KDE object using the default Metric parameters and a given
   * Kernel bandwidth (<b>only for kernels that require a bandwidth and are
   * constructed like kernel(bandwidth)</b>).
   *
   * @param bandwidth Bandwidth of the kernel.
   * @param relError Relative error tolerance of the model.
   * @param absError Absolute error tolerance of the model.
   * @param breadthFirst Whether the tree should be traversed using a
   *        breadth-first approach.
   */
  KDE(const double bandwidth,
      const double relError = 1e-6,
      const double absError = 0,
      const bool breadthFirst = false);

  /**
   * Initialize KDE object using custom instantiated Metric and Kernel objects.
   *
   * @param metric Instantiated metric object.
   * @param kernel Instantiated kernel object.
   * @param relError Relative error tolerance of the model.
   * @param absError Absolute error tolerance of the model.
   * @param breadthFirst Whether the tree should be traversed using a
   *        breadth-first approach.
   */
  KDE(MetricType& metric,
      KernelType& kernel,
      const double relError = 1e-6,
      const double absError = 0,
      const bool breadthFirst = false);

  /**
   * Construct KDE object as a copy of the given model. This may be
   * computationally intensive!
   *
   * @param other KDE object to copy.
   */
  KDE(const KDE& other);

  /**
   * Construct KDE object taking ownership of the given model.
   *
   * @param other KDE object to take ownership of.
   */
  KDE(KDE&& other);

  /**
   * Copy a KDE model.
   *
   * Use std::move if the object to copy is no longer needed.
   *
   * @param other KDE model to copy.
   */
  KDE& operator=(KDE other);

  /**
   * Destroy the KDE object. If this object created any trees, they will be
   * deleted. If you created the trees then you have to delete them yourself.
   */
  ~KDE();

  /**
   * Trains the KDE model. It builds a tree using a reference set.
   *
   * Use std::move if the reference set is no longer needed.
   *
   * @param referenceSet Set of reference data.
   */
  void Train(MatType referenceSet);

  /**
   * Trains the KDE model. Sets the reference tree to an already created tree.
   *
   * @param referenceTree New already created reference tree.
   */
  void Train(Tree* referenceTree);

  /**
   * Estimate density of each point in the query set given the data of the
   * reference set. The result is stored in an estimations vector.
   *
   * - Dimension of each point in the query set must match the dimension of each
   *   point in the reference set.
   *
   * - Use std::move if the query set is no longer needed.
   *
   * @pre The model has to be previously trained.
   * @param querySet Set of query points to get the density of.
   * @param estimations Object which will hold the density of each query point.
   */
  void Evaluate(const MatType& querySet, arma::vec& estimations);

  /**
   * Estimate density of each point in the query set given the data of an
   * already created query tree. The result is stored in an estimations vector.
   *
   * - Dimension of each point in the queryTree dataset must match the dimension
   *    of each point in the reference set.
   *
   * - Use std::move if the query tree is no longer needed.
   *
   * @pre The model has to be previously trained.
   * @param queryTree Tree of query points to get the density of.
   * @param oldFromNewQueries Mappings of query points to the tree dataset.
   * @param estimations Object which will hold the density of each query point.
   */
  void Evaluate(Tree* queryTree,
                const std::vector<size_t>& oldFromNewQueries,
                arma::vec& estimations);

  //! Get the kernel.
  const KernelType& Kernel() const { return kernel; }

  //! Modify the kernel.
  KernelType& Kernel() { return kernel; }

  //! Get the reference tree.
  Tree* ReferenceTree() { return referenceTree; }

  //! Get relative error tolerance.
  double RelativeError() const { return relError; }

  //! Modify relative error tolerance (0 <= newError <= 1).
  void RelativeError(const double newError);

  //! Get absolute error tolerance.
  double AbsoluteError() const { return absError; }

  //! Modify absolute error tolerance (0 <= newError).
  void AbsoluteError(const double newError);

  //! Get whether breadth-first traversal is being used.
  bool BreadthFirst() const { return breadthFirst; }

  //! Modify whether breadth-first traversal is being used.
  bool& BreadthFirst() { return breadthFirst; }

  //! Check whether reference tree is owned by the KDE model.
  bool OwnsReferenceTree() const { return ownsReferenceTree; }

  //! Check whether KDE model is trained or not.
  bool IsTrained() const { return trained; }

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Kernel.
  KernelType* kernel;

  //! Metric.
  MetricType* metric;

  //! Reference tree.
  Tree* referenceTree;

  //! Relative error tolerance.
  double relError;

  //! Absolute error tolerance.
  double absError;

  //! If true, a breadth-first approach is used when evaluating.
  bool breadthFirst;

  //! If true, the KDE object is responsible for deleting the kernel.
  bool ownsKernel;

  //! If true, the KDE object is responsible for deleting the metric.
  bool ownsMetric;

  //! If true, the KDE object is responsible for deleting the reference tree.
  bool ownsReferenceTree;

  //! If true, the KDE object is trained.
  bool trained;

  //! Check whether absolute and relative error values are compatible.
  void CheckErrorValues(const double relError, const double absError) const;
};

} // namespace kde
} // namespace mlpack

// Include implementation.
#include "kde_impl.hpp"

#endif // MLPACK_METHODS_KDE_KDE_HPP
