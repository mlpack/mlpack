/**
 * @file kde.hpp
 * @author Roberto Hueso
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
#include <mlpack/core/tree/binary_space_tree.hpp>

#include "kde_stat.hpp"

namespace mlpack {
namespace kde /** Kernel Density Estimation. */ {

//! KDEMode represents the ways in which KDE algorithm can be executed.
enum KDEMode
{
  DUAL_TREE_MODE,
  SINGLE_TREE_MODE
};

/**
 * The KDE class is a template class for performing Kernel Density Estimations.
 * In statistics, kernel density estimation is a way to estimate the
 * probability density function of a variable in a non parametric way.
 * This implementation performs this estimation using a tree-independent
 * dual-tree algorithm. Details about this algorithm are available in KDERules.
 *
 * @tparam KernelType Kernel function to use for KDE calculations.
 * @tparam MetricType Metric to use for KDE calculations.
 * @tparam MatType Type of data to use.
 * @tparam TreeType Type of tree to use; must satisfy the TreeType policy API.
 * @tparam DualTreeTraversalType Type of dual-tree traversal to use.
 * @tparam SingleTreeTraversalType Type of single-tree traversal to use.
 */
template<typename KernelType = kernel::GaussianKernel,
         typename MetricType = mlpack::metric::EuclideanDistance,
         typename MatType = arma::mat,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType = tree::KDTree,
         template<typename RuleType> class DualTreeTraversalType =
             TreeType<MetricType,
                      kde::KDEStat,
                      MatType>::template DualTreeTraverser,
         template<typename RuleType> class SingleTreeTraversalType =
             TreeType<MetricType,
                      kde::KDEStat,
                      MatType>::template SingleTreeTraverser>
class KDE
{
 public:
  //! Convenience typedef.
  typedef TreeType<MetricType, kde::KDEStat, MatType> Tree;

  /**
   * Initialize KDE object using custom instantiated Metric and Kernel objects.
   *
   * @param relError Relative error tolerance of the model.
   * @param absError Absolute error tolerance of the model.
   * @param kernel Instantiated kernel object.
   * @param mode Mode for the algorithm.
   * @param metric Instantiated metric object.
   */
  KDE(const double relError = 0.05,
      const double absError = 0,
      KernelType kernel = KernelType(),
      const KDEMode mode = DUAL_TREE_MODE,
      MetricType metric = MetricType());

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
   * - If TreeTraits<TreeType>::RearrangesDataset is false then it is possible
   *   to use an empty oldFromNewReferences vector.
   *
   * @param referenceTree Built reference tree.
   * @param oldFromNewReferences Permutations of reference points obtained
   *                             during tree generation.
   */
  void Train(Tree* referenceTree, std::vector<size_t>* oldFromNewReferences);

  /**
   * Estimate density of each point in the query set given the data of the
   * reference set. The result is stored in an estimations vector.
   * Estimations might not be normalized.
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
  void Evaluate(MatType querySet, arma::vec& estimations);

  /**
   * Estimate density of each point in the query set given the data of an
   * already created query tree. The result is stored in an estimations vector.
   * Estimations might not be normalized.
   *
   * - Dimension of each point in the queryTree dataset must match the dimension
   *    of each point in the reference set.
   *
   * - Use std::move if the query tree is no longer needed.
   *
   * @pre The model has to be previously trained and mode has to be dual-tree.
   * @param queryTree Tree of query points to get the density of.
   * @param oldFromNewQueries Mappings of query points to the tree dataset.
   * @param estimations Object which will hold the density of each query point.
   */
  void Evaluate(Tree* queryTree,
                const std::vector<size_t>& oldFromNewQueries,
                arma::vec& estimations);

  /**
   * Estimate density of each point in the reference set given the data of the
   * reference set. It does not compute the estimation of a point with itself.
   * The result is stored in an estimations vector. Estimations might not be
   * normalized.
   *
   * @pre The model has to be previously trained.
   * @param estimations Object which will hold the density of each reference
   *                    point.
   */
  void Evaluate(arma::vec& estimations);

  //! Get the kernel.
  const KernelType& Kernel() const { return kernel; }

  //! Modify the kernel.
  KernelType& Kernel() { return kernel; }

  //! Get the metric.
  const MetricType& Metric() const { return metric; }

  //! Modify the metric.
  MetricType& Metric() { return metric; }

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

  //! Check whether reference tree is owned by the KDE model.
  bool OwnsReferenceTree() const { return ownsReferenceTree; }

  //! Check whether KDE model is trained or not.
  bool IsTrained() const { return trained; }

  //! Get the mode of KDE.
  KDEMode Mode() const { return mode; }

  //! Modify the mode of KDE.
  KDEMode& Mode() { return mode; }

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Kernel.
  KernelType kernel;

  //! Metric.
  MetricType metric;

  //! Reference tree.
  Tree* referenceTree;

  //! Permutations of reference points.
  std::vector<size_t>* oldFromNewReferences;

  //! Relative error tolerance.
  double relError;

  //! Absolute error tolerance.
  double absError;

  //! If true, the KDE object is responsible for deleting the reference tree.
  bool ownsReferenceTree;

  //! If true, the KDE object is trained.
  bool trained;

  //! Mode of the KDE algorithm.
  KDEMode mode;

  //! Check whether absolute and relative error values are compatible.
  static void CheckErrorValues(const double relError, const double absError);

  //! Rearrange estimations vector if required.
  static void RearrangeEstimations(const std::vector<size_t>& oldFromNew,
                                   arma::vec& estimations);
};

} // namespace kde
} // namespace mlpack

// Include implementation.
#include "kde_impl.hpp"

#endif // MLPACK_METHODS_KDE_KDE_HPP
