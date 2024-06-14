/**
 * @file core/tree/example_tree.hpp
 * @author Ryan Curtin
 *
 * An example tree.  This contains all the functions that mlpack trees must
 * implement (although the actual implementations here don't make any sense
 * because this is just an example).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_EXAMPLE_TREE_HPP
#define MLPACK_CORE_TREE_EXAMPLE_TREE_HPP

namespace mlpack {

/**
 * This is not an actual space tree but instead an example tree that exists to
 * show and document all the functions that mlpack trees must implement.  For a
 * better overview of trees, see @ref trees.  Also be aware that the
 * implementations of each of the methods in this example tree are entirely fake
 * and do not work; this example tree exists for its API, not its
 * implementation.
 *
 * Note that trees often have different properties.  These properties are known
 * at compile-time through the TreeTraits class, and some properties may imply
 * the existence (or non-existence) of certain functions.  Refer to the
 * TreeTraits for more documentation on that.
 *
 * The three template parameters below must be template parameters to the tree,
 * in the order given below.  More template parameters are fine, but they must
 * come after the first three.
 *
 * @tparam DistanceType This defines the space in which the tree will be built.
 *      For some trees, arbitrary distance metrics cannot be used, and a
 *      template metaprogramming approach should be used to issue a compile-time
 *      error if a distance metric cannot be used with a specific tree type.
 *      One example is the BinarySpaceTree tree type, which cannot work with the
 *      IPMetric class.
 * @tparam StatisticType A tree node can hold a statistic, which is sometimes
 *      useful for various dual-tree algorithms.  The tree itself does not need
 *      to know anything about how the statistic works, but it needs to hold a
 *      StatisticType in each node.  It can be assumed that the StatisticType
 *      class has a constructor StatisticType(const ExampleTree&).
 * @tparam MatType A tree could be built on a dense matrix or a sparse matrix.
 *      All mlpack trees should be able to support any Armadillo-compatible
 *      matrix type.  When the tree is written it should be assumed that MatType
 *      has the same functionality as arma::mat.
 */
template<typename DistanceType = LMetric<2, true>,
         typename StatisticType = EmptyStatistic,
         typename MatType = arma::mat>
class ExampleTree
{
 public:
  /**
   * This constructor will build the tree given a dataset and an instantiated
   * distance metric.  Note that the parameter is a MatType& and not an
   * arma::mat&.  The dataset is not modified by the tree-building process (if
   * it is, see the documentation for TreeTraits::RearrangesDataset for how to
   * deal with that situation).  The DistanceType parameter is necessary even
   * though some distance metrics do not hold any state.  This is so that the
   * tree does not have to worry about instantiating the metric (if the tree had
   * to worry about this, this would almost certainly incur additional runtime
   * complexity and a larger runtime size of the tree node objects, which is to
   * be avoided).  The metric can't be const, in case DistanceType::Evaluate()
   * is non-const.
   *
   * When this constructor is finished, the entire tree will be built and ready
   * to use.  The constructor should call the constructor of the statistic for
   * each node that is built (see EmptyStatistic for more information).
   *
   * @param dataset The dataset that the tree will be built on.
   * @param distance The instantiated distance metric to use to build the
   *     dataset.
   */
  ExampleTree(const MatType& dataset,
              DistanceType& distance);

  //! Return the number of children of this node.
  size_t NumChildren() const;

  //! Return a particular child of this node.
  const ExampleTree& Child(const size_t i) const;
  //! Modify a particular child of this node.
  ExampleTree& Child(const size_t i);

  //! Return the parent node (NULL if this is the root of the tree).
  ExampleTree* Parent() const;

  //! Return the number of points held in this node.
  size_t NumPoints() const;

  /**
   * Return the index of a particular point of this node.  mlpack trees do not,
   * in general, hold the actual dataset, and instead just hold the indices of
   * the points they contain.  Thus, you might use this function in code like
   * this:
   *
   * @code
   * arma::vec thirdPoint = dataset.col(treeNode.Point(2));
   * @endcode
   */
  size_t Point(const size_t i) const;

  /**
   * Get the number of descendant points.  This is the number of unique points
   * held in this node plus the number of points held in all descendant nodes.
   * This could be calculated at build-time and cached, or could be calculated
   * at run-time.  This may be harder to calculate for trees that may hold
   * points in multiple nodes (like cover trees and spill trees, for instance).
   */
  size_t NumDescendants() const;

  /**
   * Get the index of a particular descendant point.  The ordering of the
   * descendants does not matter, as long as calling Descendant(0) through
   * Descendant(NumDescendants() - 1) will return the indices of every
   * unique descendant point of the node.
   */
  size_t Descendant(const size_t i) const;

  //! Get the statistic for this node.
  const StatisticType& Stat() const;
  //! Modify the statistic for this node.
  StatisticType& Stat();

  //! Get the instantiated distance for this node.
  const DistanceType& Distance() const;
  //! Modify the instantiated distance for this node.
  DistanceType& Distance();

  /**
   * Return the minimum distance between this node and a point.  It is not
   * required that the exact minimum distance between the node and the point is
   * returned but instead a lower bound on the minimum distance will suffice.
   * See the definitions in @ref trees for more information.
   *
   * @param point Point to return [lower bound on] minimum distance to.
   */
  double MinDistance(const MatType& point) const;

  /**
   * Return the minimum distance between this node and another node.  It is not
   * required that the exact minimum distance between the two nodes be returned
   * but instead a lower bound on the minimum distance will suffice.  See the
   * definitions in @ref trees for more information.
   *
   * @param other Node to return [lower bound on] minimum distance to.
   */
  double MinDistance(const ExampleTree& other) const;

  /**
   * Return the maximum distance between this node and a point.  It is not
   * required that the exact maximum distance between the node and the point is
   * returned but instead an upper bound on the maximum distance will suffice.
   * See the definitions in @ref trees for more information.
   *
   * @param point Point to return [upper bound on] maximum distance to.
   */
  double MaxDistance(const MatType& point) const;

  /**
   * Return the maximum distance between this node and another node.  It is not
   * required that the exact maximum distance between the two nodes be returned
   * but instead an upper bound on the maximum distance will suffice.  See the
   * definitions in @ref trees for more information.
   *
   * @param other Node to return [upper bound on] maximum distance to.
   */
  double MaxDistance(const ExampleTree& other) const;

  /**
   * Return both the minimum and maximum distances between this node and a point
   * as a Range object.  This overload is given because it is possible
   * that, for some tree types, calculation of both at once is faster than a
   * call to MinDistance() then MaxDistance().  It is not necessary that the
   * minimum and maximum distances be exact; it is sufficient to return a lower
   * bound on the minimum distance and an upper bound on the maximum distance.
   * See the definitions in @ref trees for more information.
   *
   * @param point Point to return [bounds on] minimum and maximum distances to.
   */
  Range RangeDistance(const MatType& point) const;

  /**
   * Return both the minimum and maximum distances between this node and another
   * node as a Range object.  This overload is given because it is
   * possible that, for some tree types, calculation of both at once is faster
   * than a call to MinDistance() then MaxDistance().  It is not necessary that
   * the minimum and maximum distances be exact; it is sufficient to return a
   * lower bound on the minimum distance and an upper bound on the maximum
   * distance.  See the definitions in @ref trees for more information.
   *
   * @param other Node to return [bounds on] minimum and maximum distances to.
   */
  Range RangeDistance(const ExampleTree& other) const;

  /**
   * Fill the given vector with the center of the node.
   *
   * @param centroid Vector to be filled with the center of the node.
   */
  void Centroid(arma::vec& centroid) const;

  /**
   * Get the distance from the center of the node to the furthest descendant
   * point of this node.  This does not necessarily need to be the exact
   * furthest descendant distance but instead can be an upper bound.  See the
   * definitions in @ref trees for more information.
   */
  double FurthestDescendantDistance() const;

  /**
   * Get the distance from the center of this node to the center of the parent
   * node.
   */
  double ParentDistance() const;

 private:
  //! This member is just here so the ExampleTree compiles without warnings.  It
  //! is not required to be a member in every type of tree.
  StatisticType stat;

  /**
   * This member is just here so the ExampleTree compiles without warnings.  It
   * is not required to be a member in every type of tree.  Be aware that
   * storing the distance metric as a member and not a reference may mean that
   * for some distance metrics (such as MahalanobisDistance in high
   * dimensionality) may incur lots of unnecessary matrix copying.
   */
  DistanceType& metric;
};

} // namespace mlpack

#endif
