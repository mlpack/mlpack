/**
 * @file spill_tree_impl.hpp
 *
 * Implementation of generalized hybrid spill tree (SpillTree).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_SPILL_TREE_IMPL_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_SPILL_TREE_IMPL_HPP

// In case it wasn't included already for some reason.
#include "spill_tree.hpp"

namespace mlpack {
namespace tree {

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
SpillTree(
    const MatType& data,
    const double tau,
    const size_t maxLeafSize,
    const double rho) :
    left(NULL),
    right(NULL),
    parent(NULL),
    count(0),
    pointsIndex(NULL),
    overlappingNode(false),
    hyperplane(),
    bound(data.n_rows),
    parentDistance(0), // Parent distance for the root is 0: it has no parent.
    dataset(&data),
    localDataset(false)
{
  arma::Col<size_t> points;
  if (dataset->n_cols > 0)
    // Fill points with all possible indexes: 0 .. (dataset->n_cols - 1).
    points = arma::linspace<arma::Col<size_t>>(0, dataset->n_cols - 1,
        dataset->n_cols);

  // Do the actual splitting of this node.
  SplitNode(points, maxLeafSize, tau, rho);

  // Create the statistic depending on if we are a leaf or not.
  stat = StatisticType(*this);
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
SpillTree(
    MatType&& data,
    const double tau,
    const size_t maxLeafSize,
    const double rho) :
    left(NULL),
    right(NULL),
    parent(NULL),
    count(0),
    pointsIndex(NULL),
    overlappingNode(false),
    hyperplane(),
    bound(data.n_rows),
    parentDistance(0), // Parent distance for the root is 0: it has no parent.
    dataset(new MatType(std::move(data))),
    localDataset(true)
{
  arma::Col<size_t> points;
  if (dataset->n_cols > 0)
    // Fill points with all possible indexes: 0 .. (dataset->n_cols - 1).
    points = arma::linspace<arma::Col<size_t>>(0, dataset->n_cols - 1,
        dataset->n_cols);

  // Do the actual splitting of this node.
  SplitNode(points, maxLeafSize, tau, rho);

  // Create the statistic depending on if we are a leaf or not.
  stat = StatisticType(*this);
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
SpillTree(
    SpillTree* parent,
    arma::Col<size_t>& points,
    const double tau,
    const size_t maxLeafSize,
    const double rho) :
    left(NULL),
    right(NULL),
    parent(parent),
    count(0),
    pointsIndex(NULL),
    overlappingNode(false),
    hyperplane(),
    bound(parent->Dataset().n_rows),
    dataset(&parent->Dataset()), // Point to the parent's dataset.
    localDataset(false)
{
  // Perform the actual splitting.
  SplitNode(points, maxLeafSize, tau, rho);

  // Create the statistic depending on if we are a leaf or not.
  stat = StatisticType(*this);
}

/**
 * Create a hybrid spill tree by copying the other tree.  Be careful!  This can
 * take a long time and use a lot of memory.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
SpillTree(const SpillTree& other) :
    left(NULL),
    right(NULL),
    parent(other.parent),
    count(other.count),
    pointsIndex(NULL),
    overlappingNode(other.overlappingNode),
    hyperplane(other.hyperplane),
    bound(other.bound),
    stat(other.stat),
    parentDistance(other.parentDistance),
    furthestDescendantDistance(other.furthestDescendantDistance),
    // Copy matrix, but only if we are the root and the other tree has its own
    // copy of the dataset.
    dataset((other.parent == NULL && other.localDataset) ?
        new MatType(*other.dataset) : other.dataset),
    localDataset(other.parent == NULL && other.localDataset)
{
  // Create left and right children (if any).
  if (other.Left())
  {
    left = new SpillTree(*other.Left());
    left->Parent() = this; // Set parent to this, not other tree.
  }

  if (other.Right())
  {
    right = new SpillTree(*other.Right());
    right->Parent() = this; // Set parent to this, not other tree.
  }

  // If vector of indexes, copy it.
  if (other.pointsIndex)
    pointsIndex = new arma::Col<size_t>(*other.pointsIndex);

  // Propagate matrix, but only if we are the root.
  if (parent == NULL && localDataset)
  {
    std::queue<SpillTree*> queue;
    if (left)
      queue.push(left);
    if (right)
      queue.push(right);
    while (!queue.empty())
    {
      SpillTree* node = queue.front();
      queue.pop();

      node->dataset = dataset;
      if (node->left)
        queue.push(node->left);
      if (node->right)
        queue.push(node->right);
    }
  }
}

/**
 * Move constructor.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
SpillTree(SpillTree&& other) :
    left(other.left),
    right(other.right),
    parent(other.parent),
    count(other.count),
    pointsIndex(other.pointsIndex),
    overlappingNode(other.overlappingNode),
    hyperplane(other.hyperplane),
    bound(std::move(other.bound)),
    stat(std::move(other.stat)),
    parentDistance(other.parentDistance),
    furthestDescendantDistance(other.furthestDescendantDistance),
    minimumBoundDistance(other.minimumBoundDistance),
    dataset(other.dataset),
    localDataset(other.localDataset)
{
  // Now we are a clone of the other tree.  But we must also clear the other
  // tree's contents, so it doesn't delete anything when it is destructed.
  other.left = NULL;
  other.right = NULL;
  other.count = 0;
  other.pointsIndex = NULL;
  other.parentDistance = 0.0;
  other.furthestDescendantDistance = 0.0;
  other.minimumBoundDistance = 0.0;
  other.dataset = NULL;
  other.localDataset = false;

  //Set new parent.
  if (left)
    left->parent = this;
  if (right)
    right->parent = this;
}

/**
 * Initialize the tree from an archive.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
template<typename Archive>
SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
SpillTree(
    Archive& ar,
    const typename boost::enable_if<typename Archive::is_loading>::type*) :
    SpillTree() // Create an empty SpillTree.
{
  // We've delegated to the constructor which gives us an empty tree, and now we
  // can serialize from it.
  ar >> data::CreateNVP(*this, "tree");
}

/**
 * Deletes this node, deallocating the memory for the children and calling their
 * destructors in turn.  This will invalidate any pointers or references to any
 * nodes which are children of this one.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
    ~SpillTree()
{
  delete left;
  delete right;
  delete pointsIndex;

  // If we're the root and we own the dataset, delete it.
  if (!parent && localDataset)
    delete dataset;
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
inline bool SpillTree<MetricType, StatisticType, MatType, HyperplaneType,
    SplitType>::IsLeaf() const
{
  return !left;
}

/**
 * Returns the number of children in this node.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
inline size_t SpillTree<MetricType, StatisticType, MatType, HyperplaneType,
    SplitType>::NumChildren() const
{
  if (left && right)
    return 2;
  if (left)
    return 1;

  return 0;
}

/**
 * Return the index of the nearest child node to the given query point (this
 * is an efficient estimation based on the splitting hyperplane, the node
 * returned is not necessarily the nearest).  If this is a leaf node, it will
 * return NumChildren() (invalid index).
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
template<typename VecType>
size_t SpillTree<MetricType, StatisticType, MatType, HyperplaneType,
    SplitType>::GetNearestChild(
    const VecType& point,
    typename boost::enable_if<IsVector<VecType> >::type*)
{
  if (IsLeaf() || !left || !right)
    return 0;

  if (hyperplane.Left(point))
    return 0;
  return 1;
}

/**
 * Return the index of the furthest child node to the given query point (this
 * is an efficient estimation based on the splitting hyperplane, the node
 * returned is not necessarily the furthest).  If this is a leaf node, it will
 * return NumChildren() (invalid index).
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
template<typename VecType>
size_t SpillTree<MetricType, StatisticType, MatType, HyperplaneType,
    SplitType>::GetFurthestChild(
    const VecType& point,
    typename boost::enable_if<IsVector<VecType> >::type*)
{
  if (IsLeaf() || !left || !right)
    return 0;

  if (hyperplane.Left(point))
    return 1;
  return 0;
}

/**
 * Return the index of the nearest child node to the given query node (this
 * is an efficient estimation based on the splitting hyperplane, the node
 * returned is not necessarily the nearest).  If it can't decide it will
 * return NumChildren() (invalid index).
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
size_t SpillTree<MetricType, StatisticType, MatType, HyperplaneType,
    SplitType>::GetNearestChild(const SpillTree& queryNode)
{
  if (IsLeaf() || !left || !right)
    return 0;

  if (hyperplane.Left(queryNode.Bound()))
    return 0;
  if (hyperplane.Right(queryNode.Bound()))
    return 1;
  // Can't decide.
  return 2;
}

/**
 * Return the index of the furthest child node to the given query point (this
 * is an efficient estimation based on the splitting hyperplane, the node
 * returned is not necessarily the furthest).  If this is a leaf node, it will
 * return NumChildren() (invalid index).
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
size_t SpillTree<MetricType, StatisticType, MatType, HyperplaneType,
    SplitType>::GetFurthestChild(const SpillTree& queryNode)
{
  if (IsLeaf() || !left || !right)
    return 0;

  if (hyperplane.Left(queryNode.Bound()))
    return 1;
  if (hyperplane.Right(queryNode.Bound()))
    return 0;
  // Can't decide.
  return 2;
}

/**
 * Return a bound on the furthest point in the node from the center.  This
 * returns 0 unless the node is a leaf.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
inline typename SpillTree<MetricType, StatisticType, MatType, HyperplaneType,
    SplitType>::ElemType
SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
    FurthestPointDistance() const
{
  if (!IsLeaf())
    return 0.0;

  // Otherwise return the distance from the center to a corner of the bound.
  return 0.5 * bound.Diameter();
}

/**
 * Return the furthest possible descendant distance.  This returns the maximum
 * distance from the center to the edge of the bound and not the empirical
 * quantity which is the actual furthest descendant distance.  So the actual
 * furthest descendant distance may be less than what this method returns (but
 * it will never be greater than this).
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
inline typename SpillTree<MetricType, StatisticType, MatType, HyperplaneType,
    SplitType>::ElemType
SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
    FurthestDescendantDistance() const
{
  return furthestDescendantDistance;
}

//! Return the minimum distance from the center to any bound edge.
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
inline typename SpillTree<MetricType, StatisticType, MatType, HyperplaneType,
    SplitType>::ElemType
SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
    MinimumBoundDistance() const
{
  return bound.MinWidth() / 2.0;
}

/**
 * Return the specified child.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
inline SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>&
SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
    Child(const size_t child) const
{
  if (child == 0)
    return *left;
  else
    return *right;
}

/**
 * Return the number of points contained in this node.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
inline size_t SpillTree<MetricType, StatisticType, MatType, HyperplaneType,
    SplitType>::NumPoints() const
{
  if (IsLeaf())
    return count;
  return 0;
}

/**
 * Return the number of descendants contained in the node.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
inline size_t SpillTree<MetricType, StatisticType, MatType, HyperplaneType,
    SplitType>::NumDescendants() const
{
  return count;
}

/**
 * Return the index of a particular descendant contained in this node.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
inline size_t SpillTree<MetricType, StatisticType, MatType, HyperplaneType,
    SplitType>::Descendant(const size_t index) const
{
  if (IsLeaf())
    return (*pointsIndex)[index];
  size_t num = left->NumDescendants();
  if (index < num)
    return left->Descendant(index);
  if (right)
    return right->Descendant(index - num);
  // This should never happen.
  return (size_t() - 1);
}

/**
 * Return the index of a particular point contained in this node.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
inline size_t SpillTree<MetricType, StatisticType, MatType, HyperplaneType,
    SplitType>::Point(const size_t index) const
{
  if (IsLeaf())
    return (*pointsIndex)[index];
  // This should never happen.
  return (size_t() - 1);
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
void SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
    SplitNode(arma::Col<size_t>& points,
              const size_t maxLeafSize,
              const double tau,
              const double rho)
{
  // We need to expand the bounds of this node properly.
  for (size_t i = 0; i < points.n_elem; i++)
    bound |= dataset->col(points[i]);

  // Calculate the furthest descendant distance.
  furthestDescendantDistance = 0.5 * bound.Diameter();

  // Now, check if we need to split at all.
  if (points.n_elem <= maxLeafSize)
  {
    pointsIndex = new arma::Col<size_t>();
    pointsIndex->swap(points);
    count = pointsIndex->n_elem;
    return; // We can't split this.
  }

  const bool split = SplitType<MetricType, MatType>::SplitSpace(bound,
      *dataset, points, hyperplane);
  // The node may not be always split. For instance, if all the points are the
  // same, we can't split them.
  if (!split)
  {
    pointsIndex = new arma::Col<size_t>();
    pointsIndex->swap(points);
    count = pointsIndex->n_elem;
    return; // We can't split this.
  }

  arma::Col<size_t> leftPoints, rightPoints;
  // Split the node.
  overlappingNode = SplitPoints(tau, rho, points, leftPoints, rightPoints);

  // We don't need the information in points, so lets clean it.
  arma::Col<size_t>().swap(points);

  // Now we will recursively split the children by calling their constructors
  // (which perform this splitting process).
  left = new SpillTree(this, leftPoints, tau, maxLeafSize, rho);
  right = new SpillTree(this, rightPoints, tau, maxLeafSize, rho);

  // Update count number, to represent the number of descendant points.
  count = left->NumDescendants() + right->NumDescendants();

  // Calculate parent distances for those two nodes.
  arma::vec center, leftCenter, rightCenter;
  Center(center);
  left->Center(leftCenter);
  right->Center(rightCenter);

  const ElemType leftParentDistance = MetricType::Evaluate(center, leftCenter);
  const ElemType rightParentDistance = MetricType::Evaluate(center,
      rightCenter);

  left->ParentDistance() = leftParentDistance;
  right->ParentDistance() = rightParentDistance;
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
bool SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
    SplitPoints(const double tau,
                const double rho,
                const arma::Col<size_t>& points,
                arma::Col<size_t>& leftPoints,
                arma::Col<size_t>& rightPoints)
{
  arma::vec projections(points.n_elem);
  size_t left = 0, right = 0, leftFrontier = 0, rightFrontier = 0;

  // Count the number of points to the left/right of the splitting hyperplane.
  for (size_t i = 0; i < points.n_elem; i++)
  {
    // Store projection value for future use.
    projections[i] = hyperplane.Project(dataset->col(points[i]));
    if (projections[i] <= 0)
    {
      left++;
      if (projections[i] > -tau)
        leftFrontier++;
    }
    else
    {
      right++;
      if (projections[i] < tau)
        rightFrontier++;
    }
  }

  const double p1 = double (left + rightFrontier) / points.n_elem;
  const double p2 = double (right + leftFrontier) / points.n_elem;

  if ((p1 <= rho || rightFrontier == 0) &&
      (p2 <= rho || leftFrontier == 0))
  {
    // Perform the actual splitting considering the overlapping buffer.  Points
    // with projection value in the range (-tau, tau) are included in both,
    // leftPoints and rightPoints.
    leftPoints.resize(left + rightFrontier);
    rightPoints.resize(right + leftFrontier);
    for (size_t i = 0, rc = 0, lc = 0; i < points.n_elem; i++)
    {
      if (projections[i] < tau || projections[i] <= 0)
        leftPoints[lc++] = points[i];
      if (projections[i] > -tau)
        rightPoints[rc++] = points[i];
    }
    // Return true, because it is a overlapping node.
    return true;
  }

  // Perform the actual splitting ignoring the overlapping buffer.  Points
  // with projection value less than or equal to zero are included in leftPoints
  // and points with projection value greater than zero are included in
  // rightPoints.
  leftPoints.resize(left);
  rightPoints.resize(right);
  for (size_t i = 0, rc = 0, lc = 0; i < points.n_elem; i++)
  {
    if (projections[i] <= 0)
      leftPoints[lc++] = points[i];
    else
      rightPoints[rc++] = points[i];
  }
  // Return false, because it isn't a overlapping node.
  return false;
}

// Default constructor (private), for boost::serialization.
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
    SpillTree() :
    left(NULL),
    right(NULL),
    parent(NULL),
    count(0),
    pointsIndex(NULL),
    overlappingNode(false),
    stat(*this),
    parentDistance(0),
    furthestDescendantDistance(0),
    dataset(NULL),
    localDataset(false)
{
  // Nothing to do.
}

/**
 * Serialize the tree.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
             class SplitType>
template<typename Archive>
void SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
    Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  // If we're loading, and we have children, they need to be deleted.
  if (Archive::is_loading::value)
  {
    if (left)
      delete left;
    if (right)
      delete right;
    if (!parent && localDataset)
      delete dataset;
  }

  ar & CreateNVP(parent, "parent");
  ar & CreateNVP(count, "count");
  ar & CreateNVP(pointsIndex, "pointsIndex");
  ar & CreateNVP(overlappingNode, "overlappingNode");
  ar & CreateNVP(hyperplane, "hyperplane");
  ar & CreateNVP(bound, "bound");
  ar & CreateNVP(stat, "statistic");
  ar & CreateNVP(parentDistance, "parentDistance");
  ar & CreateNVP(furthestDescendantDistance, "furthestDescendantDistance");
  ar & CreateNVP(dataset, "dataset");

  if (Archive::is_loading::value && parent == NULL)
    localDataset = true;

  // Save children last; otherwise boost::serialization gets confused.
  ar & CreateNVP(left, "left");
  ar & CreateNVP(right, "right");

  // Due to quirks of boost::serialization, if a tree is saved as an object and
  // not a pointer, the first level of the tree will be duplicated on load.
  // Therefore, if we are the root of the tree, then we need to make sure our
  // children's parent links are correct, and delete the duplicated node if
  // necessary.
  if (Archive::is_loading::value)
  {
    // Get parents of left and right children, or, NULL, if they don't exist.
    SpillTree* leftParent = left ? left->Parent() : NULL;
    SpillTree* rightParent = right ? right->Parent() : NULL;

    // Reassign parent links if necessary.
    if (left && left->Parent() != this)
      left->Parent() = this;
    if (right && right->Parent() != this)
      right->Parent() = this;

    // Do we need to delete the left parent?
    if (leftParent != NULL && leftParent != this)
    {
      // Sever the duplicate parent's children.  Ensure we don't delete the
      // dataset, by faking the duplicated parent's parent (that is, we need to
      // set the parent to something non-NULL; 'this' works).
      leftParent->Parent() = this;
      leftParent->Left() = NULL;
      leftParent->Right() = NULL;
      delete leftParent;
    }

    // Do we need to delete the right parent?
    if (rightParent != NULL && rightParent != this && rightParent != leftParent)
    {
      // Sever the duplicate parent's children, in the same way as above.
      rightParent->Parent() = this;
      rightParent->Left() = NULL;
      rightParent->Right() = NULL;
      delete rightParent;
    }
  }
}

} // namespace tree
} // namespace mlpack

#endif
