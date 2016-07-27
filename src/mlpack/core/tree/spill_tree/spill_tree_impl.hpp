/**
 * @file spill_tree_impl.hpp
 *
 * Implementation of generalized hybrid spill tree (SpillTree).
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
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
SpillTree<MetricType, StatisticType, MatType, SplitType>::
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
    splitDimension(0),
    splitValue(0),
    bound(data.n_rows),
    parentDistance(0), // Parent distance for the root is 0: it has no parent.
    dataset(new MatType(data)) // Copies the dataset.
{
  std::vector<size_t> points;
  points.reserve(dataset->n_cols);
  for (size_t i = 0; i < dataset->n_cols; i++)
    points.push_back(i);

  // Do the actual splitting of this node.
  SplitNode(points, points.size(), maxLeafSize, tau, rho);

  // Create the statistic depending on if we are a leaf or not.
  stat = StatisticType(*this);
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
SpillTree<MetricType, StatisticType, MatType, SplitType>::
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
    splitDimension(0),
    splitValue(0),
    bound(data.n_rows),
    parentDistance(0), // Parent distance for the root is 0: it has no parent.
    dataset(new MatType(std::move(data)))
{
  std::vector<size_t> points;
  points.reserve(dataset->n_cols);
  for (size_t i = 0; i < dataset->n_cols; i++)
    points.push_back(i);

  // Do the actual splitting of this node.
  SplitNode(points, points.size(), maxLeafSize, tau, rho);

  // Create the statistic depending on if we are a leaf or not.
  stat = StatisticType(*this);
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
SpillTree<MetricType, StatisticType, MatType, SplitType>::
SpillTree(
    SpillTree* parent,
    std::vector<size_t>& points,
    const size_t overlapIndex,
    const double tau,
    const size_t maxLeafSize,
    const double rho) :
    left(NULL),
    right(NULL),
    parent(parent),
    count(0),
    pointsIndex(NULL),
    overlappingNode(false),
    splitDimension(0),
    splitValue(0),
    bound(parent->Dataset().n_rows),
    dataset(&parent->Dataset()) // Point to the parent's dataset.
{
  // Perform the actual splitting.
  SplitNode(points, overlapIndex, maxLeafSize, tau, rho);

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
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
SpillTree<MetricType, StatisticType, MatType, SplitType>::
SpillTree(const SpillTree& other) :
    left(NULL),
    right(NULL),
    parent(other.parent),
    count(other.count),
    pointsIndex(NULL),
    overlappingNode(other.overlappingNode),
    splitDimension(other.splitDimension),
    splitValue(other.splitValue),
    bound(other.bound),
    stat(other.stat),
    parentDistance(other.parentDistance),
    furthestDescendantDistance(other.furthestDescendantDistance),
    // Copy matrix, but only if we are the root.
    dataset((other.parent == NULL) ? new MatType(*other.dataset) : NULL)
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
    pointsIndex = new std::vector<size_t>(*other.pointsIndex);

  // Propagate matrix, but only if we are the root.
  if (parent == NULL)
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
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
SpillTree<MetricType, StatisticType, MatType, SplitType>::
SpillTree(SpillTree&& other) :
    left(other.left),
    right(other.right),
    parent(other.parent),
    count(other.count),
    overlappingNode(other.overlappingNode),
    pointsIndex(other.pointsIndex),
    splitDimension(other.splitDimension),
    splitValue(other.splitValue),
    bound(std::move(other.bound)),
    stat(std::move(other.stat)),
    parentDistance(other.parentDistance),
    furthestDescendantDistance(other.furthestDescendantDistance),
    minimumBoundDistance(other.minimumBoundDistance),
    dataset(other.dataset)
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
}

/**
 * Initialize the tree from an archive.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
template<typename Archive>
SpillTree<MetricType, StatisticType, MatType, SplitType>::
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
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
SpillTree<MetricType, StatisticType, MatType, SplitType>::
    ~SpillTree()
{
  delete left;
  delete right;
  delete pointsIndex;

  // If we're the root, delete the matrix.
  if (!parent)
    delete dataset;
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
inline bool SpillTree<MetricType, StatisticType, MatType, SplitType>::
    IsLeaf() const
{
  return !left;
}

/**
 * Returns the number of children in this node.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
inline size_t SpillTree<MetricType, StatisticType, MatType, SplitType>::
    NumChildren() const
{
  if (left && right)
    return 2;
  if (left)
    return 1;

  return 0;
}

/**
 * Return a bound on the furthest point in the node from the center.  This
 * returns 0 unless the node is a leaf.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
inline typename SpillTree<MetricType, StatisticType, MatType, SplitType>::
    ElemType
SpillTree<MetricType, StatisticType, MatType, SplitType>::
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
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
inline typename SpillTree<MetricType, StatisticType, MatType, SplitType>::
    ElemType
SpillTree<MetricType, StatisticType, MatType, SplitType>::
    FurthestDescendantDistance() const
{
  return furthestDescendantDistance;
}

//! Return the minimum distance from the center to any bound edge.
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
inline typename SpillTree<MetricType, StatisticType, MatType, SplitType>::
    ElemType
SpillTree<MetricType, StatisticType, MatType, SplitType>::
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
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
inline SpillTree<MetricType, StatisticType, MatType, SplitType>&
SpillTree<MetricType, StatisticType, MatType, SplitType>::
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
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
inline size_t SpillTree<MetricType, StatisticType, MatType, SplitType>::
    NumPoints() const
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
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
inline size_t SpillTree<MetricType, StatisticType, MatType, SplitType>::
    NumDescendants() const
{
  return count;
}

/**
 * Return the index of a particular descendant contained in this node.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
inline size_t SpillTree<MetricType, StatisticType, MatType, SplitType>::
    Descendant(const size_t index) const
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
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
inline size_t SpillTree<MetricType, StatisticType, MatType, SplitType>::
    Point(const size_t index) const
{
  if (IsLeaf())
    return (*pointsIndex)[index];
  // This should never happen.
  return (size_t() - 1);
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
void SpillTree<MetricType, StatisticType, MatType, SplitType>::
    SplitNode(std::vector<size_t>& points,
              const size_t overlapIndex,
              const size_t maxLeafSize,
              const double tau,
              const double rho)
{
  // We need to expand the bounds of this node properly, ignoring overlapping
  // points (they will be included in the bound of the other node).
  for (size_t i = 0; i < overlapIndex; i++)
    bound |= dataset->cols(points[i], points[i]);

  // Calculate the furthest descendant distance.
  furthestDescendantDistance = 0.5 * bound.Diameter();

  // Now, check if we need to split at all.
  if (points.size() <= maxLeafSize)
  {
    pointsIndex = new std::vector<size_t>();
    pointsIndex->swap(points);
    count = pointsIndex->size();
    return; // We can't split this.
  }

  const bool split = SplitType<bound::HRectBound<MetricType>,
      MatType>::SplitNode(bound, *dataset, points, splitDimension, splitValue);
  // The node may not be always split. For instance, if all the points are the
  // same, we can't split them.
  if (!split)
  {
    pointsIndex = new std::vector<size_t>();
    pointsIndex->swap(points);
    count = pointsIndex->size();
    return; // We can't split this.
  }

  std::vector<size_t> leftPoints, rightPoints;
  size_t overlapIndexLeft, overlapIndexRight;
  // Split the node.
  overlappingNode = SplitPoints(splitDimension, splitValue, tau, rho, points,
      leftPoints, rightPoints, overlapIndexLeft, overlapIndexRight);

  // We don't need the information in points, so lets clean it.
  std::vector<size_t>().swap(points);

  // Now we will recursively split the children by calling their constructors
  // (which perform this splitting process).
  left = new SpillTree(this, leftPoints, overlapIndexLeft, tau, maxLeafSize,
      rho);
  right = new SpillTree(this, rightPoints, overlapIndexRight, tau, maxLeafSize,
      rho);

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
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
bool SpillTree<MetricType, StatisticType, MatType, SplitType>::
    SplitPoints(const size_t splitDimension,
                const double splitVal,
                const double tau,
                const double rho,
                const std::vector<size_t>& points,
                std::vector<size_t>& leftPoints,
                std::vector<size_t>& rightPoints,
                size_t& overlapIndexLeft,
                size_t& overlapIndexRight)
{
  std::vector<size_t> leftFrontier, rightFrontier;

  // Perform the actual splitting.  This will order the dataset such that points
  // with value in dimension splitDimension less than or equal to splitVal are
  // included in leftPoints, and points with value in dimension splitDimension
  // greater than splitVal are included in rightPoints.
  for (size_t i = 0; i < points.size(); i++)
    if ((*dataset)(splitDimension, points[i]) <= splitVal)
    {
      leftPoints.push_back(points[i]);
      if ((*dataset)(splitDimension, points[i]) > splitVal - tau)
        leftFrontier.push_back(points[i]);
    }
    else
    {
      rightPoints.push_back(points[i]);
      if ((*dataset)(splitDimension, points[i]) < splitVal + tau)
        rightFrontier.push_back(points[i]);
    }

  const double p1 = double (leftPoints.size() + rightFrontier.size()) /
      points.size();
  const double p2 = double (rightPoints.size() + leftFrontier.size()) /
      points.size();

  overlapIndexLeft = leftPoints.size();
  overlapIndexRight = rightPoints.size();

  if ((p1 <= rho || rightFrontier.empty()) && (p2 <= rho || leftFrontier.empty()))
  {
    leftPoints.insert(leftPoints.end(), rightFrontier.begin(),
        rightFrontier.end());
    rightPoints.insert(rightPoints.end(), leftFrontier.begin(),
        leftFrontier.end());
    return true;
  }

  return false;
}

// Default constructor (private), for boost::serialization.
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
SpillTree<MetricType, StatisticType, MatType, SplitType>::
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
    dataset(NULL)
{
  // Nothing to do.
}

/**
 * Serialize the tree.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
template<typename Archive>
void SpillTree<MetricType, StatisticType, MatType, SplitType>::
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
    if (!parent)
      delete dataset;
  }

  ar & CreateNVP(parent, "parent");
  ar & CreateNVP(count, "count");
  ar & CreateNVP(pointsIndex, "pointsIndex");
  ar & CreateNVP(overlappingNode, "overlappingNode");
  ar & CreateNVP(bound, "bound");
  ar & CreateNVP(stat, "statistic");
  ar & CreateNVP(parentDistance, "parentDistance");
  ar & CreateNVP(furthestDescendantDistance, "furthestDescendantDistance");
  ar & CreateNVP(dataset, "dataset");

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
