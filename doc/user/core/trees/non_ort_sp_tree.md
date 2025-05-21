# `NonOrtSPTree`

The `NonOrtSPTree` class implements the hybrid spill tree with
non-axis-orthogonal splitting hyperplanes; this is a binary space partitioning
tree that allows overlapping volumes between nodes.  This type of tree can be
more effective than trees like the [`KDTree`](kdtree.md) for approximate nearest
neighbor search and related tasks.

`NonOrtSPTree` supports three template parameters for configurable behavior, and
implements all the functionality required by the [TreeType
API](../../../developer/trees.md#the-treetype-api), plus some additional
functionality specific to spill trees.  `NonOrtSPTree` is built on the more
generic [`SpillTree`](spill_tree.md) class, so if fully custom behavior is
desired, that

 * [Template parameters](#template-parameters)
 * [Constructors](#constructors)
 * [Basic tree properties](#basic-tree-properties)
 * [Bounding distances with the tree](#bounding-distances-with-the-tree)
 * [Tree traversals](#tree-traversals)
 * [Example usage](#example-usage)

## See also

<!-- TODO: add links to all distance-based algorithms and other trees? -->

 * [`SpillTree`](spill_tree.md)
 * [`SPTree`](sp_tree.md)
 * [`MeanSPTree`](mean_sp_tree.md)
 * [`NonOrtMeanSPTree`](non_ort_mean_sp_tree.md)
 * [`BinarySpaceTree`](binary_space_tree.md)
 * [An Investigation of Practical Approximate Nearest Neighbor Algorithms (pdf)](https://proceedings.neurips.cc/paper/2004/file/1102a326d5f7c9e04fc3c89d0ede88c9-Paper.pdf)
 * [Tree-Independent Dual-Tree Algorithms (pdf)](https://www.ratml.org/pub/pdf/2013tree.pdf)

## Template parameters

In accordance with the
[TreeType API](../../../developer/trees.md#template-parameters-required-by-the-treetype-policy)
(see also
[this more detailed section](../../../developer/trees.md#template-parameters)),
the `NonOrtSPTree` class takes three template parameters:

```
NonOrtSPTree<DistanceType, StatisticType, MatType>
```

 * `DistanceType`: the [distance metric](../distances.md) to use for distance
   computations.

 * `StatisticType`: this holds auxiliary information in each tree node.  By
   default, [`EmptyStatistic`](binary_space_tree.md#emptystatistic) is used,
   which holds no information.
   - See the [`StatisticType`](binary_space_tree.md#statistictype) section in
     the `BinarySpaceTree` documentation for more details.

 * `MatType`: the type of matrix used to represent points.  Must be a type
   matching the [Armadillo API](../../matrices.md).  By default, `arma::mat` is
   used, but other types such as `arma::fmat` or similar will work just fine.

The `NonOrtSPTree` class itself is a convenience typedef of the generic
[`SpillTree`](spill_tree.md) class, using the
[`Hyperplane`](spill_tree.md#hyperplane) class as the splitting hyperplane type,
and the [`MidpointSpaceSplit`](spill_tree.md#midpointspacesplit) class as the
splitting strategy.

If no template parameters are explicitly specified, then defaults are used:

```
NonOrtSPTree<> = NonOrtSPTree<EuclideanDistance, EmptyStatistic, arma::mat>
```

## Constructors

`NonOrtSPTree`s are constructed by iteratively finding splitting hyperplanes,
and points within a margin of the hyperplane are assigned to *both* child nodes.
Unlike the constructors of
[`BinarySpaceTree`](binary_space_tree.md#constructors), the dataset is not
permuted during construction.

---

 * `node = NonOrtSPTree(data, tau=0.0, maxLeafSize=20, rho=0.7)`
   - Construct a `NonOrtSPTree` on the given `data`, using the specified
     hyperparameters to control tree construction behavior.
   - By default, a reference to `data` is stored.  If `data` goes out of scope
     after tree construction, memory errors will occur!  To avoid this, either
     pass the dataset or a copy with `std::move()` (e.g. `std::move(data)`);
     when doing this, `data` will be set to an empty matrix.

---

 * `node = NonOrtSPTree<DistanceType, StatisticType, MatType>(data, tau=0.0, maxLeafSize=20, rho=0.7)`
   - Construct a `NonOrtSPTree` on the given `data`, using custom template
     parameters, and using the specified hyperparameters to control tree
     construction behavior.
   - By default, a reference to `data` is stored.  If `data` goes out of scope
     after tree construction, memory errors will occur!  To avoid this, either
     pass the dataset or a copy with `std::move()` (e.g. `std::move(data)`);
     when doing this, `data` will be set to an empty matrix.

---

 * `node = NonOrtSPTree()`
   - Construct an empty `NonOrtSPTree` with no children, no points, and default
     template parameters.

---

***Notes:***

 - The name `node` is used here for `NonOrtSPTree` objects instead of `tree`,
   because each `NonOrtSPTree` object is a single node in the tree.  The
   constructor returns the node that is the root of the tree.

 - Inserting individual points or removing individual points from a
   `NonOrtSPTree` is not supported, because this generally results in a tree
   with very suboptimal hyperplane splits.  It is better to simply build a new
   `NonOrtSPTree` on the modified dataset.  For trees that support individual
   insertion and deletions, see the [`RectangleTree`](rectangle_tree.md) class
   and all its variants (e.g.  [`RTree`](r_tree.md),
   [`RStarTree`](r_star_tree.md), etc.).

 - See also the
   [developer documentation on tree constructors](../../../developer/trees.md#constructors-and-destructors).

---

### Constructor parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`MatType`](../../matrices.md) | [Column-major](../../matrices.md#representing-data-in-mlpack) matrix to build the tree on. | _(N/A)_ |
| `tau` | `double` | Width of spill margin: points within `tau` of the splitting hyperplane of a node will be contained in both left and right children. | `0.0` |
| `maxLeafSize` | `size_t` | Maximum number of points to store in each leaf. | `20` |
| `rho` | `double` | Balance threshold. When splitting, if either overlapping node would contain a fraction of more than `rho` of the points, a non-overlapping split is performed. Must be in the range `[0.0, 1.0)`. | `0.7` |

***Caveats***:

 * `tau` must be manually tuned for the properties of each dataset; the default,
   `0.0`, will never allow overlap between nodes (and thus the created tree will
   essentially be a non-overlapping [`BinarySpaceTree`](binary_space_tree.md)).

 * If `tau` is set too large, nodes will overlap too much and search quality
   will be degraded.

 * `rho` implicitly controls the depth of the tree by forcing very overlapping
   children to be non-overlapping.  As `rho` gets closer to `1`, more overlap is
   allowed, which in turn makes the tree deeper.  If `rho` is set to `0.5` or
   less, then all splits will be non-overlapping (and the tree will essentially
   be a [`BinarySpaceTree`](binary_space_tree.md)).

## Basic tree properties

Once an `NonOrtSPTree` object is constructed, various properties of the tree can
be accessed or inspected.  Many of these functions are required by the
[TreeType API](../../../developer/trees.md#the-treetype-api).

### Navigating the tree

 * `node.NumChildren()` returns the number of children in `node`.  This is
   either `2` if `node` has children, or `0` if `node` is a leaf.

 * `node.IsLeaf()` returns a `bool` indicating whether or not `node` is a leaf.

 * `node.Child(i)` returns a `NonOrtSPTree&` that is the `i`th child.
   - `i` must be `0` or `1`.
   - This function should only be called if `node.NumChildren()` is not `0`
     (e.g. if `node` is not a leaf).  Note that this returns a valid
     `NonOrtSPTree&` that can itself be used just like the root node of the
     tree!
   - `node.Left()` and `node.Right()` are convenience functions specific to
     `NonOrtSPTree` that will return `NonOrtSPTree*` (pointers) to the left and
     right children, respectively, or `NULL` if `node` has no children.

 * `node.Parent()` will return an `NonOrtSPTree*` that points to the parent of
   `node`, or `NULL` if `node` is the root of the `NonOrtSPTree`.

---

### Accessing members of a tree

 * `node.Overlap()` will return a `bool` that is `true` if `node`'s children are
   overlapping, and `false` otherwise.

 * `node.Hyperplane()` will return an [`Hyperplane`](spill_tree.md#hyperplane)
   object that represents the splitting hyperplane of `node`.
   - All points in `node.Left()` are to the left of `node.Hyperplane()` if
     `node.Overlap()` is `false`; otherwise, all points in `node.Left()` are to
     the left of `node.Hyperplane() + tau`.
   - All points in `node.Right()` are to the right of `node.Hyperplane()` if
     `node.Overlap()` is `false`; otherwise, all points in `node.Right()` are to
     the right of `node.Hyperplane() - tau`.

 * `node.Bound()` will return a
   [`const BallBound&`](binary_space_tree.md#ballbound) representing the
   bounding box associated with `node`.
   - If a [custom `DistanceType` and/or `MatType`](#template-parameters) are
     specified, then a `const BallBound<DistanceType, ElemType>&` is returned.
     * `ElemType` is the element type of the specified `MatType` (e.g. `double`
       for `arma::mat`, `float` for `arma::fmat`, etc.).

 * `node.Stat()` will return a `StatisticType&` holding the statistics of the
   node that were computed during tree construction.

 * `node.Distance()` will return a `DistanceType&` that can be used to make
   distance computations.

See also the
[developer documentation](../../../developer/trees.md#basic-tree-functionality)
for basic tree functionality in mlpack.

---

### Accessing data held in a tree

 * `node.Dataset()` will return a `const MatType&` that is the dataset the
   tree was built on.

 * `node.NumPoints()` returns a `size_t` indicating the number of points held
   directly in `node`.
   - If `node` is not a leaf, this will return `0`, as `NonOrtSPTree` only holds
     points directly in its leaves.
   - If `node` is a leaf, then the number of points will be less than or equal
     to the `maxLeafSize` that was specified when the tree was constructed.

 * `node.Point(i)` returns a `size_t` indicating the index of the `i`'th point
   in `node.Dataset()`.
   - `i` must be in the range `[0, node.NumPoints() - 1]` (inclusive).
   - `node` must be a leaf (as non-leaves do not hold any points).
   - The `i`'th point in `node` can then be accessed as
     `node.Dataset().col(node.Point(i))`.
   - Accessing the actual `i`'th point itself can be done with, e.g.,
     `node.Dataset().col(node.Point(i))`.

 * `node.NumDescendants()` returns a `size_t` indicating the number of points
   held in all descendant leaves of `node`.
   - If `node` is the root of the tree, then `node.NumDescendants()` will be
     equal to `node.Dataset().n_cols`.

 * `node.Descendant(i)` returns a `size_t` indicating the index of the `i`'th
   descendant point in `node.Dataset()`.
   - `i` must be in the range `[0, node.NumDescendants() - 1]` (inclusive).
   - `node` does not need to be a leaf.
   - The `i`'th descendant point in `node` can then be accessed as
     `node.Dataset().col(node.Descendant(i))`.
   - Accessing the actual `i`'th descendant itself can be done with, e.g.,
     `node.Dataset().col(node.Descendant(i))`.

---

### Accessing computed bound quantities of a tree

The following quantities are cached for each node in a `NonOrtSPTree`, and so
accessing them does not require any computation.  In the documentation below,
`ElemType` is the element type of the given `MatType`; e.g., if `MatType` is
`arma::mat`, then `ElemType` is `double`.

 * `node.FurthestPointDistance()` returns an `ElemType` representing the
   distance between the center of the bound of `node` and the furthest point
   held by `node`.
   - If `node` is not a leaf, this returns 0 (because `node` does not hold any
     points).

 * `node.FurthestDescendantDistance()` returns an `ElemType` representing the
   distance between the center of the bound of `node` and the furthest
   descendant point held by `node`.

 * `node.MinimumBoundDistance()` returns an `ElemType` representing the minimum
   possible distance from the center of the node to any edge of its bound.

 * `node.ParentDistance()` returns an `ElemType` representing the distance
   between the center of the bound of `node` and the center of the bound of its
   parent.
   - If `node` is the root of the tree, `0` is returned.

***Note:*** for more details on each bound quantity, see the [developer
documentation](../../../developer/trees.md#complex-tree-functionality-and-bounds)
on bound quantities for trees.

---

### Other functionality

 * `node.Center(center)` computes the center of the bound of `node` and stores
   it in `center`.
   - `center` should be of type `arma::Col<ElemType>&`, where `ElemType` is the
     element type of the specified `MatType`.
   - `center` will be set to have size equivalent to the dimensionality of the
     dataset held by `node`.
   - This is equivalent to calling `node.Bound().Center(center)`.

 * A `NonOrtSPTree` can be serialized with
   [`data::Save()` and `data::Load()`](../../load_save.md#mlpack-objects).

## Bounding distances with the tree

The primary use of trees in mlpack is bounding distances to points or other tree
nodes.  The following functions can be used for these tasks.

 * `node.GetNearestChild(point)`
 * `node.GetFurthestChild(point)`
   - Return a `size_t` indicating the index of the child (`0` for left, `1` for
     right) that is closest to (or furthest from) `point`, with respect
     to the `MinDistance()` (or `MaxDistance()`) function.
   - If there is a tie, `0` (the left child) is returned.
   - If `node` is a leaf, `0` is returned.
   - `point` should be a column vector type of the same type as `MatType`.
     (e.g., if `MatType` is `arma::mat`, then `point` should be an `arma::vec`.)

 * `node.GetNearestChild(other)`
 * `node.GetFurthestChild(other)`
   - Return a `size_t` indicating the index of the child (`0` for left, `1` for
     right) that is closest to (or furthest from) the `NonOrtSPTree` node
     `other`, with respect to the `MinDistance()` (or `MaxDistance()`) function.
   - If there is a tie, `0` (the left child) is returned.
   - If `node` is a leaf, `0` is returned.

---

 * `node.MinDistance(point)`
 * `node.MinDistance(other)`
   - Return a `double` indicating the minimum possible distance between `node`
     and `point`, or the `NonOrtSPTree` node `other`.
   - This is equivalent to the minimum possible distance between any point
     contained in the bounding hyperrectangle of `node` and `point`, or between
     any point contained in the bounding hyperrectangle of `node` and any point
     contained in the bounding hyperrectangle of `other`.
   - `point` should be a column vector type of the same type as `MatType`.
     (e.g., if `MatType` is `arma::mat`, then `point` should be an `arma::vec`.)

 * `node.MaxDistance(point)`
 * `node.MaxDistance(other)`
   - Return a `double` indicating the maximum possible distance between `node`
     and `point`, or the `NonOrtSPTree` node `other`.
   - This is equivalent to the maximum possible distance between any point
     contained in the bounding hyperrectangle of `node` and `point`, or between
     any point contained in the bounding hyperrectangle of `node` and any point
     contained in the bounding hyperrectangle of `other`.
   - `point` should be a column vector type of the same type as `MatType`.
     (e.g., if `MatType` is `arma::mat`, then `point` should be an `arma::vec`.)

 * `node.RangeDistance(point)`
 * `node.RangeDistance(other)`
   - Return a [`RangeType<ElemType>`](../math.md#range) whose lower bound is
     `node.MinDistance(point)` or `node.MinDistance(other)`, and whose upper
      bound is `node.MaxDistance(point)` or `node.MaxDistance(other)`.
   - `ElemType` is the element type of `MatType`.
   - `point` should be a column vector type of the same type as `MatType`.
     (e.g., if `MatType` is `arma::mat`, then `point` should be an `arma::vec`.)

## Tree traversals

Like every mlpack tree, the `NonOrtSPTree` class provides a [single-tree and
dual-tree traversal](../../../developer/trees.md#traversals) that can be paired
with a [`RuleType` class](../../../developer/trees.md#rules) to implement a
single-tree or dual-tree algorithm.

 * `NonOrtSPTree::SingleTreeTraverser`
   - Implements a depth-first single-tree traverser.

 * `NonOrtSPTree::DualTreeTraverser`
   - Implements a dual-depth-first dual-tree traverser.

However, spill trees are primarily useful because the overlapping nodes allow
*defeatist* search to be effective.  Defeatist search is non-backtracking: the
tree is traversed to one leaf only.  For example, finding the approximate
nearest neighbor of a point `p` with defeatist search is done by recursing in
the tree, choosing the child with smallest minimum distance to `p`, and when a
leaf is encountered, choosing the closest point in the leaf to `p` as the
nearest neighbor.  This is the strategy used in the
[original spill tree paper (pdf)](https://proceedings.neurips.cc/paper/2004/file/1102a326d5f7c9e04fc3c89d0ede88c9-Paper.pdf).

Defeatist traversers, matching the API for a regular
[traversal](../../../developer/trees.md#traversals) are made available as the
following two classes:

 * `NonOrtSPTree::DefeatistSingleTreeTraverser`
   - Implements a depth-first single-tree defeatist traverser with no
     backtracking.  Traversal will terminate after the first leaf is visited.

 * `NonOrtSPTree::DefeatistDualTreeTraverser`
   - Implements a dual-depth-first dual-tree defeatist traversal with no
     backtracking.  For each query leaf node, traversal will terminate after the
     first reference leaf node is visited.

Any [`RuleType`](../../../developer/trees.md#rules) that is being used with a
defeatist traversal, in addition to the functions required by the `RuleType`
API, must implement the following functions:

```
// This is only required for single-tree defeatist traversals.
// It should return the index of the branch that should be chosen for the given
// query point and reference node.
template<typename VecType, typename TreeType>
size_t GetBestChild(const VecType& queryPoint, TreeType& referenceNode);

// This is only required for dual-tree defeatist traversals.
// It should return the index of the best child of the reference node that
// should be chosen for the given query node.
template<typename TreeType>
size_t GetBestChild(TreeType& queryNode, TreeType& referenceNode);

// Return the minimum number of base cases (point-to-point computations) that
// are required during the traversal.
size_t MinimumBaseCases();
```

## Example usage

Build an `NonOrtSPTree` on the `cloud` dataset and print basic statistics about
the tree.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset, true);

// Build the spill tree with a tau (margin) of 0.2 and a leaf size of 10.
// (This means that nodes are split until they contain 10 or fewer points.)
//
// The std::move() means that `dataset` will be empty after this call, and no
// data will be copied during tree building.
//
// When C++20 is enabled, then the <> is not necessary and the following line
// will work:
// mlpack::NonOrtSPTree tree(std::move(dataset), 0.2, 10);
mlpack::NonOrtSPTree<> tree(std::move(dataset), 0.2, 10);

// Print the bounding ball of the root node.
std::cout << "Bounding ball of root node:" << std::endl;
std::cout << "  Center: " << tree.Bound().Center().t();
std::cout << "  Radius: " << tree.Bound().Radius() << "." << std::endl;
std::cout << std::endl;

// Print the number of descendant points of the root, and of each of its
// children.
std::cout << "Descendant points of root:        "
    << tree.NumDescendants() << "." << std::endl;
std::cout << "Descendant points of left child:  "
    << tree.Left()->NumDescendants() << "." << std::endl;
std::cout << "Descendant points of right child: "
    << tree.Right()->NumDescendants() << "." << std::endl;
std::cout << std::endl;

// Compute the center of the NonOrtSPTree.  THis is the same as the center of
// the bounding ball of the root.
arma::vec center;
tree.Center(center);
std::cout << "Center of tree: " << center.t();
```

---

Build two `NonOrtSPTree`s on subsets of the corel dataset and compute minimum
and maximum distances between different nodes in the tree.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::mat dataset;
mlpack::data::Load("corel-histogram.csv", dataset, true);

// Build trees on the first half and the second half of points.  Use a tau
// (overlap) parameter of 0.3, which is tuned to this dataset, and a rho value
// of 0.6 to prevent the trees getting too deep.
mlpack::NonOrtSPTree<> tree1(dataset.cols(0, dataset.n_cols / 2), 0.3, 20, 0.6);
mlpack::NonOrtSPTree<> tree2(dataset.cols(dataset.n_cols / 2 + 1,
                                          dataset.n_cols - 1), 0.3, 20, 0.6);

// Compute the maximum distance between the trees.
std::cout << "Maximum distance between tree root nodes: "
    << tree1.MaxDistance(tree2) << "." << std::endl;

// Get the leftmost grandchild of the first tree's root---if it exists.
if (!tree1.IsLeaf() && !tree1.Child(0).IsLeaf())
{
  mlpack::NonOrtSPTree<>& node1 = tree1.Child(0).Child(0);

  // Get the rightmost grandchild of the second tree's root---if it exists.
  if (!tree2.IsLeaf() && !tree2.Child(1).IsLeaf())
  {
    mlpack::NonOrtSPTree<>& node2 = tree2.Child(1).Child(1);

    // Print the minimum and maximum distance between the nodes.
    mlpack::Range dists = node1.RangeDistance(node2);
    std::cout << "Possible distances between two grandchild nodes: ["
        << dists.Lo() << ", " << dists.Hi() << "]." << std::endl;

    // Print the minimum distance between the first node and the first
    // descendant point of the second node.
    const size_t descendantIndex = node2.Descendant(0);
    const double descendantMinDist =
        node1.MinDistance(node2.Dataset().col(descendantIndex));
    std::cout << "Minimum distance between grandchild node and descendant "
        << "point: " << descendantMinDist << "." << std::endl;

    // Which child of node2 is closer to node1?
    const size_t closerIndex = node2.GetNearestChild(node1);
    if (closerIndex == 0)
      std::cout << "The left child of node2 is closer to node1." << std::endl;
    else if (closerIndex == 1)
      std::cout << "The right child of node2 is closer to node1." << std::endl;
    else // closerIndex == 2 in this case.
      std::cout << "Both children of node2 are equally close to node1."
          << std::endl;

    // And which child of node1 is further from node2?
    const size_t furtherIndex = node1.GetFurthestChild(node2);
    if (furtherIndex == 0)
      std::cout << "The left child of node1 is further from node2."
          << std::endl;
    else if (furtherIndex == 1)
      std::cout << "The right child of node1 is further from node2."
          << std::endl;
    else // furtherIndex == 2 in this case.
      std::cout << "Both children of node1 are equally far from node2."
          << std::endl;
  }
}
```

---

Build a `NonOrtSPTree` on 32-bit floating point data and save it to disk.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::fmat dataset;
mlpack::data::Load("corel-histogram.csv", dataset);

// Build the NonOrtSPTree using 32-bit floating point data as the matrix type.
// We will still use the default EmptyStatistic and EuclideanDistance
// parameters.
mlpack::NonOrtSPTree<mlpack::EuclideanDistance,
                     mlpack::EmptyStatistic,
                     arma::fmat> tree(std::move(dataset), 0.1, 20, 0.6);

// Save the tree to disk with the name 'tree'.
mlpack::data::Save("tree.bin", "tree", tree);

std::cout << "Saved tree with " << tree.Dataset().n_cols << " points to "
    << "'tree.bin'." << std::endl;
```

---

Load a 32-bit floating point `NonOrtSPTree` from disk, then traverse it manually
and find the number of nodes whose children overlap.

```c++
// This assumes the tree has already been saved to 'tree.bin' (as in the example
// above).

// This convenient typedef saves us a long type name!
using TreeType = mlpack::NonOrtSPTree<mlpack::EuclideanDistance,
                                      mlpack::EmptyStatistic,
                                      arma::fmat>;

TreeType tree;
mlpack::data::Load("tree.bin", "tree", tree);
std::cout << "Tree loaded with " << tree.NumDescendants() << " points."
    << std::endl;

// Recurse in a depth-first manner.  Count both the total number of non-leaves,
// and the number of non-leaves that have overlapping children.
size_t overlapCount = 0;
size_t totalInternalNodeCount = 0;
std::stack<TreeType*> stack;
stack.push(&tree);
while (!stack.empty())
{
  TreeType* node = stack.top();
  stack.pop();

  if (node->IsLeaf())
    continue;

  if (node->Overlap())
    ++overlapCount;
  ++totalInternalNodeCount;

  stack.push(node->Left());
  stack.push(node->Right());
}

// Note that it would be possible to use TreeType::SingleTreeTraverser to
// perform the recursion above, but that is more well-suited for more complex
// tasks that require pruning and other non-trivial behavior; so using a simple
// stack is the better option here.

// Print the results.
std::cout << overlapCount << " out of " << totalInternalNodeCount
    << " internal nodes have overlapping children." << std::endl;
```

---

Use a defeatist traversal to find the approximate nearest neighbor of the third
and fourth points in the `corel-histogram` dataset.  (Note: this can also be
done more easily with the `KNN` class!  This example is a demonstration of how
to use the defeatist traverser.)

<!-- TODO: link to KNN class -->

For this example, we must first define a
[`RuleType` class](../../../developer/trees.md#rules).

```c++
// For simplicity, this only implements those methods required by single-tree
// traversals, and cannot be used with a dual-tree traversal.
//
// `.Reset()` must be called before any additional single-tree traversals after
// the first is run.
class SpillNearestNeighborRule
{
 public:
  // Store the dataset internally.
  SpillNearestNeighborRule(const arma::mat& dataset) :
      dataset(dataset),
      nearestNeighbor(size_t(-1)),
      nearestDistance(DBL_MAX) { }

  // Compute the base case (point-to-point comparison).
  double BaseCase(const size_t queryIndex, const size_t referenceIndex)
  {
    // Skip the base case if the points are the same.
    if (queryIndex == referenceIndex)
      return 0.0;

    const double dist = mlpack::EuclideanDistance::Evaluate(
        dataset.col(queryIndex), dataset.col(referenceIndex));

    if (dist < nearestDistance)
    {
      nearestNeighbor = referenceIndex;
      nearestDistance = dist;
    }

    return dist;
  }

  // Score the given node in the tree; if it is sufficiently far away that it
  // cannot contain a better nearest neighbor candidate, we can prune it.
  template<typename TreeType>
  double Score(const size_t queryIndex, const TreeType& referenceNode) const
  {
    const double minDist = referenceNode.MinDistance(dataset.col(queryIndex));
    if (minDist > nearestDistance)
      return DBL_MAX; // Prune: this cannot contain a better candidate!

    return minDist;
  }

  // Rescore the given node/point combination.  Note that this will not be used
  // by the defeatist traversal as it never backtracks, but we include it for
  // completeness because the RuleType API requires it.
  template<typename TreeType>
  double Rescore(const size_t, const TreeType&, const double oldScore) const
  {
    if (oldScore > nearestDistance)
      return DBL_MAX; // Prune: the node is too far away.
    return oldScore;
  }

  // This is required by defeatist traversals to select the best reference
  // child to recurse into for overlapping nodes.
  template<typename TreeType>
  size_t GetBestChild(const size_t queryIndex, TreeType& referenceNode)
      const
  {
    return referenceNode.GetNearestChild(dataset.col(queryIndex));
  }

  // We must perform at least two base cases in order to have a result.  Note
  // that this is two, and not one, because we skip base cases where the query
  // and reference points are the same.  That can only happen a maximum of once,
  // so to ensure that we compare a query point to a different reference point
  // at least once, we must return 2 here.
  size_t MinimumBaseCases() const { return 2; }

  // Get the results (to be called after the traversal).
  size_t NearestNeighbor() const { return nearestNeighbor; }
  double NearestDistance() const { return nearestDistance; }

  // Reset the internal statistics for an additional traversal.
  void Reset()
  {
    nearestNeighbor = size_t(-1);
    nearestDistance = DBL_MAX;
  }

 private:
  const arma::mat& dataset;

  size_t nearestNeighbor;
  double nearestDistance;
};
```

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::mat dataset;
mlpack::data::Load("corel-histogram.csv", dataset, true);

// Build two trees, one with a lot of overlap, and one with no overlap
// (e.g. tau = 0).
mlpack::NonOrtSPTree<> tree1(dataset, 0.5, 10), tree2(dataset, 0.0, 10);

// Construct the rule types, and then the traversals.
SpillNearestNeighborRule r1(dataset), r2(dataset);

mlpack::NonOrtSPTree<>::DefeatistSingleTreeTraverser<SpillNearestNeighborRule>
    t1(r1), t2(r2);

// Search for the approximate nearest neighbor of point 3 using both trees.
t1.Traverse(3, tree1);
t2.Traverse(3, tree2);

std::cout << "Approximate nearest neighbor of point 3:" << std::endl;
std::cout << " - Non-axis-aligned spill tree with overlap 0.5 found: point "
    << r1.NearestNeighbor() << ", distance " << r1.NearestDistance()
    << "." << std::endl;

std::cout << " - Non-axis-aligned spill tree with no overlap found: point "
    << r2.NearestNeighbor() << ", distance " << r2.NearestDistance()
    << "." << std::endl;

// Now search for point 6.
r1.Reset();
r2.Reset();

t1.Traverse(6, tree1);
t2.Traverse(6, tree2);

std::cout << "Approximate nearest neighbor of point 6:" << std::endl;
std::cout << " - Non-axis-aligned spill tree with overlap 0.5 found: point "
    << r1.NearestNeighbor() << ", distance " << r1.NearestDistance()
    << "." << std::endl;

std::cout << " - Non-axis-aligned spill tree with no overlap found: point "
    << r2.NearestNeighbor() << ", distance " << r2.NearestDistance()
    << "." << std::endl;
```
