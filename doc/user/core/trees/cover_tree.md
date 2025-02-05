# `CoverTree`

<!-- TODO: link to knn.md once it's done -->

The `CoverTree` class implements the cover tree, a hierarchical tree structure
with favorable theoretical properties.  The cover tree is useful for efficient
distance operations (such as nearest neighbor search) in low to moderate
dimensions.

mlpack's `CoverTree` implementation supports three template parameters for
configurable behavior, and implements all the functionality required by the
[TreeType API](../../../developer/trees.md#the-treetype-api), plus some
additional functionality specific to cover trees.

Due to the extra bookkeeping and complexity required to achieve its theoretical
guarantees, the `CoverTree` is often not as fast for nearest neighbor search as
the [`KDTree`](kdtree.md).  However, `CoverTree` is more flexible: it is able to
work with any [distance metric](../distances.md), not just
[`LMetric`](../distances.md#lmetric).

 * [Template parameters](#template-parameters)
 * [Constructors](#constructors)
 * [Basic tree properties](#basic-tree-properties)
 * [Bounding distances with the tree](#bounding-distances-with-the-tree)
 * [Tree traversals](#tree-traversals)
 * [Example usage](#example-usage)

## See also

<!-- TODO: add links to all distance-based algorithms and other trees? -->

 * [Cover tree on Wikipedia](https://en.wikipedia.org/wiki/Cover_tree)
 * [`KDTree`](kdtree.md)
 * [Cover trees for nearest neighbor (pdf)](https://www.hunch.net/~jl/projects/cover_tree/paper/paper.pdf)
 * [Tree-Independent Dual-Tree Algorithms (pdf)](https://www.ratml.org/pub/pdf/2013tree.pdf)

## Template parameters

The `CoverTree` class takes four template parameters, the first three of which
are required by the
[TreeType API](../../../developer/trees.md#template-parameters-required-by-the-treetype-policy)
(see also [this more detailed section](../../../developer/trees.md#template-parameters)).

```
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>
```

 * `DistanceType`: the [distance metric](../distances.md) to use for distance
   computations.  By default, this is
   [`EuclideanDistance`](../distances.md#lmetric).
 * [`StatisticType`](binary_space_tree.md#statistictype): this holds auxiliary
   information in each tree node.  By default,
   [`EmptyStatistic`](binary_space_tree.md#emptystatistic) is used, which holds
   no information.
 * `MatType`: the type of matrix used to represent points.  Must be a type
   matching the [Armadillo API](../../matrices.md).  By default, `arma::mat` is
   used, but other types such as `arma::fmat` or similar will work just fine.
 * `RootPointPolicy`: controls how the root of the tree is selected.  By
   default, `FirstPointIsRoot` is used, which simply uses the first point of the
   dataset as the root of the tree.
   - A custom `RootPointPolicy` must implement the function
     `static size_t ChooseRoot(const MatType& dataset)`, where the `size_t`
     returned indicates the index of the point in `dataset` that should be used
     as the root of the tree.

If no template parameters are explicitly specified, then defaults are used:

```
CoverTree<> = CoverTree<EuclideanDistance, EmptyStatistic, arma::mat,
                        FirstPointIsRoot>
```

## Constructors

`CoverTree`s are constructed level-by-level, without modifying the input
dataset.

---

 * `node = CoverTree(data, base=2.0)`
 * `node = CoverTree(data, distance, base=2.0)`
   - Construct a `CoverTree` on the given `data`, using the given `base` if
     specified.
   - Optionally, specify an instantiated distance metric `distance` to use to
     construct the tree.

---

***Notes:***

 - The name `node` is used here for `CoverTree` objects instead of `tree`,
   because each `CoverTree` object is a single node in the tree.  The
   constructor returns the node that is the root of the tree.

 - In a `CoverTree`, it is not guaranteed that the ball bounds for nodes are
   disjoint; they may be overlapping.  This is because for many datasets, it is
   geometrically impossible to construct disjoint balls that cover the
   entire set of points.

 - Inserting individual points or removing individual points from a `CoverTree`
   is not supported, because this generally results in a cover tree with very
   loose bounding balls.  It is better to simply build a new `CoverTree` on the
   modified dataset.  For trees that support individual insertion and deletions,
   see the `RectangleTree` class and all its variants (e.g. `RTree`,
   `RStarTree`, etc.).

 - See also the
   [developer documentation on tree constructors](../../../developer/trees.md#constructors-and-destructors).

<!-- TODO: add links to RectangleTree above when it is documented -->

---

### Constructor parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../../matrices.md) | [Column-major](../../matrices.md#representing-data-in-mlpack) matrix to build the tree on.  Optionally pass with `std::move(data)` to transfer ownership to the tree. | _(N/A)_ |
| `distance` | [`DistanceType`](#template-parameters) | Instantiated distance metric (optional). | `EuclideanDistance()` |
| `base` | `double` | Shrinkage factor of each level of the cover tree.  Must be greater than 1. | `2.0` |

***Notes:***

 - According to [the original paper (pdf)](https://www.hunch.net/~jl/projects/cover_tree/paper/paper.pdf),
   sometimes a smaller base (more like 1.3 or 1.5) can provide better empirical
   results in practice.

 - An instantiated `distance` is only necessary when a
   [custom `DistanceType`](#template-parameters) was specified as a template
   parameter, and that distance type that require state.  So, this is not needed
   when using the default `EuclideanDistance`.

## Basic tree properties

Once a `CoverTree` object is constructed, various properties of the tree can be
accessed or inspected.  Many of these functions are required by the [TreeType
API](../../../developer/trees.md#the-treetype-api).

### Navigating the tree

 * `node.NumChildren()` returns the number of children in `node`.  If `0`, then
   `node` is a leaf.

 * `node.IsLeaf()` returns a `bool` indicating whether or not `node` is a leaf.

 * `node.Child(i)` returns a `CoverTree&` that is the `i`th child.
   - This function should only be called if `node.NumChildren()` is not `0`
     (e.g. if `node` is not a leaf).  Note that this returns a valid
     `CoverTree&` that can itself be used just like the root node of the tree!

 * `node.Parent()` will return a `CoverTree*` that points to the parent of
   `node`, or `NULL` if `node` is the root of the `CoverTree`.

 * `node.Base()` will return the value of `base` used to
   [build the tree](#constructors).  This is the same for all nodes in a tree.

 * `node.Scale()` will return an `int` representing the level of the node in the
   cover tree.  Larger values represent higher levels in the tree, and `INT_MIN`
   means that `node` is a leaf.
   - All descendant points are contained within a distance of
     `node.Base()` raised to a power of `node.Scale()`.

---

### Accessing members of a tree

 * `node.Stat()` will return an `EmptyStatistic&` (or a `StatisticType&` if a
   [custom `StatisticType`](#template-parameters) was specified as a template
   parameter) holding the statistics of the node that were computed during tree
   construction.

 * `node.Distance()` will return a
   [`EuclideanDistance&`](../distances.md#lmetric) (or a `DistanceType&` if a
   [custom `DistanceType`](#template-parameters) was specified as a template
   parameter).

See also the
[developer documentation](../../../developer/trees.md#basic-tree-functionality)
for basic tree functionality in mlpack.

---

### Accessing data held in a tree

 * `node.Dataset()` will return a `const arma::mat&` that is the dataset the
   tree was built on.
   - If a [custom `MatType`](#template-parameters) is being used, the return
     type will be `const MatType&` instead of `const arma::mat&`.

 * `node.NumPoints()` returns `1`: all cover tree nodes hold only one point.

 * `node.Point()` returns a `size_t` indicating the index of the point held by
   `node` in `node.Dataset()`.
   - For consistency with other tree types, `node.Point(i)` is also available,
     but `i` must be `0` (because cover tree nodes must hold only one point).
   - The point in `node` can then be accessed as
     `node.Dataset().col(node.Point())`.

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
   - Descendant point indices are not necessarily contiguous for cover trees;
     that is, `node.Descendant(i) + 1` is not necessarily
     `node.Descendant(i + 1)`.

---

### Accessing computed bound quantities of a tree

The following quantities are cached for each node in a `CoverTree`, and so
accessing them does not require any computation.

 * `node.FurthestPointDistance()` returns a `double` representing the distance
   between the center of the bounding ball of `node` and the furthest point held
   by `node`.  This value is always `0` for cover trees, as they only hold one
   point, which is the center of the bounding ball.

 * `node.FurthestDescendantDistance()` returns a `double` representing the
   distance between the center of the bounding ball of `node` and the furthest
   descendant point held by `node`.
   - This will always be less than `node.Base()` raised to the power of
     `node.Scale()`.

 * `node.MinimumBoundDistance()` returns a `double` representing the minimum
   possible distance from the center of the node to any edge of the bounding
   ball of `node`.
   - For cover trees, this quantity is equivalent to
     `node.FurthestDescendantDistance()`.

 * `node.ParentDistance()` returns a `double` representing the distance between
   the center of the bounding ball of `node` and the center of the bounding ball
   of its parent.
   - This is equivalent to the distance between
     `node.Dataset().col(node.Point())` and
     `node.Dataset().col(node.Parent()->Point())`, if `node` is not the root of
     the tree.
   - If `node` is the root of the tree, `0` is returned.

***Notes:***

 - If a [custom `MatType`](#template-parameters) was specified when constructing
   the `CoverTree`, then the return type of each method is the element type of
   the given `MatType` instead of `double`.  (e.g., if `MatType` is
   `arma::fmat`, then the return type is `float`.)

 - For more details on each bound quantity, see the
   [developer documentation](../../../developer/trees.md#complex-tree-functionality-and-bounds)
   on bound quantities for trees.

---

### Other functionality

 * `node.Center(center)` stores the center of the bounding ball of `node` in
   `center`.
   - `center` should be of type `arma::vec&`.  (If a [custom
     `MatType`](#template-parameters) was specified when constructing the
     `CoverTree`, the type is instead the column vector type for the given
     `MatType`; e.g., `arma::fvec&` when `MatType` is `arma::fmat`.)
   - `center` will be set to have size equivalent to the dimensionality of the
     dataset held by `node`.
   - For cover trees, this sets `center` to have the same values as
     `node.Dataset().col(node.Point())` (e.g. the point held by `node`).

 * A `CoverTree` can be serialized with
   [`data::Save()` and `data::Load()`](../../load_save.md#mlpack-objects).

## Bounding distances with the tree

The primary use of trees in mlpack is bounding distances to points or other tree
nodes.  The following functions can be used for these tasks.

 * `node.GetNearestChild(point)`
 * `node.GetFurthestChild(point)`
   - Return a `size_t` indicating the index of the child that is closest to (or
     furthest from) `point`, with respect to the `MinDistance()` (or
     `MaxDistance()`) function.
   - If there is a tie, the child with the highest index is returned.
   - If `node` is a leaf, `0` is returned.
   - `point` should be of type `arma::vec`.  (If a [custom
     `MatType`](#template-parameters) was specified when constructing the
     `CoverTree`, the type is instead the column vector type for the given
     `MatType`; e.g., `arma::fvec` when `MatType` is `arma::fmat`.)

 * `node.GetNearestChild(other)`
 * `node.GetFurthestChild(other)`
   - Return a `size_t` indicating the index of the child that is closest to (or
     furthest from) the `CoverTree` node `other`, with respect to the
     `MinDistance()` (or `MaxDistance()`) function.
   - If there is a tie, the child with the highest index is returned.
   - If `node` is a leaf, `0` is returned.

---

 * `node.MinDistance(point)`
 * `node.MinDistance(other)`
   - Return a `double` indicating the minimum possible distance between `node`
     and `point`, or the `CoverTree` node `other`.
   - This is equivalent to the minimum possible distance between any point
     contained in the bounding ball of `node` and `point`, or between any point
     contained in the bounding ball of `node` and any point contained in the
     bounding ball of `other`.
   - `point` should be of type `arma::vec`.  (If a [custom
     `MatType`](#template-parameters) was specified when constructing the
     `CoverTree`, the type is instead the column vector type for the given
     `MatType`, and the return type is the element type of `MatType`; e.g.,
     `point` should be `arma::fvec` when `MatType` is `arma::fmat`, and the
     returned distance is `float`).

 * `node.MaxDistance(point)`
 * `node.MaxDistance(other)`
   - Return a `double` indicating the maximum possible distance between `node`
     and `point`, or the `CoverTree` node `other`.
   - This is equivalent to the maximum possible distance between any point
     contained in the bounding ball of `node` and `point`, or between any point
     contained in the bounding ball of `node` and any point contained in the
     bounding ball of `other`.
   - `point` should be of type `arma::vec`.  (If a [custom
     `MatType`](#template-parameters) was specified when constructing the
     `CoverTree`, the type is instead the column vector type for the given
     `MatType`, and the return type is the element type of `MatType`; e.g.,
     `point` should be `arma::fvec` when `MatType` is `arma::fmat`, and the
     returned distance is `float`).

 * `node.RangeDistance(point)`
 * `node.RangeDistance(other)`
   - Return a [`Range`](../math.md#range) whose lower bound is
     `node.MinDistance(point)` or `node.MinDistance(other)`, and whose upper
      bound is `node.MaxDistance(point)` or `node.MaxDistance(other)`.
   - `point` should be of type `arma::vec`.  (If a
     [custom `MatType`](#template-parameters) was specified when constructing
     the `CoverTree`, the type is instead the column vector type for the given
     `MatType`, and the return type is a `RangeType` with element type the same
     as `MatType`; e.g., `point` should be `arma::fvec` when `MatType` is
     `arma::fmat`, and the returned type is
     [`RangeType<float>`](../math.md#range)).

## Tree traversals

Like every mlpack tree, the `CoverTree` class provides a [single-tree and
dual-tree traversal](../../../developer/trees.md#traversals) that can be paired
with a [`RuleType` class](../../../developer/trees.md#rules) to implement a
single-tree or dual-tree algorithm.

 * `CoverTree::SingleTreeTraverser`
   - Implements a breadth-first single-tree traverser: each level (scale) of the
     tree is visited, and base cases are computed and nodes are pruned before
     descending to the next level.

 * `CoverTree::DualTreeTraverser`
   - Implements a joint depth-first and breadth-first traversal as in the
     [original paper (pdf)](https://www.hunch.net/~jl/projects/cover_tree/paper/paper.pdf).
   - The query tree is descended in a depth-first manner; the reference tree is
     descended level-wise in a breadth-first manner, pruning node combinations
     where possible.
   - The level of the query tree and reference tree are held as even as possible
     during the traversal; so, in general, query and reference recursions will
     alternate.

## Example usage

Build a `CoverTree` on the `cloud` dataset and print basic statistics about the
tree.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset, true);

// Build the cover tree with default options.
//
// The std::move() means that `dataset` will be empty after this call, and the
// tree will "own" the dataset.  No data will be copied during tree building,
// regardless of whether we used `std::move()`.
//
// Note that the '<>' isn't necessary if C++20 is being used (e.g.
// `mlpack::CoverTree tree(...)` will work fine in C++20 or newer).
mlpack::CoverTree<> tree(std::move(dataset));

// Print the point held by the root node and the radius of the ball that
// contains all points:
std::cout << "Root node:" << std::endl;
std::cout << " - Base: " << tree.Base() << "." << std::endl;
std::cout << " - Scale: " << tree.Scale() << "." << std::endl;
std::cout << " - Point: " << tree.Dataset().col(tree.Point()).t();
std::cout << std::endl;

// Print the number of descendant points of the root, and of each of its
// children.
std::cout << "Descendant points of root:        "
    << tree.NumDescendants() << "." << std::endl;
std::cout << "Number of children of root: " << tree.NumChildren() << "."
    << std::endl;
for (size_t c = 0; c < tree.NumChildren(); ++c)
{
  std::cout << " - Descendant points of child " << c << ": "
      << tree.Child(c).NumDescendants() << "." << std::endl;
}
```

---

Build two `CoverTree`s on subsets of the corel dataset and compute minimum and
maximum distances between different nodes in the tree.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::mat dataset;
mlpack::data::Load("corel-histogram.csv", dataset, true);

// Build cover trees on the first half and the second half of points.
mlpack::CoverTree<> tree1(dataset.cols(0, dataset.n_cols / 2));
mlpack::CoverTree<> tree2(dataset.cols(dataset.n_cols / 2 + 1,
    dataset.n_cols - 1));

// Compute the maximum distance between the trees.
std::cout << "Maximum distance between tree root nodes: "
    << tree1.MaxDistance(tree2) << "." << std::endl;

// Get a grandchild of the first tree's root---if it exists.
if (!tree1.IsLeaf() && !tree1.Child(0).IsLeaf())
{
  mlpack::CoverTree<>& node1 = tree1.Child(0).Child(0);

  // Get a grandchild of the second tree's root---if it exists.
  if (!tree2.IsLeaf() && !tree2.Child(0).IsLeaf())
  {
    mlpack::CoverTree<>& node2 = tree2.Child(0).Child(0);

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
    const size_t closestIndex = node2.GetNearestChild(node1);
    std::cout << "Child " << closestIndex << " of node2 is closest to node1."
        << std::endl;

    // And which child of node1 is further from node2?
    const size_t furthestIndex = node1.GetFurthestChild(node2);
    std::cout << "Child " << furthestIndex << " of node1 is furthest from "
        << "node2." << std::endl;
  }
}
```

---

Build a `CoverTree` on 32-bit floating point data and save it to disk.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::fmat dataset;
mlpack::data::Load("corel-histogram.csv", dataset);

// Build the CoverTree using 32-bit floating point data as the matrix type.
// We will still use the default EmptyStatistic and EuclideanDistance
// parameters.
mlpack::CoverTree<mlpack::EuclideanDistance,
                  mlpack::EmptyStatistic,
                  arma::fmat> tree(dataset);

// Save the CoverTree to disk with the name 'tree'.
mlpack::data::Save("tree.bin", "tree", tree);

std::cout << "Saved tree with " << tree.Dataset().n_cols << " points to "
    << "'tree.bin'." << std::endl;
```

---

Load a 32-bit floating point `CoverTree` from disk, then traverse it manually
and find the number of leaf nodes with fewer than 10 points.

```c++
// This assumes the tree has already been saved to 'tree.bin' (as in the example
// above).

// This convenient typedef saves us a long type name!
using TreeType = mlpack::CoverTree<mlpack::EuclideanDistance,
                                   mlpack::EmptyStatistic,
                                   arma::fmat>;

TreeType tree;
mlpack::data::Load("tree.bin", "tree", tree);
std::cout << "Tree loaded with " << tree.NumDescendants() << " points."
    << std::endl;

// Recurse in a depth-first manner.  Count both the total number of leaves, and
// the number of nodes with more than 100 descendants.
size_t moreThan100Count = 0;
size_t totalLeafCount = 0;
std::stack<TreeType*> stack;
stack.push(&tree);
while (!stack.empty())
{
  TreeType* node = stack.top();
  stack.pop();

  if (node->NumDescendants() > 100)
    ++moreThan100Count;

  if (node->IsLeaf())
    ++totalLeafCount;

  for (size_t c = 0; c < node->NumChildren(); ++c)
    stack.push(&node->Child(c));
}

// Note that it would be possible to use TreeType::SingleTreeTraverser to
// perform the recursion above, but that is more well-suited for more complex
// tasks that require pruning and other non-trivial behavior; so using a simple
// stack is the better option here.

// Print the results.
std::cout << "Tree contains " << totalLeafCount << " leaves." << std::endl;
std::cout << moreThan100Count << " nodes have more than 100 descendants."
    << std::endl;
```
