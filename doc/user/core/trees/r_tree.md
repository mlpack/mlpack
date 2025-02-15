# `RTree`

The `RTree` class implements the R tree, a well-known multidimensional space
partitioning tree that can insert and remove points dynamically.

The `RTree` implementation in mlpack supports three template parameters for
configurable behavior, and implements all the functionality required by the
[TreeType API](../../../developer/trees.md#the-treetype-api), plus some
additional functionality specific to R trees.

The R tree is generally less efficient for machine learning tasks than other
trees such as the [`KDTree`](kdtree.md) or [`Octree`](octree.md), but those
trees do not support dynamic insertion or deletion of points.  If insert/delete
functionality is required, then the R tree or other variants of
[`RectangleTree`](rectangle_tree.md) should be chosen instead.

 * [Template parameters](#template-parameters)
 * [Constructors](#constructors)
 * [Basic tree properties](#basic-tree-properties)
 * [Bounding distances with the tree](#bounding-distances-with-the-tree)
 * [Tree traversals](#tree-traversals)
 * [Example usage](#example-usage)

## See also

<!-- TODO: add links to all distance-based algorithms and other trees? -->

 * [`RectangleTree`](rectangle_tree.md)
 * [R-Tree on Wikipedia](https://en.wikipedia.org/wiki/R-tree)
 * [R-Trees: A Dynamic Index Structure for Spatial Searching (pdf)](http://www-db.deis.unibo.it/courses/SI-LS/papers/Gut84.pdf)
 * [Tree-Independent Dual-Tree Algorithms (pdf)](https://www.ratml.org/pub/pdf/2013tree.pdf)

## Template parameters

In accordance with the [TreeType
API](../../../developer/trees.md#template-parameters-required-by-the-treetype-policy)
(see also [this more detailed section](../../../developer/trees.md#template-parameters)),
the `RTree` class takes three template parameters:

```
RTree<DistanceType, StatisticType, MatType>
```

 * `DistanceType`: the [distance metric](../distances.md) to use for distance
   computations.  `RTree` requires that this is
   [`EuclideanDistance`](../distances.md#lmetric), and a compilation error will
   be thrown if any other `DistanceType` is specified.

 * `StatisticType`: this holds auxiliary information in each tree node.  By
   default, [`EmptyStatistic`](rectangle_tree.md#emptystatistic) is used, which
   holds no information.
   - See the [`StatisticType`](rectangle_tree.md#statistictype) section for more
     details.

 * `MatType`: the type of matrix used to represent points.  Must be a type
   matching the [Armadillo API](../../matrices.md).  By default, `arma::mat` is
   used, but other types such as `arma::fmat` or similar will work just fine.

The `RTree` class itself is a convenience typedef of the generic
[`RectangleTree`](rectangle_tree.md) class, using the
[`RTreeSplit`](rectangle_tree.md#rtreesplit) class as the split strategy, the
[`RTreeDescentHeuristic`](rectangle_tree.md#rtreedescentheuristic) class as the
descent strategy, and
[`NoAuxiliaryInformation`](rectangle_tree.md#auxiliaryinformationtype) as the
auxiliary information type.

If no template parameters are explicitly specified, then defaults are used:

```
RTree<> = RTree<EuclideanDistance, EmptyStatistic, arma::mat>
```

## Constructors

`RTree`s are constructed by inserting points in a dataset sequentially.
The dataset is not permuted during the construction process.

---

 * `node = RTree(data)`
 * `node = RTree(data, maxLeafSize=20, minLeafSize=8)`
 * `node = RTree(data, maxLeafSize=20, minLeafSize=8, maxNumChildren=5, minNumChildren=2)`
   - Construct an `RTree` on the given `data` with the given construction
     parameters.
   - By default, `data` is copied.  Avoid a copy by using `std::move()` (e.g.
     `std::move(data)`); when doing this, `data` will be set to an empty matrix.

---

 * `node = RTree<DistanceType, StatisticType, MatType>(data)`
 * `node = RTree<DistanceType, StatisticType, MatType>(data, maxLeafSize=20, minLeafSize=8)`
 * `node = RTree<DistanceType, StatisticType, MatType>(data, maxLeafSize=20, minLeafSize=8, maxNumChildren=5, minNumChildren=2)`
   - Construct an `RTree` on the given `data`, using custom template parameters
     to control the behavior of the tree and the given construction parameters.
   - By default, `data` is copied.  Avoid a copy by using `std::move()` (e.g.
     `std::move(data)`); when doing this, `data` will be set to an empty matrix.

---

 * `node = RTree(dimensionality)`
   - Construct an empty `RTree` with no children, no points, and
     default template parameters.
   - Use `node.Insert()` to insert points into the tree.  All points must have
     dimensionality `dimensionality`.

---

 * `node.Insert(x)`
   - Insert the point `x` into the tree.
   - `x` should have vector type compatible with the chosen `MatType`; so, for
     default `MatType`, `arma::vec` is the expected type.
   - If a custom `MatType` is specified (e.g. `arma::fmat`), then `x` should
     have type equivalent to the corresponding column vector type (e.g.
     `arma::fvec`).
   - Due to tree rebalancing, this may change the internal structure of the
     tree; so references and pointers to children of `node` may become invalid.
   - ***Warning:*** This will throw an exception if `node` is not the root of
     the tree!

 * `node.Delete(i)
   - Delete the point with index `i` from the tree.
   - The point to be deleted from the tree will be `node.Dataset().col(i)`;
     after deleting, the column will be removed from `node.Dataset()` and all
     indexes held in all tree nodes will be updated.  (Thus, this operation can
     be expensive!)
   - Due to tree rebalancing, this may change the internal structure of the
     tree; so references and pointers to children of `node` may become invalid.
   - ***Warning:*** This will throw an exception if `node` is not the root of
     the tree!

---

***Notes:***

 - The name `node` is used here for `RTree` objects instead of `tree`, because
   each `RTree` object is a single node in the tree.  The constructor returns
   the node that is the root of the tree.

 - See also the
   [developer documentation on tree constructors](../../../developer/trees.md#constructors-and-destructors).

---

### Constructor parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`MatType`](../../matrices.md) | [Column-major](../../matrices.md#representing-data-in-mlpack) matrix to build the tree on.  Pass with `std::move(data)` to avoid copying the matrix. | _(N/A)_ |
| `maxLeafSize` | `size_t` | Maximum number of points to store in each leaf. | `20` |
| `minLeafSize` | `size_t` | Minimum number of points to store in each leaf. | `8` |
| `maxNumChildren` | `size_t` | Maximum number of children allowed in each non-leaf node. | `5` |
| `minNumChildren` | `size_t` | Minimum number of children in each non-leaf node. | `2` |
| `dimensionality` | `size_t` | Dimensionality of points to be held in the tree. | _(N/A)_ |
| | | |
| `x` | [`arma::vec`](../../matrices.md) | Column vector: point to insert into tree.  Should have type matching the column vector type associated with `MatType`, and must have `node.Dataset().n_rows` elements. | _(N/A)_ |
| `i` | `size_t` | Index of point in `node.Dataset()` to delete from `node`. | _(N/A)_ |

## Basic tree properties

Once an `RTree` object is constructed, various properties of the tree can be
accessed or inspected.  Many of these functions are required by the [TreeType
API](../../../developer/trees.md#the-treetype-api).

### Navigating the tree

 * `node.NumChildren()` returns the number of children in `node`.  This is `0`
   if `node` is a leaf, and between the values of `node.MinNumChildren()` and
   `node.MaxNumChildren()` (inclusive) otherwise.

 * `node.IsLeaf()` returns a `bool` indicating whether or not `node` is a leaf.

 * `node.Child(i)` returns an `RTree&` that is the `i`th child.
   - `i` must be less than `node.NumChildren()`.
   - This function should only be called if `node.NumChildren()` is not `0`
     (e.g. if `node` is not a leaf).  Note that this returns a valid `RTree&`
     that can itself be used just like the root node of the tree!

 * `node.Parent()` will return an `RTree*` that points to the parent of `node`,
   or `NULL` if `node` is the root of the `RectangleTree`.

---

### Accessing members of a tree

 * `node.Bound()` will return an
   [`HRectBound<DistanceType, ElemType>&`](binary_space_tree.md#hrectbound)
   object that represents the hyperrectangle bounding box of `node`.
   - `ElemType` is the element type of `MatType`; so, if default template
     parameters are used, `ElemType` is `double`.
   - `bound` is a hyperrectangle that encloses all the descendant points of
     `node`.  It may be somewhat loose (e.g. points may not be very near the
     edges).

 * `node.Stat()` will return a `StatisticType&` holding the statistics of the
   node that were computed during tree construction.

 * `node.Distance()` will return a `EuclideanDistance&`.  Since
   `EuclideanDistance` has no members, this function is not likely to be useful,
   but it is required by the TreeType API.

 * `node.MinNumChildren()` returns the minimum number of children that the
   node is required to have as a `size_t`.  If points are deleted such that the
   number of children falls below this limit, then `node` will become a leaf and
   the tree will be rebalanced.

 * `node.MaxNumChildren()` returns the maximum number of children that the
   node is required to have as a `size_t`.  If points are inserted such that the
   number of children goes above this limit, new nodes will be added and the
   tree will be rebalanced.

 * `node.MaxLeafSize()` returns the maximum number of points that the node is
   allowed to hold as a `size_t`.  If the number of points held by `node`
   exceeds this limit during insertion, then `node` will be split and the tree
   will be rebalanced.

 * `node.MinLeafSize()` returns the minimum number of points that the node is
   allowed to hold as a `size_t`.  If the number of points held by `node` goes
   under this limit during deletion, then `node` will be deleted (if possible)
   and the tree will be rebalanced.

See also the
[developer documentation](../../../developer/trees.md#basic-tree-functionality)
for basic tree functionality in mlpack.

---

### Accessing data held in a tree

 * `node.Dataset()` will return a `const MatType&` that is an internally-held
   representation of the dataset the tree was built on.

 * `node.NumPoints()` returns a `size_t` indicating the number of points held
   directly in `node`.
   - If `node` is not a leaf, this will return `0`, as `RTree` only holds points
     directly in its leaves.
   - If `node` is a leaf, then this will return values between
     `node.MinLeafSize()` and `node.MaxLeafSize()` (inclusive).
   - If the tree has fewer than `node.MinLeafSize()` points total, then
     `node.NumPoints()` will return a value less than `node.MinLeafSize()`.

 * `node.Point(i)` returns a `size_t` indicating the index of the `i`'th point
   in `node.Dataset()`.
   - `i` must be in the range `[0, node.NumPoints() - 1]` (inclusive).
   - `node` must be a leaf (as non-leaves do not hold any points).
   - The `i`'th point in `node` can then be accessed as
     `node.Dataset().col(node.Point(i))`.
   - Accessing the actual `i`'th point itself can be done with, e.g.,
     `node.Dataset().col(node.Point(i))`.
   - Point indices are not necessarily contiguous for `RTree`s; that is,
     `node.Point(i) + 1` is not necessarily `node.Point(i + 1)`.

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
   - Descendant point indices are not necessarily contiguous for
     `RTree`s; that is, `node.Descendant(i) + 1` is not necessarily
     `node.Descendant(i + 1)`.

---

### Accessing computed bound quantities of a tree

The following quantities are cached for each node in a `RTree`, and so accessing
them does not require any computation.  In the documentation below, `ElemType`
is the element type of the given `MatType`; e.g., if `MatType` is `arma::mat`,
then `ElemType` is `double`.

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

 * `node.Center(center)` computes the center of the hyperrectangle bounding box
   of `node` and stores it in `center`.
   - `center` should be of type `arma::Col<ElemType>&`, where `ElemType` is the
     element type of the specified `MatType`.
   - `center` will be set to have size equivalent to the dimensionality of the
     dataset held by `node`.
   - This is equivalent to calling `node.Bound().Center(center)`.

 * An `RTree` can be serialized with
   [`data::Save()` and `data::Load()`](../../load_save.md#mlpack-objects).

## Bounding distances with the tree

The primary use of trees in mlpack is bounding distances to points or other tree
nodes.  The following functions can be used for these tasks.

 * `node.GetNearestChild(point)`
 * `node.GetFurthestChild(point)`
   - Return a `size_t` indicating the index of the child that is closest to (or
     furthest from) `point`, with respect to the `MinDistance()` (or
     `MaxDistance()`) function.
   - If there is a tie, the node with the lowest index is returned.
   - If `node` is a leaf, `0` is returned.
   - `point` should be a column vector type of the same type as `MatType`.
     (e.g., if `MatType` is `arma::mat`, then `point` should be an `arma::vec`.)

 * `node.GetNearestChild(other)`
 * `node.GetFurthestChild(other)`
   - Return a `size_t` indicating the index of the child that is closest to (or
     furthest from) the `RTree` node `other`, with respect to the
     `MinDistance()` (or `MaxDistance()`) function.
   - If there is a tie, the node with the lowest index is returned.
   - If `node` is a leaf, `0` is returned.

---

 * `node.MinDistance(point)`
 * `node.MinDistance(other)`
   - Return a `double` indicating the minimum possible distance between `node`
     and `point`, or the `RTree` node `other`.
   - This is equivalent to the minimum possible distance between any point
     contained in the bounding hyperrectangle of `node` and `point`, or between
     any point contained in the bounding hyperrectangle of `node` and any point
     contained in the bounding hyperrectangle of `other`.
   - `point` should be a column vector type of the same type as `MatType`.
     (e.g., if `MatType` is `arma::mat`, then `point` should be an `arma::vec`.)

 * `node.MaxDistance(point)`
 * `node.MaxDistance(other)`
   - Return a `double` indicating the maximum possible distance between `node`
     and `point`, or the `RTree` node `other`.
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

Like every mlpack tree, the `RTree` class provides a [single-tree and
dual-tree traversal](../../../developer/trees.md#traversals) that can be paired
with a [`RuleType` class](../../../developer/trees.md#rules) to implement a
single-tree or dual-tree algorithm.

 * `RTree::SingleTreeTraverser`
   - Implements a depth-first single-tree traverser.

 * `RTree::DualTreeTraverser`
   - Implements a dual-depth-first dual-tree traverser.

## Example usage

Build an `RTree` on the `cloud` dataset and print basic statistics about the
tree.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset, true);

// Build the rectangle tree with a leaf size of 10.  (This means that leaf nodes
// cannot contain more than 10 points.)
//
// The std::move() means that `dataset` will be empty after this call, and no
// data will be copied during tree building.
//
// Note that the '<>' is not necessary if C++20 is being used (e.g.
// `mlpack::RTree tree(...)` will work fine in C++20 or newer).
mlpack::RTree<> tree(std::move(dataset));

// Print the bounding box of the root node.
std::cout << "Bounding box of root node:" << std::endl;
for (size_t i = 0; i < tree.Bound().Dim(); ++i)
{
  std::cout << " - Dimension " << i << ": [" << tree.Bound()[i].Lo() << ", "
      << tree.Bound()[i].Hi() << "]." << std::endl;
}
std::cout << std::endl;

// Print the number of children in the root, and the allowable range.
std::cout << "Number of children of root: " << tree.NumChildren()
    << "; allowable range: [" << tree.MinNumChildren() << ", "
    << tree.MaxNumChildren() << "]." << std::endl;

// Print the number of descendant points of the root, and of each of its
// children.
std::cout << "Descendant points of root:        "
    << tree.NumDescendants() << "." << std::endl;
for (size_t i = 0; i < tree.NumChildren(); ++i)
{
  std::cout << "Descendant points of child " << i << ":  "
      << tree.Child(i).NumDescendants() << "." << std::endl;
}
std::cout << std::endl;

// Compute the center of the RTree.
arma::vec center;
tree.Center(center);
std::cout << "Center of tree: " << center.t();
```

---

Build two `RTree`s on subsets of the corel dataset and compute minimum and
maximum distances between different nodes in the tree.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::mat dataset;
mlpack::data::Load("corel-histogram.csv", dataset, true);

// Build trees on the first half and the second half of points.
mlpack::RTree<> tree1(dataset.cols(0, dataset.n_cols / 2));
mlpack::RTree<> tree2(dataset.cols(dataset.n_cols / 2 + 1, dataset.n_cols - 1));

// Compute the maximum distance between the trees.
std::cout << "Maximum distance between tree root nodes: "
    << tree1.MaxDistance(tree2) << "." << std::endl;

// Get the leftmost grandchild of the first tree's root---if it exists.
if (!tree1.IsLeaf() && !tree1.Child(0).IsLeaf())
{
  mlpack::RTree<>& node1 = tree1.Child(0).Child(0);

  // Get the leftmost grandchild of the second tree's root---if it exists.
  if (!tree2.IsLeaf() && !tree2.Child(0).IsLeaf())
  {
    mlpack::RTree<>& node2 = tree2.Child(0).Child(0);

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
    std::cout << "Child " << closestIndex << " is closest to node1."
        << std::endl;

    // And which child of node1 is further from node2?
    const size_t furthestIndex = node1.GetFurthestChild(node2);
    std::cout << "Child " << furthestIndex << " is furthest from node2."
        << std::endl;
  }
}
```

---

Build an `RTree` on 32-bit floating point data and save it to disk.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::fmat dataset;
mlpack::data::Load("corel-histogram.csv", dataset);

// Build the RTree using 32-bit floating point data as the matrix type.  We will
// still use the default EmptyStatistic and EuclideanDistance parameters.  A
// leaf size of 100 is used here.
mlpack::RTree<mlpack::EuclideanDistance,
              mlpack::EmptyStatistic,
              arma::fmat> tree(std::move(dataset), 100);

// Save the tree to disk with the name 'tree'.
mlpack::data::Save("tree.bin", "tree", tree);

std::cout << "Saved tree with " << tree.Dataset().n_cols << " points to "
    << "'tree.bin'." << std::endl;
```

---

Load a 32-bit floating point `RTree` from disk, then traverse it manually and
find the number of leaf nodes with less than 10 points.

```c++
// This assumes the tree has already been saved to 'tree.bin' (as in the example
// above).

// This convenient typedef saves us a long type name!
using TreeType = mlpack::RTree<mlpack::EuclideanDistance,
                               mlpack::EmptyStatistic,
                               arma::fmat>;

TreeType tree;
mlpack::data::Load("tree.bin", "tree", tree);
std::cout << "Tree loaded with " << tree.NumDescendants() << " points."
    << std::endl;

// Recurse in a depth-first manner.  Count both the total number of leaves, and
// the number of leaves with less than 10 points.
size_t leafCount = 0;
size_t totalLeafCount = 0;
std::stack<TreeType*> stack;
stack.push(&tree);
while (!stack.empty())
{
  TreeType* node = stack.top();
  stack.pop();

  if (node->NumPoints() < 10)
    ++leafCount;
  ++totalLeafCount;

  for (size_t i = 0; i < node->NumChildren(); ++i)
    stack.push(&node->Child(i));
}

// Note that it would be possible to use TreeType::SingleTreeTraverser to
// perform the recursion above, but that is more well-suited for more complex
// tasks that require pruning and other non-trivial behavior; so using a simple
// stack is the better option here.

// Print the results.
std::cout << leafCount << " out of " << totalLeafCount << " leaves have fewer "
    << "than 10 points." << std::endl;
```

---

Build an `RTree` by iteratively inserting points from the corel dataset, print
some information, and then remove a few randomly chosen points.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::mat dataset;
mlpack::data::Load("corel-histogram.csv", dataset, true);

// Create an empty tree of the right dimensionality.
mlpack::RTree<> t(dataset.n_rows);

// Insert points one by one for the first half of the dataset.
for (size_t i = 0; i < dataset.n_cols / 2; ++i)
  t.Insert(dataset.col(i));

std::cout << "After inserting half the points, the root node has "
    << t.NumDescendants() << " descendant points and "
    << t.NumChildren() << " child nodes." << std::endl;

// For the second half, insert the points backwards.
for (size_t i = dataset.n_cols - 1; i >= dataset.n_cols / 2; --i)
  t.Insert(dataset.col(i));

std::cout << "After inserting all the points, the root node has "
    << t.NumDescendants() << " descendant points and "
    << t.NumChildren() << " child nodes." << std::endl;

// Remove three random points.
t.Delete(mlpack::math::RandInt(0, t.NumDescendants()));
std::cout << "After removing 1 point, the root node has " << t.NumDescendants()
    << " descendant points." << std::endl;
t.Delete(mlpack::math::RandInt(0, t.NumDescendants()));
std::cout << "After removing 2 points, the root node has " << t.NumDescendants()
    << " descendant points." << std::endl;
t.Delete(mlpack::math::RandInt(0, t.NumDescendants()));
std::cout << "After removing 3 points, the root node has " << t.NumDescendants()
    << " descendant points." << std::endl;
```
