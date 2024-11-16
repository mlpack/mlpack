# `MeanSplitKDTree`

<!-- TODO: link to knn.md once it's done -->

The `MeanSplitKDTree` class represents a `k`-dimensional binary space
partitioning tree, and is a well-known data structure for efficient distance
operations (such as nearest neighbor search) in low dimensions---typically less
than 100.  This is very similar to the [`KDTree`](kdtree.md) class, except that
a different splitting strategy is used to split nodes in the tree.

In general, a `MeanSplitKDTree` will be a better balanced tree and have fewer
nodes than a `KDTree`.  However, counterintuitively, a more balanced tree can be
*worse* for search tasks like nearest neighbor search, because unbalanced nodes
are more easily pruned away during search.  In general, using a `KDTree` for
nearest neighbor search is 20-80% faster, *but this is not true for every
dataset or task*.

mlpack's `MeanSplitKDTree` implementation supports three template parameters for
configurable behavior, and implements all the functionality required by the
[TreeType API](../../../developer/trees.md#the-treetype-api), plus some
additional functionality specific to kd-trees.

 * [Template parameters](#template-parameters)
 * [Constructors](#constructors)
 * [Basic tree properties](#basic-tree-properties)
 * [Bounding distances with the tree](#bounding-distances-with-the-tree)
 * [Tree traversals](#tree-traversals)
 * [Example usage](#example-usage)

## See also

<!-- TODO: add links to all distance-based algorithms and other trees? -->

 * [kd-tree on Wikipedia](https://en.wikipedia.org/wiki/Kd-tree)
 * [`BinarySpaceTree`](binary_space_tree.md)
 * [`MeanSplit`](binary_space_tree.md#meansplit)
 * [Binary space partitioning on Wikipedia](https://dl.acm.org/doi/pdf/10.1145/361002.361007)
 * [original kd-tree paper (pdf)](https://dl.acm.org/doi/pdf/10.1145/361002.361007)
 * [Tree-Independent Dual-Tree Algorithms (pdf)](https://www.ratml.org/pub/pdf/2013tree.pdf)

## Template parameters

In accordance with the [TreeType
API](../../../developer/trees.md#template-parameters-required-by-the-treetype-policy)
(see also [this more detailed section](../../../developer/trees.md#template-parameters)),
the `MeanSplitKDTree` class takes three template parameters:

```
MeanSplitKDTree<DistanceType, StatisticType, MatType>
```

 * `DistanceType`: the [distance metric](../distances.md) to use for distance
   computations.  For the `MeanSplitKDTree`, this must be an
   [`LMetric`](../distances.md#lmetric).  By default, this is
   [`EuclideanDistance`](../distances.md#lmetric).
 * [`StatisticType`](binary_space_tree.md#statistictype): this holds auxiliary
   information in each tree node.  By default,
   [`EmptyStatistic`](binary_space_tree.md#emptystatistic) is used, which holds
   no information.
 * `MatType`: the type of matrix used to represent points.  Must be a type
   matching the [Armadillo API](../../matrices.md).  By default, `arma::mat` is
   used, but other types such as `arma::fmat` or similar will work just fine.

The `MeanSplitKDTree` class itself is a convenience typedef of the generic
[`BinarySpaceTree`](binary_space_tree.md) class, using the
[`HRectBound`](binary_space_tree.md#hrectbound) class as the bounding structure,
and using the [`MeanSplit`](binary_space_tree.md#meansplit) splitting strategy
for construction, which splits a node in the dimension of maximum variance on
the midpoint of the bound's range in that dimension.

If no template parameters are explicitly specified, then defaults are used:

```
MeanSplitKDTree<> = MeanSplitKDTree<EuclideanDistance, EmptyStatistic, arma::mat>
```

## Constructors

`MeanSplitKDTree`s are efficiently constructed by permuting points in a dataset
in a quicksort-like algorithm.  However, this means that the ordering of points
in the tree's dataset (accessed with `node.Dataset()`) after construction may be
different.

---

 * `node = MeanSplitKDTree(data, maxLeafSize=20)`
 * `node = MeanSplitKDTree(data, oldFromNew, maxLeafSize=20)`
 * `node = MeanSplitKDTree(data, oldFromNew, newFromOld, maxLeafSize=20)`
   - Construct a `MeanSplitKDTree` on the given `data`, using `maxLeafSize` as
     the maximum number of points held in a leaf.
   - By default, `data` is copied.  Avoid a copy by using `std::move()` (e.g.
     `std::move(data)`); when doing this, `data` will be set to an empty matrix.
   - Optionally, construct mappings from old points to new points.  `oldFromNew`
     and `newFromOld` will have length `data.n_cols`, and:
     * `oldFromNew[i]` indicates that point `i` in the tree's dataset was
       originally point `oldFromNew[i]` in `data`; that is,
       `node.Dataset().col(i)` is the point `data.col(oldFromNew[i])`.
     * `newFromOld[i]` indicates that point `i` in `data` is now point
       `newFromOld[i]` in the tree's dataset; that is,
       `node.Dataset().col(newFromOld[i])` is the point `data.col(i)`.

---

 * `node = MeanSplitKDTree<DistanceType, StatisticType, MatType>(data, maxLeafSize=20)`
 * `node = MeanSplitKDTree<DistanceType, StatisticType, MatType>(data, oldFromNew, maxLeafSize=20)`
 * `node = MeanSplitKDTree<DistanceType, StatisticType, MatType>(data, oldFromNew, newFromOld, maxLeafSize=20)`
   - Construct a `MeanSplitKDTree` on the given `data`, using custom template
     parameters to control the behavior of the tree, using `maxLeafSize` as the
     maximum number of points held in a leaf.
   - By default, `data` is copied.  Avoid a copy by using `std::move()` (e.g.
     `std::move(data)`); when doing this, `data` will be set to an empty matrix.
   - Optionally, construct mappings from old points to new points.  `oldFromNew`
     and `newFromOld` will have length `data.n_cols`, and:
     * `oldFromNew[i]` indicates that point `i` in the tree's dataset was
       originally point `oldFromNew[i]` in `data`; that is,
       `node.Dataset().col(i)` is the point `data.col(oldFromNew[i])`.
     * `newFromOld[i]` indicates that point `i` in `data` is now point
       `newFromOld[i]` in the tree's dataset; that is,
       `node.Dataset().col(newFromOld[i])` is the point `data.col(i)`.

---

 * `node = MeanSplitKDTree()`
   - Construct an empty mean-split kd-tree with no children and no points.

---

***Notes:***

 - The name `node` is used here for `MeanSplitKDTree` objects instead of `tree`,
   because each `MeanSplitKDTree` object is a single node in the tree.  The
   constructor returns the node that is the root of the tree.

 - Inserting individual points or removing individual points from a
   `MeanSplitKDTree` is not supported, because this generally results in a
   mean-split kd-tree with very loose bounding boxes.  It is better to simply
   build a new `MeanSplitKDTree` on the modified dataset.  For trees that
   support individual insertion and deletions, see the `RectangleTree` class and
   all its variants (e.g. `RTree`, `RStarTree`, etc.).

 - See also the
   [developer documentation on tree constructors](../../../developer/trees.md#constructors-and-destructors).

<!-- TODO: add links to RectangleTree above when it is documented -->

---

### Constructor parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../../matrices.md) | [Column-major](../../matrices.md#representing-data-in-mlpack) matrix to build the tree on.  Pass with `std::move(data)` to avoid copying the matrix. | _(N/A)_ |
| `maxLeafSize` | `size_t` | Maximum number of points to store in each leaf. | `20` |
| `oldFromNew` | `std::vector<size_t>` | Mappings from points in `node.Dataset()` to points in `data`. | _(N/A)_ |
| `newFromOld` | `std::vector<size_t>` | Mappings from points in `data` to points in `node.Dataset()`. | _(N/A)_ |

## Basic tree properties

Once a `MeanSplitKDTree` object is constructed, various properties of the tree
can be accessed or inspected.  Many of these functions are required by the
[TreeType API](../../../developer/trees.md#the-treetype-api).

### Navigating the tree

 * `node.NumChildren()` returns the number of children in `node`.  This is
   either `2` if `node` has children, or `0` if `node` is a leaf.

 * `node.IsLeaf()` returns a `bool` indicating whether or not `node` is a leaf.

 * `node.Child(i)` returns a `MeanSplitKDTree&` that is the `i`th child.
   - `i` must be `0` or `1`.
   - This function should only be called if `node.NumChildren()` is not `0`
     (e.g. if `node` is not a leaf).  Note that this returns a valid
     `MeanSplitKDTree&` that can itself be used just like the root node of the
     tree!
   - `node.Left()` and `node.Right()` are convenience functions specific to
     `MeanSplitKDTree` that will return `MeanSplitKDTree*` (pointers) to the
     left and right children, respectively, or `NULL` if `node` has no children.

 * `node.Parent()` will return a `MeanSplitKDTree*` that points to the parent of
   `node`, or `NULL` if `node` is the root of the `MeanSplitKDTree`.

---

### Accessing members of a tree

 * `node.Bound()` will return an
   [`HRectBound&`](binary_space_tree.md#hrectbound) object that represents the
   hyperrectangle bounding box of `node`.  This is the smallest hyperrectangle
   that encloses all the descendant points of `node`.

 * `node.Stat()` will return an `EmptyStatistic&` (or a `StatisticType&` if a
   [custom `StatisticType`](#template-parameters) was specified as a template
   parameter) holding the statistics of the node that were computed during tree
   construction.

 * `node.Distance()` will return a
   [`EuclideanDistance&`](../distances.md#lmetric) (or a `DistanceType&` if a
   [custom `DistanceType`](#template-parameters) was specified as a template
   parameter).
   - This function is required by the
     [TreeType API](../../../developer/trees.md#the-treetype-api), but given
     that `MeanSplitKDTree` requires an [`LMetric`](../distances.md#lmetric) to
     be used, and `LMetric` only has `static` functions and holds no state, this
     function is not likely to be useful.

See also the
[developer documentation](../../../developer/trees.md#basic-tree-functionality)
for basic tree functionality in mlpack.

---

### Accessing data held in a tree

 * `node.Dataset()` will return a `const arma::mat&` that is the dataset the
   tree was built on.  Note that this is a permuted version of the `data` matrix
   passed to the constructor.
   - If a [custom `MatType`](#template-parameters) is being used, the return
     type will be `const MatType&` instead of `const arma::mat&`.

 * `node.NumPoints()` returns a `size_t` indicating the number of points held
   directly in `node`.
   - If `node` is not a leaf, this will return `0`, as `MeanSplitKDTree` only
     holds points directly in its leaves.
   - If `node` is a leaf, then the number of points will be less than or equal
     to the `maxLeafSize` that was specified when the tree was constructed.

 * `node.Point(i)` returns a `size_t` indicating the index of the `i`'th point
   in `node.Dataset()`.
   - `i` must be in the range `[0, node.NumPoints() - 1]` (inclusive).
   - `node` must be a leaf (as non-leaves do not hold any points).
   - The `i`'th point in `node` can then be accessed as
     `node.Dataset().col(node.Point(i))`.
   - In a `MeanSplitKDTree`, because of the permutation of points done [during
     construction](#constructors), point indices are contiguous:
     `node.Point(i + j)` is the same as `node.Point(i) + j` for valid `i` and
     `j`.
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
   - In a `MeanSplitKDTree`, because of the permutation of points done [during
     construction](#constructors), point indices are contiguous:
     `node.Descendant(i + j)` is the same as `node.Descendant(i) + j` for valid
     `i` and `j`.
   - Accessing the actual `i`'th descendant itself can be done with, e.g.,
     `node.Dataset().col(node.Descendant(i))`.

 * `node.Begin()` returns a `size_t` indicating the index of the first
   descendant point of `node`.
   - This is equivalent to `node.Descendant(0)`.

 * `node.Count()` returns a `size_t` indicating the number of descendant points
   of `node`.
   - This is equivalent to `node.NumDescendants()`.

---

### Accessing computed bound quantities of a tree

The following quantities are cached for each node in a `MeanSplitKDTree`, and so
accessing them does not require any computation.

 * `node.FurthestPointDistance()` returns a `double` representing the distance
   between the center of the bounding hyperrectangle of `node` and the furthest
   point held by `node`.
   - If `node` is not a leaf, this returns 0 (because `node` does not hold any
     points).

 * `node.FurthestDescendantDistance()` returns a `double` representing the
   distance between the center of the bounding hyperrectangle of `node` and the
   furthest descendant point held by `node`.

 * `node.MinimumBoundDistance()` returns a `double` representing minimum
   possible distance from the center of the node to any edge of the
   hyperrectangle bound.
   - This quantity is half the width of the smallest dimension of
     `node.Bound()`.

 * `node.ParentDistance()` returns a `double` representing the distance between
   the center of the bounding hyperrectangle of `node` and the center of the
   bounding hyperrectangle of its parent.
   - If `node` is the root of the tree, `0` is returned.

***Notes:***

 - If a [custom `MatType`](#template-parameters) was specified when constructing
   the `MeanSplitKDTree`, then the return type of each method is the element
   type of the given `MatType` instead of `double`.  (e.g., if `MatType` is
   `arma::fmat`, then the return type is `float`.)

 - For more details on each bound quantity, see the
   [developer documentation](../../../developer/trees.md#complex-tree-functionality-and-bounds)
   on bound quantities for trees.

---

### Other functionality

 * `node.Center(center)` computes the center of the bounding hyperrectangle of
   `node` and stores it in `center`.
   - `center` should be of type `arma::vec&`.  (If a [custom
     `MatType`](#template-parameters) was specified when constructing the
     `MeanSplitKDTree`, the type is instead the column vector type for the given
     `MatType`; e.g., `arma::fvec&` when `MatType` is `arma::fmat`.)
   - `center` will be set to have size equivalent to the dimensionality of the
     dataset held by `node`.
   - This is equivalent to calling `node.Bound().Center(center)`.

 * A `MeanSplitKDTree` can be serialized with
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
   - `point` should be of type `arma::vec`.  (If a [custom
     `MatType`](#template-parameters) was specified when constructing the
     `MeanSplitKDTree`, the type is instead the column vector type for the given
     `MatType`; e.g., `arma::fvec` when `MatType` is `arma::fmat`.)

 * `node.GetNearestChild(other)`
 * `node.GetFurthestChild(other)`
   - Return a `size_t` indicating the index of the child (`0` for left, `1` for
     right) that is closest to (or furthest from) the `MeanSplitKDTree` node
    `other`, with respect to the `MinDistance()` (or `MaxDistance()`) function.
   - If there is a tie, `2` (an invalid index) is returned. ***Note that this
     behavior differs from the version above that takes a point.***
   - If `node` is a leaf, `0` is returned.

---

 * `node.MinDistance(point)`
 * `node.MinDistance(other)`
   - Return a `double` indicating the minimum possible distance between `node`
     and `point`, or the `MeanSplitKDTree` node `other`.
   - This is equivalent to the minimum possible distance between any point
     contained in the bounding hyperrectangle of `node` and `point`, or between
     any point contained in the bounding hyperrectangle of `node` and any point
     contained in the bounding hyperrectangle of `other`.
   - `point` should be of type `arma::vec`.  (If a [custom
     `MatType`](#template-parameters) was specified when constructing the
     `MeanSplitKDTree`, the type is instead the column vector type for the given
     `MatType`, and the return type is the element type of `MatType`; e.g.,
     `point` should be `arma::fvec` when `MatType` is `arma::fmat`, and the
     returned distance is `float`).

 * `node.MaxDistance(point)`
 * `node.MaxDistance(other)`
   - Return a `double` indicating the maximum possible distance between `node`
     and `point`, or the `MeanSplitKDTree` node `other`.
   - This is equivalent to the maximum possible distance between any point
     contained in the bounding hyperrectangle of `node` and `point`, or between
     any point contained in the bounding hyperrectangle of `node` and any point
     contained in the bounding hyperrectangle of `other`.
   - `point` should be of type `arma::vec`.  (If a [custom
     `MatType`](#template-parameters) was specified when constructing the
     `MeanSplitKDTree`, the type is instead the column vector type for the given
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
     the `MeanSplitKDTree`, the type is instead the column vector type for the
     given `MatType`, and the return type is a `RangeType` with element type the
     same as `MatType`; e.g., `point` should be `arma::fvec` when `MatType` is
     `arma::fmat`, and the returned type is
     [`RangeType<float>`](../math.md#range)).

### Tree traversals

Like every mlpack tree, the `MeanSplitKDTree` class provides a [single-tree and
dual-tree traversal](../../../developer/trees.md#traversals) that can be paired
with a [`RuleType` class](../../../developer/trees.md#rules) to implement a
single-tree or dual-tree algorithm.

 * `MeanSplitKDTree::SingleTreeTraverser`
   - Implements a depth-first single-tree traverser.

 * `MeanSplitKDTree::DualTreeTraverser`
   - Implements a dual-depth-first dual-tree traverser.

In addition to those two classes, which are required by the
[`TreeType` policy](../../../developer/trees.md), an additional traverser is
available:

 * `MeanSplitKDTree::BreadthFirstDualTreeTraverser`
   - Implements a dual-breadth-first dual-tree traverser.
   - ***Note:*** this traverser is not useful for all tasks; because the
     `MeanSplitKDTree` only holds points in the leaves, this means that no base
     cases (e.g. comparisons between points) will be called until *all* pairs of
     intermediate nodes have been scored!

## Example usage

Build a `MeanSplitKDTree` on the `cloud` dataset and print basic statistics
about the tree.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset, true);

// Build the kd-tree with a leaf size of 10.  (This means that nodes are split
// until they contain 10 or fewer points.)
//
// The std::move() means that `dataset` will be empty after this call, and no
// data will be copied during tree building.
//
// Note that the '<>' isn't necessary if C++20 is being used (e.g.
// `mlpack::MeanSplitKDTree tree(...)` will work fine in C++20 or newer).
mlpack::MeanSplitKDTree<> tree(std::move(dataset));

// Print the bounding box of the root node.
std::cout << "Bounding box of root node:" << std::endl;
for (size_t i = 0; i < tree.Bound().Dim(); ++i)
{
  std::cout << " - Dimension " << i << ": [" << tree.Bound()[i].Lo() << ", "
      << tree.Bound()[i].Hi() << "]." << std::endl;
}
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

// Compute the center of the kd-tree.
arma::vec center;
tree.Center(center);
std::cout << "Center of kd-tree: " << center.t();
```

---

Build two `MeanSplitKDTree`s on subsets of the corel dataset and compute minimum
and maximum distances between different nodes in the tree.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::mat dataset;
mlpack::data::Load("corel-histogram.csv", dataset, true);

// Build mean-split kd-trees on the first half and the second half of points.
mlpack::MeanSplitKDTree<> tree1(dataset.cols(0, dataset.n_cols / 2));
mlpack::MeanSplitKDTree<> tree2(dataset.cols(dataset.n_cols / 2 + 1,
    dataset.n_cols - 1));

// Compute the maximum distance between the trees.
std::cout << "Maximum distance between tree root nodes: "
    << tree1.MaxDistance(tree2) << "." << std::endl;

// Get the leftmost grandchild of the first tree's root---if it exists.
if (!tree1.IsLeaf() && !tree1.Child(0).IsLeaf())
{
  mlpack::MeanSplitKDTree<>& node1 = tree1.Child(0).Child(0);

  // Get the rightmost grandchild of the second tree's root---if it exists.
  if (!tree2.IsLeaf() && !tree2.Child(1).IsLeaf())
  {
    mlpack::MeanSplitKDTree<>& node2 = tree2.Child(1).Child(1);

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

Build a `MeanSplitKDTree` on 32-bit floating point data and save it to disk.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::fmat dataset;
mlpack::data::Load("corel-histogram.csv", dataset);

// Build the MeanSplitKDTree using 32-bit floating point data as the matrix
// type.  We will still use the default EmptyStatistic and EuclideanDistance
// parameters.  A leaf size of 100 is used here.
mlpack::MeanSplitKDTree<mlpack::EuclideanDistance,
                        mlpack::EmptyStatistic,
                        arma::fmat> tree(std::move(dataset), 100);

// Save the MeanSplitKDTree to disk with the name 'tree'.
mlpack::data::Save("tree.bin", "tree", tree);

std::cout << "Saved tree with " << tree.Dataset().n_cols << " points to "
    << "'tree.bin'." << std::endl;
```

---

Load a 32-bit floating point `MeanSplitKDTree` from disk, then traverse it
manually and find the number of leaf nodes with fewer than 10 points.

```c++
// This assumes the tree has already been saved to 'tree.bin' (as in the example
// above).

// This convenient typedef saves us a long type name!
using TreeType = mlpack::MeanSplitKDTree<mlpack::EuclideanDistance,
                                         mlpack::EmptyStatistic,
                                         arma::fmat>;

TreeType tree;
mlpack::data::Load("tree.bin", "tree", tree);
std::cout << "Tree loaded with " << tree.NumDescendants() << " points."
    << std::endl;

// Recurse in a depth-first manner.  Count both the total number of leaves, and
// the number of leaves with fewer than 10 points.
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

  if (!node->IsLeaf())
  {
    stack.push(node->Left());
    stack.push(node->Right());
  }
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

Build a `MeanSplitKDTree` and map between original points and new points.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset, true);

// Build the tree.
std::vector<size_t> oldFromNew, newFromOld;
mlpack::MeanSplitKDTree<> tree(dataset, oldFromNew, newFromOld);

// oldFromNew and newFromOld will be set to the same size as the dataset.
std::cout << "Number of points in dataset: " << dataset.n_cols << "."
    << std::endl;
std::cout << "Size of oldFromNew: " << oldFromNew.size() << "." << std::endl;
std::cout << "Size of newFromOld: " << newFromOld.size() << "." << std::endl;
std::cout << std::endl;

// See where point 42 in the tree's dataset came from.
std::cout << "Point 42 in the permuted tree's dataset:" << std::endl;
std::cout << "  " << tree.Dataset().col(42).t();
std::cout << "Was originally point " << oldFromNew[42] << ":" << std::endl;
std::cout << "  " << dataset.col(oldFromNew[42]).t();
std::cout << std::endl;

// See where point 7 in the original dataset was mapped.
std::cout << "Point 7 in original dataset:" << std::endl;
std::cout << "  " << dataset.col(7).t();
std::cout << "Mapped to point " << newFromOld[7] << ":" << std::endl;
std::cout << "  " << tree.Dataset().col(newFromOld[7]).t();
```

---

Compare the `MeanSplitKDTree` to a `KDTree` on a dataset.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::mat dataset;
mlpack::data::Load("corel-histogram.csv", dataset, true);

// Build the trees.
mlpack::KDTree<> kdtree(dataset);
mlpack::MeanSplitKDTree<> mskdtree(dataset);

// Compute the number of nodes and leaves in each tree, the average volume of
// leaf nodes, and the average volume of non-leaf nodes.
double leafVolume = 0.0;
double nonleafVolume = 0.0;
size_t numNodes = 0;
size_t numLeaves = 0;

// We will compute the quantities using a stack to do a depth-first traversal of
// the tree.
std::stack<mlpack::KDTree<>*> kdStack;
kdStack.push(&kdtree);
while (!kdStack.empty())
{
  mlpack::KDTree<>* node = kdStack.top();
  kdStack.pop();

  ++numNodes;
  if (node->IsLeaf())
  {
    ++numLeaves;
    leafVolume += node->Bound().Volume();
  }
  else
  {
    nonleafVolume += node->Bound().Volume();
    kdStack.push(node->Left());
    kdStack.push(node->Right());
  }
}

// Print statistics about the KDTree.
std::cout << "KDTree statistics:" << std::endl;
std::cout << " - Number of nodes: " << numNodes << "." << std::endl;
std::cout << " - Number of leaves: " << numLeaves << "." << std::endl;
std::cout << " - Average leaf volume: " << (leafVolume / numLeaves) << "."
    << std::endl;
std::cout << " - Average non-leaf volume: "
    << (nonleafVolume / (numNodes - numLeaves)) << "." << std::endl;

// Now compute the same quantities for the MeanSplitKDTree.
leafVolume = 0.0;
nonleafVolume = 0.0;
numLeaves = 0;
numNodes = 0;
std::stack<mlpack::MeanSplitKDTree<>*> mskdStack;
mskdStack.push(&mskdtree);
while (!mskdStack.empty())
{
  mlpack::MeanSplitKDTree<>* node = mskdStack.top();
  mskdStack.pop();

  ++numNodes;
  if (node->IsLeaf())
  {
    ++numLeaves;
    leafVolume += node->Bound().Volume();
  }
  else
  {
    nonleafVolume += node->Bound().Volume();
    mskdStack.push(node->Left());
    mskdStack.push(node->Right());
  }
}

// Print statistics about the MeanSplitKDTree.
std::cout << "MeanSplitKDTree statistics:" << std::endl;
std::cout << " - Number of nodes: " << numNodes << "." << std::endl;
std::cout << " - Number of leaves: " << numLeaves << "." << std::endl;
std::cout << " - Average leaf volume: " << (leafVolume / numLeaves) << "."
    << std::endl;
std::cout << " - Average non-leaf volume: "
    << (nonleafVolume / (numNodes - numLeaves)) << "." << std::endl;
```
