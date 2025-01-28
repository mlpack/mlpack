# `Octree`

<!-- TODO: link to knn.md once it's done -->

The `Octree` class implements the generalized octree, a hierarchical tree
structure.  When built on data in two dimensions, it is also called a
'quadtree'.  The generalized octree is most useful on data in only two or three
dimensions, as its number of children is exponential in the data dimension:
so, e.g., an octree node in two dimensions has up to four children; in three
dimensions has up to eight; and so on.

mlpack's `Octree` implementation differs from many textbook descriptions of
quadtrees and octrees in that mlpack allows the bounding box for an `Octree`
node to be a general hyperrectangle instead of a hypercube (e.g. each dimension
may have a different width).  This allows the `Octree` to fit the data points
with tighter bounds for each node---very important for pruning in distance-based
tasks.

For higher-dimensional use cases (e.g. data dimension of 5 or more), the
[`KDTree`](kdtree.md), other [`BinarySpaceTree`](binary_space_tree.md) variants,
or virtually any other type of tree will perform better.

mlpack's `Octree` implementation supports three template parameters for
configurable behavior, and implements all the functionality required by the
[TreeType API](../../../developer/trees.md#the-treetype-api), plus some
additional functionality specific to octrees.

 * [Template parameters](#template-parameters)
 * [Constructors](#constructors)
 * [Basic tree properties](#basic-tree-properties)
 * [Bounding distances with the tree](#bounding-distances-with-the-tree)
 * [Tree traversals](#tree-traversals)
 * [Example usage](#example-usage)

## See also

<!-- TODO: add links to all distance-based algorithms and other trees? -->

 * [Octree on Wikipedia](https://en.wikipedia.org/wiki/Octree)
 * [Quadtree on Wikipedia](https://en.wikipedia.org/wiki/Quadtree)
 * [Binary space partitioning on Wikipedia](https://dl.acm.org/doi/pdf/10.1145/361002.361007)
 * [Tree-Independent Dual-Tree Algorithms (pdf)](https://www.ratml.org/pub/pdf/2013tree.pdf)

## Template parameters

In accordance with the [TreeType
API](../../../developer/trees.md#template-parameters-required-by-the-treetype-policy)
(see also [this more detailed section](../../../developer/trees.md#template-parameters)),
the `Octree` class takes three template parameters:

```
Octree<DistanceType, StatisticType, MatType>
```

 * `DistanceType`: the [distance metric](../distances.md) to use for distance
   computations.  For the `Octree`, this must be an
   [`LMetric`](../distances.md#lmetric).  By default, this is
   [`EuclideanDistance`](../distances.md#lmetric).
 * [`StatisticType`](binary_space_tree.md#statistictype): this holds auxiliary
   information in each tree node.  By default,
   [`EmptyStatistic`](binary_space_tree.md#emptystatistic) is used, which holds
   no information.
 * `MatType`: the type of matrix used to represent points.  Must be a type
   matching the [Armadillo API](../../matrices.md).  By default, `arma::mat` is
   used, but other types such as `arma::fmat` or similar will work just fine.

If no template parameters are explicitly specified, then defaults are used:

```
Octree<> = Octree<EuclideanDistance, EmptyStatistic, arma::mat>
```

## Constructors

`Octree`s are efficiently constructed by permuting points in a dataset in a
quicksort-like algorithm.  However, this means that the ordering of points in
the tree's dataset (accessed with `node.Dataset()`) after construction may be
different.

---

 * `node = Octree(data, maxLeafSize=20)`
 * `node = Octree(data, oldFromNew, maxLeafSize=20)`
 * `node = Octree(data, oldFromNew, newFromOld, maxLeafSize=20)`
   - Construct an `Octree` on the given `data`, using `maxLeafSize` as the
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

 * `node = Octree<DistanceType, StatisticType, MatType>(data, maxLeafSize=20)`
 * `node = Octree<DistanceType, StatisticType, MatType>(data, oldFromNew, maxLeafSize=20)`
 * `node = Octree<DistanceType, StatisticType, MatType>(data, oldFromNew, newFromOld, maxLeafSize=20)`
   - Construct an `Octree` on the given `data`, using custom template parameters
     to control the behavior of the tree, using `maxLeafSize` as the maximum
     number of points held in a leaf.
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

 * `node = Octree()`
   - Construct an empty octree with no children and no points.

---

***Notes:***

 - The name `node` is used here for `Octree` objects instead of `tree`, because
   each `Octree` object is a single node in the tree.  The constructor returns
   the node that is the root of the tree.

 - Inserting individual points or removing individual points from an `Octree` is
   not supported, because this generally results in a octree with very loose
   bounding boxes.  It is better to simply build a new `Octree` on the modified
   dataset.  For trees that support individual insertion and deletions, see the
   `RectangleTree` class and all its variants (e.g. `RTree`, `RStarTree`, etc.).

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

Once an `Octree` object is constructed, various properties of the tree can be
accessed or inspected.  Many of these functions are required by the [TreeType
API](../../../developer/trees.md#the-treetype-api).

### Navigating the tree

 * `node.NumChildren()` returns the number of children in `node`.  This is `0`
   if `node` is a leaf, and anywhere between `1` and
   `pow(2, node.Dataset().n_rows)` otherwise.

 * `node.IsLeaf()` returns a `bool` indicating whether or not `node` is a leaf.

 * `node.Child(i)` returns an `Octree&` that is the `i`th child.
   - `i` must be less than `node.NumChildren()`.
   - This function should only be called if `node.NumChildren()` is not `0`
     (e.g. if `node` is not a leaf).  Note that this returns a valid `Octree&`
     that can itself be used just like the root node of the tree!

 * `node.Parent()` will return an `Octree*` that points to the parent of `node`,
   or `NULL` if `node` is the root of the `Octree`.

---

### Accessing members of a tree

 * `node.Bound()` will return an
   [`HRectBound&`](binary_space_tree.md#hrectbound) object that represents the
   hyperrectangle bounding box of `node`.  This is the smallest hyperrectangle
   that encloses all the descendant points of `node`.
    - ***Note:*** mlpack's `Octree` implementation does not require that the
      bounding structure is a hypercube---each dimension of the bound is allowed
      to have a different width.

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
     that `Octree` requires an [`LMetric`](../distances.md#lmetric) to be used,
     and `LMetric` only has `static` functions and holds no state, this function
     is not likely to be useful.

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
   - If `node` is not a leaf, this will return `0`, as `Octree` only holds
     points directly in its leaves.
   - If `node` is a leaf, then the number of points will be less than or equal
     to the `maxLeafSize` that was specified when the tree was constructed.

 * `node.Point(i)` returns a `size_t` indicating the index of the `i`'th point
   in `node.Dataset()`.
   - `i` must be in the range `[0, node.NumPoints() - 1]` (inclusive).
   - `node` must be a leaf (as non-leaves do not hold any points).
   - The `i`'th point in `node` can then be accessed as
     `node.Dataset().col(node.Point(i))`.
   - In an `Octree`, because of the permutation of points done [during
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
   - In an `Octree`, because of the permutation of points done [during
     construction](#constructors), point indices are contiguous:
     `node.Descendant(i + j)` is the same as `node.Descendant(i) + j` for valid
     `i` and `j`.
   - Accessing the actual `i`'th descendant itself can be done with, e.g.,
     `node.Dataset().col(node.Descendant(i))`.

 * `node.Begin()` returns a `size_t` indicating the index of the first
   descendant point of `node`.
   - This is equivalent to `node.Descendant(0)`.

 * `node.Count()` returns a `size_t` indicating the number of descendant points    of `node`.
   - This is equivalent to `node.NumDescendants()`.

---

### Accessing computed bound quantities of a tree

The following quantities are cached for each node in an `Octree`, and so
accessing them does not require any computation.

 * `node.FurthestPointDistance()` returns a `double` representing the distance
   between the center of the bounding hyperrectangle of `node` and the furthest
   point held by `node`.
   - If `node` is not a leaf, this returns 0 (because `node` does not hold any
     points).

 * `node.FurthestDescendantDistance()` returns a `double` representing the
   distance between the center of the bounding hyperrectangle of `node` and the
   furthest descendant point held by `node`.

 * `node.MinimumBoundDistance()` returns a `double` representing the minimum
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
   the `Octree`, then the return type of each method is the element type of the
   given `MatType` instead of `double`.  (e.g., if `MatType` is `arma::fmat`,
   then the return type is `float`.)

 - For more details on each bound quantity, see the
   [developer documentation](../../../developer/trees.md#complex-tree-functionality-and-bounds)
   on bound quantities for trees.

---

### Other functionality

 * `node.Center(center)` computes the center of the bounding hyperrectangle of
   `node` and stores it in `center`.
   - `center` should be of type `arma::vec&`.  (If a [custom
     `MatType`](#template-parameters) was specified when constructing the
     `Octree`, the type is instead the column vector type for the given
     `MatType`; e.g., `arma::fvec&` when `MatType` is `arma::fmat`.)
   - `center` will be set to have size equivalent to the dimensionality of the
     dataset held by `node`.
   - This is equivalent to calling `node.Bound().Center(center)`.

 * An `Octree` can be serialized with
   [`data::Save()` and `data::Load()`](../../load_save.md#mlpack-objects).

## Bounding distances with the tree

The primary use of trees in mlpack is bounding distances to points or other tree
nodes.  The following functions can be used for these tasks.

 * `node.GetNearestChild(point)`
 * `node.GetFurthestChild(point)`
   - Return a `size_t` indicating the index of the child (`0` for left, `1` for
     right) that is closest to (or furthest from) `point`, with respect
     to the `MinDistance()` (or `MaxDistance()`) function.
   - If there is a tie, the child with the lowest index is returned.
   - If `node` is a leaf, `0` is returned.
   - `point` should be of type `arma::vec`.  (If a [custom
     `MatType`](#template-parameters) was specified when constructing the
     `Octree`, the type is instead the column vector type for the given
     `MatType`; e.g., `arma::fvec` when `MatType` is `arma::fmat`.)

 * `node.GetNearestChild(other)`
 * `node.GetFurthestChild(other)`
   - Return a `size_t` indicating the index of the child (`0` for left, `1` for
     right) that is closest to (or furthest from) the `Octree` node `other`,
     with respect to the `MinDistance()` (or `MaxDistance()`) function.
   - If there is a tie, the child with the lowest index is returned.
   - If `node` is a leaf, `0` is returned.

---

 * `node.MinDistance(point)`
 * `node.MinDistance(other)`
   - Return a `double` indicating the minimum possible distance between `node`
     and `point`, or the `Octree` node `other`.
   - This is equivalent to the minimum possible distance between any point
     contained in the bounding hyperrectangle of `node` and `point`, or between
     any point contained in the bounding hyperrectangle of `node` and any point
     contained in the bounding hyperrectangle of `other`.
   - `point` should be of type `arma::vec`.  (If a [custom
     `MatType`](#template-parameters) was specified when constructing the
     `Octree`, the type is instead the column vector type for the given
     `MatType`, and the return type is the element type of `MatType`; e.g.,
     `point` should be `arma::fvec` when `MatType` is `arma::fmat`, and the
     returned distance is `float`).

 * `node.MaxDistance(point)`
 * `node.MaxDistance(other)`
   - Return a `double` indicating the maximum possible distance between `node`
     and `point`, or the `Octree` node `other`.
   - This is equivalent to the maximum possible distance between any point
     contained in the bounding hyperrectangle of `node` and `point`, or between
     any point contained in the bounding hyperrectangle of `node` and any point
     contained in the bounding hyperrectangle of `other`.
   - `point` should be of type `arma::vec`.  (If a [custom
     `MatType`](#template-parameters) was specified when constructing the
     `Octree`, the type is instead the column vector type for the given
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
     the `Octree`, the type is instead the column vector type for the given
     `MatType`, and the return type is a `RangeType` with element type the same
     as `MatType`; e.g., `point` should be `arma::fvec` when `MatType` is
     `arma::fmat`, and the returned type is
     [`RangeType<float>`](../math.md#range)).

### Tree traversals

Like every mlpack tree, the `Octree` class provides a [single-tree and dual-tree
traversal](../../../developer/trees.md#traversals) that can be paired with a
[`RuleType` class](../../../developer/trees.md#rules) to implement a single-tree
or dual-tree algorithm.

 * `Octree::SingleTreeTraverser`
   - Implements a depth-first single-tree traverser.

 * `Octree::DualTreeTraverser`
   - Implements a dual-depth-first dual-tree traverser.

## Example usage

Build an `Octree` on the `iris` dataset and print basic statistics about the
tree.

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::mat dataset;
mlpack::data::Load("iris.csv", dataset, true);

// Build the octree with a leaf size of 10.  (This means that nodes are split
// until they contain 10 or fewer points.)
//
// The std::move() means that `dataset` will be empty after this call, and no
// data will be copied during tree building.
//
// Note that the '<>' isn't necessary if C++20 is being used (e.g.
// `mlpack::Octree tree(...)` will work fine in C++20 or newer).
mlpack::Octree<> tree(std::move(dataset));

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
for (size_t i = 0; i < tree.NumChildren(); ++i)
{
  std::cout << "Descendant points of child " << i << ": "
      << tree.Child(i).NumDescendants() << "." << std::endl;
}
std::cout << std::endl;

// Compute the center of the octree.
arma::vec center;
tree.Center(center);
std::cout << "Center of octree: " << center.t();
```

---

Build two `Octree`s on subsets of the tiny LCDM dataset (3-dimensional
astronomical data) and compute minimum and maximum distances between different
nodes in the tree.

```c++
// See https://datasets.mlpack.org/lcdm_tiny.csv.
arma::mat dataset;
mlpack::data::Load("lcdm_tiny.csv", dataset, true);

// Build octrees on the first half and the second half of points.
mlpack::Octree<> tree1(dataset.cols(0, dataset.n_cols / 2));
mlpack::Octree<> tree2(dataset.cols(dataset.n_cols / 2 + 1,
    dataset.n_cols - 1));

// Compute the maximum distance between the trees.
std::cout << "Maximum distance between tree root nodes: "
    << tree1.MaxDistance(tree2) << "." << std::endl;

// Get the leftmost grandchild of the first tree's root---if it exists.
if (!tree1.IsLeaf() && !tree1.Child(0).IsLeaf())
{
  mlpack::Octree<>& node1 = tree1.Child(0).Child(0);

  // Get the rightmost grandchild of the second tree's root---if it exists.
  if (!tree2.IsLeaf() && !tree2.Child(0).IsLeaf())
  {
    mlpack::Octree<>& node2 = tree2.Child(0).Child(0);

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

Build an `Octree` on 32-bit floating point data and save it to disk.

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::fmat dataset;
mlpack::data::Load("iris.csv", dataset);

// Build the Octree using 32-bit floating point data as the matrix type.
// We will still use the default EmptyStatistic and EuclideanDistance
// parameters.  A leaf size of 100 is used here.
mlpack::Octree<mlpack::EuclideanDistance,
               mlpack::EmptyStatistic,
               arma::fmat> tree(std::move(dataset), 100);

// Save the Octree to disk with the name 'tree'.
mlpack::data::Save("tree.bin", "tree", tree);

std::cout << "Saved tree with " << tree.Dataset().n_cols << " points to "
    << "'tree.bin'." << std::endl;
```

---

Load a 32-bit floating point `Octree` from disk, then traverse it manually and
find the number of leaf nodes with fewer than 10 points.

```c++
// This assumes the tree has already been saved to 'tree.bin' (as in the example
// above).

// This convenient typedef saves us a long type name!
using TreeType = mlpack::Octree<mlpack::EuclideanDistance,
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
    for (size_t i = 0; i < node->NumChildren(); ++i)
      stack.push(&node->Child(i));
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

Build an `Octree` and map between original points and new points.

```c++
// See https://datasets.mlpack.org/lcdm_tiny.csv.
arma::mat dataset;
mlpack::data::Load("lcdm_tiny.csv", dataset, true);

// Build the tree.
std::vector<size_t> oldFromNew, newFromOld;
mlpack::Octree<> tree(dataset, oldFromNew, newFromOld);

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
