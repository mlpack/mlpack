# `SpillTree`

The `SpillTree` class represents a generic multidimensional binary space
partitioning tree that allows overlapping volumes between nodes, also known as a
'hybrid spill tree'.  It is heavily templatized to control splitting behavior
and other behaviors, and is the actual class underlying trees such as the
[`SPTree`](sp_tree.md).  In general, the `SpillTree` class is not meant to be
used directly, and instead one of the handful of variants should be used
instead:

 * [`SPTree`](sp_tree.md)
 * [`MeanSPTree`](mean_sp_tree.md)
 * [`NonOrtSPTree`](non_ort_sp_tree.md)
 * [`NonOrtMeanSPTree`](non_ort_mean_sp_tree.md)

The `SpillTree` is similar to the [`BinarySpaceTree`](binary_space_tree.md),
except that the two children of a node are allowed to overlap, and thus a single
point can be contained in multiple branches of the tree.  This can be useful to,
e.g., improve nearest neighbor performance when using [defeatist traversals
without backtracking](#tree-traversals).

---

For users who want to use `SpillTree` directly or with custom behavior,
the full class is still detailed in the subsections below.  `SpillTree` supports
the [TreeType API](../../../developer/trees.md#the-treetype-api) and can be used
with mlpack's tree-based algorithms, although using custom behavior may require
a template typedef.

 * [Template parameters](#template-parameters)
 * [Constructors](#constructors)
 * [Basic tree properties](#basic-tree-properties)
 * [Bounding distances with the tree](#bounding-distances-with-the-tree)
 * [`HyperplaneType`](#hyperplanetype) template parameter
 * [`SplitType`](#splittype) template parameter
 * [Tree traversals](#tree-traversals)
 * [Example usage](#example-usage)

## See also

<!-- TODO: add links to all distance-based algorithms and other trees? -->

 * [`SPTree`](sp_tree.md)
 * [`MeanSPTree`](mean_sp_tree.md)
 * [`NonOrtSPTree`](non_ort_sp_tree.md)
 * [`NonOrtMeanSPTree`](non_ort_mean_sp_tree.md)
 * [`BinarySpaceTree`](binary_space_tree.md)
 * [An Investigation of Practical Approximate Nearest Neighbor Algorithms (pdf)](https://proceedings.neurips.cc/paper/2004/file/1102a326d5f7c9e04fc3c89d0ede88c9-Paper.pdf)
 * [Tree-Independent Dual-Tree Algorithms (pdf)](https://www.ratml.org/pub/pdf/2013tree.pdf)

## Template parameters

The `SpillTree` class takes five template parameters.  The first three of
these are required by the
[TreeType API](../../../developer/trees.md#template-parameters-required-by-the-treetype-policy)
(see also
[this more detailed section](../../../developer/trees.md#template-parameters)). The
full signature of the class is:

```
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneDistanceType,
                  typename HyperplaneMatType> class HyperplaneType,
         template<typename SplitDistanceType,
                  typename SplitMatType> class SplitType>
class SpillTree;
```

 * `DistanceType`: the [distance metric](../distances.md) to use for distance
   computations.  By default, this is
   [`EuclideanDistance`](../distances.md#lmetric).

 * `StatisticType`: this holds auxiliary information in each tree node.  By
   default, [`EmptyStatistic`](binary_space_tree.md#emptystatistic) is used,
   which holds no information.
   - See the [`StatisticType`](binary_space_tree.md#statistictype) section in
     the `BinarySpaceTree` documentation for more details.

 * `MatType`: the type of matrix used to represent points.  Must be a type
   matching the [Armadillo API](../../matrices.md).  By default, `arma::mat` is
   used, but other types such as `arma::fmat` or similar will work just fine.

 * `HyperplaneType`: the class defining the type of the hyperplane that will
   split each node.  By default,
   [`AxisOrthogonalHyperplane`](#axisorthogonalhyperplane) is used.
   - See the [`HyperplaneType`](#hyperplanetype) section for more details.

 * `SplitType`: the class defining how an individual `SpillTree` node
   should be split.  By default, [`MidpointSpaceSplit`](#midpointspacesplit) is
   used.
   - See the [`SplitType`](#splittype) section for more details.

Note that the TreeType API requires trees to have only three template
parameters.  In order to use a `SpillTree` with its five template parameters
with an mlpack algorithm that needs a TreeType, it is easiest to define a
template typedef:

```
template<typename DistanceType, typename StatisticType, typename MatType>
using CustomTree = SpillTree<DistanceType, StatisticType, MatType,
    CustomHyperplaneType, CustomSplitType>
```

Here, `CustomHyperplaneType` and `CustomSplitType` are the desired hyperplane
type and split strategy.  This is the way that all `SpillTree` variants (such as
[`SPTree`](sp_tree.md)) are defined.

## Constructors

`SpillTree`s are constructed by iteratively finding splitting hyperplanes, and
points within a margin of the hyperplane are assigned to *both* child nodes.
Unlike the constructors of
[`BinarySpaceTree`](binary_space_tree.md#constructors), the dataset is not
permuted during construction.

---

 * `node = SpillTree(data, tau=0.0, maxLeafSize=20, rho=0.7)`
   - Construct a `SpillTree` on the given `data`, using the specified
     hyperparameters to control tree construction behavior.
   - Default template parameters are used, meaning that this tree will be a
     [`SPTree`](sp_tree.md).
   - By default, a reference to `data` is stored.  If `data` goes out of scope
     after tree construction, memory errors will occur!  To avoid this, either
     pass the dataset or a copy with `std::move()` (e.g. `std::move(data)`);
     when doing this, `data` will be set to an empty matrix.

---

 * `node = SpillTree<DistanceType, StatisticType, MatType, HyperplaneType, SplitType>(data, tau=0.0, maxLeafSize=20, rho=0.7)`
   - Construct a `SpillTree` on the given `data`, using custom template
     parameters, and using the specified hyperparameters to control tree
     construction behavior.
   - By default, a reference to `data` is stored.  If `data` goes out of scope
     after tree construction, memory errors will occur!  To avoid this, either
     pass the dataset or a copy with `std::move()` (e.g. `std::move(data)`);
     when doing this, `data` will be set to an empty matrix.

---

 * `node = SpillTree()`
   - Construct an empty `SpillTree` with no children, no points, and
     default template parameters.

---

***Notes:***

 - The name `node` is used here for `SpillTree` objects instead of `tree`,
   because each `SpillTree` object is a single node in the tree.  The
   constructor returns the node that is the root of the tree.

 - Inserting individual points or removing individual points from a
   `SpillTree` is not supported, because this generally results in a tree
   with very suboptimal hyperplane splits.  It is better to simply build a new
   `SpillTree` on the modified dataset.  For trees that support individual
   insertion and deletions, see the [`RectangleTree`](rectangle_tree.md) class
   and all its variants (e.g. [`RTree`](r_tree.md),
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

Once a `SpillTree` object is constructed, various properties of the tree
can be accessed or inspected.  Many of these functions are required by the
[TreeType API](../../../developer/trees.md#the-treetype-api).

### Navigating the tree

 * `node.NumChildren()` returns the number of children in `node`.  This is
   either `2` if `node` has children, or `0` if `node` is a leaf.

 * `node.IsLeaf()` returns a `bool` indicating whether or not `node` is a leaf.

 * `node.Child(i)` returns a `SpillTree&` that is the `i`th child.
   - `i` must be `0` or `1`.
   - This function should only be called if `node.NumChildren()` is not `0`
     (e.g. if `node` is not a leaf).  Note that this returns a valid
     `SpillTree&` that can itself be used just like the root node of the
     tree!
   - `node.Left()` and `node.Right()` are convenience functions specific to
     `SpillTree` that will return `SpillTree*` (pointers) to the left and right
     children, respectively, or `NULL` if `node` has no children.

 * `node.Parent()` will return a `SpillTree*` that points to the parent of
   `node`, or `NULL` if `node` is the root of the `SpillTree`.

---

### Accessing members of a tree

 * `node.Overlap()` will return a `bool` that is `true` if `node`'s children are
   overlapping, and `false` otherwise.

 * `node.Hyperplane()` will return a `HyperplaneType&` object that represents
   the splitting hyperplane of `node`.
   - All points in `node.Left()` are to the left of `node.Hyperplane()` if
     `node.Overlap()` is `false`; otherwise, all points in `node.Left()` are to
     the left of `node.Hyperplane() + tau`.
   - All points in `node.Right()` are to the right of `node.Hyperplane()` if
     `node.Overlap()` is `false`; otherwise, all points in `node.Right()` are to
     the right of `node.Hyperplane() - tau`.

 * `node.Bound()` will return a
   [`const HRectBound&`](binary_space_tree.md#hrectbound) representing the
   bounding box associated with `node`.
   - If a [custom `HyperplaneType`](#hyperplanetype) is specified, then the
     `BoundType` associated with that hyperplane type is returned instead.
   - If a [custom `DistanceType` and/or `MatType`](#template-parameters) are
     specified, then a `const HRectBound<DistanceType, ElemType>&` is returned
     (or a `BoundType` with that `DistanceType`, if a custom `HyperplaneType`
     was also specified).
     * `ElemType` is the element type of the specified `MatType` (e.g. `double`
       for `arma::mat`, `float` for `arma::fmat`, etc.).

 * `node.Stat()` will return a `StatisticType&` holding the statistics of the
   node that were computed during tree construction.

 * `node.Distance()` will return a `DistanceType&`.

See also the
[developer documentation](../../../developer/trees.md#basic-tree-functionality)
for basic tree functionality in mlpack.

---

### Accessing data held in a tree

 * `node.Dataset()` will return a `const MatType&` that is the dataset the
   tree was built on.

 * `node.NumPoints()` returns a `size_t` indicating the number of points held
   directly in `node`.
   - If `node` is not a leaf, this will return `0`, as `SpillTree` only holds
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

The following quantities are cached for each node in a `SpillTree`, and so
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

 * A `SpillTree` can be serialized with
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
     right) that is closest to (or furthest from) the `SpillTree` node
     `other`, with respect to the `MinDistance()` (or `MaxDistance()`) function.
   - If there is a tie, `0` (the left child) is returned.
   - If `node` is a leaf, `0` is returned.

---

 * `node.MinDistance(point)`
 * `node.MinDistance(other)`
   - Return a `double` indicating the minimum possible distance between `node`
     and `point`, or the `SpillTree` node `other`.
   - This is equivalent to the minimum possible distance between any point
     contained in the bounding hyperrectangle of `node` and `point`, or between
     any point contained in the bounding hyperrectangle of `node` and any point
     contained in the bounding hyperrectangle of `other`.
   - `point` should be a column vector type of the same type as `MatType`.
     (e.g., if `MatType` is `arma::mat`, then `point` should be an `arma::vec`.)

 * `node.MaxDistance(point)`
 * `node.MaxDistance(other)`
   - Return a `double` indicating the maximum possible distance between `node`
     and `point`, or the `SpillTree` node `other`.
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

Like every mlpack tree, the `SpillTree` class provides a [single-tree and
dual-tree traversal](../../../developer/trees.md#traversals) that can be paired
with a [`RuleType` class](../../../developer/trees.md#rules) to implement a
single-tree or dual-tree algorithm.

 * `SpillTree::SingleTreeTraverser`
   - Implements a depth-first single-tree traverser.

 * `SpillTree::DualTreeTraverser`
   - Implements a dual-depth-first dual-tree traverser.

However, the spill tree is primarily useful because the overlapping nodes allow
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

 * `SpillTree::DefeatistSingleTreeTraverser`
   - Implements a depth-first single-tree defeatist traverser with no
     backtracking.  Traversal will terminate after the first leaf is visited.

 * `SpillTree::DefeatistDualTreeTraverser`
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

## `HyperplaneType`

Each node in a `SpillTree` corresponds to some region in space that contains all
of the descendant points in the node.  Similar to the [`KDTree`](kdtree.md), this
region is a hyperrectangle; however, instead of representing that hyperrectangle
explicitly like the `KDTree` with the
[`HRectBound`](binary_space_tree.md#hrectbound) class, the `SpillTree`
represents the region *implicitly*, with each node storing only the hyperplane
and margin required to determine whether a point belongs to the left node, the
right node, or both.

The type of hyperplane (e.g. axis-aligned or arbitrary) can be controlled by the
`HyperplaneType` template parameter.  mlpack supplies two drop-in classes that
can be used for `HyperplaneType`, and it is also possible to write a custom
`HyperplaneType`:

 * [`AxisOrthogonalHyperplane`](#axisorthogonalhyperplane): uses hyperplanes
   that are axis-orthogonal (or axis-aligned).
 * [`Hyperplane`](#hyperplane): uses arbitrary hyperplanes specified by any
   vector.
 * [Custom `HyperplaneType`s](#custom-hyperplanetypes): implement a fully custom
   `HyperplaneType` class

### `AxisOrthogonalHyperplane`

The `AxisOrthogonalHyperplane` class is used to provide an axis-orthogonal split
for a `SpillTree`.  That is, whether or not a point is on the left or right side
of the split is a very efficient computation using only a single dimension of
the data.

 * The `AxisOrthogonalHyperplane` class defines the following two typedefs:
   - `AxisOrthogonalHyperplane::BoundType`, which is the type of the bound used
     by the spill tree with this hyperplane, is
     [`HRectBound`](binary_space_tree.md#hrectbound), or
     `HRectBound<DistanceType, MatType>` if custom
     [`DistanceType` and/or `MatType`](#template-parameters) are specified.

   - `AxisOrthogonalHyperplane::ProjVectorType` is `AxisParallelProjVector`, a
     class that simply holds the index of the dimension of the projection
     vector.
     * For more details, see
       [the source code](/src/mlpack/core/tree/space_split/projection_vector.hpp).

 * An `AxisOrthogonalHyperplane` object `h` (e.g. returned with
   `node.Hyperplane()`) has the following members:

   - `h.Project(point)` returns a `double` that is the orthogonal projection of
     `point` onto the tangent vector of the hyperplane `h`.

   - `h.Left(point)` returns `true` if `point` is to the left of `h`.

   - `h.Right(point)` returns `true` if `point` is to the right of `h`.

   - `h.Left(bound)` returns `true` if `bound` (an
     `AxisOrthogonalHyperplane::BoundType`; see the bullet point above) is to
     the left of `h`.

   - `h.Right(bound)` returns `true` if `bound` (an
     `AxisOrthogonalHyperplane::BoundType`; see the bullet point above) is to
     the right of `h`.

 * An `AxisOrthogonalHyperplane` object can be serialized with
   [`data::Save()` and `data::Load()`](../../load_save.md#mlpack-objects).

For more details, see the
[the source code](/src/mlpack/core/tree/space_split/hyperplane.hpp).

### `Hyperplane`

The `Hyperplane` class is used to provide an arbitrary hyperplane split for a
`SpillTree`.  The computation of whether or not a point is on the left or right
side of the split is less efficient than
[`AxisOrthogonalHyperplane`](#axisorthogonalhyperplane), but `Hyperplane` is
able to represent any possible hyperplane.

 * The `Hyperplane` class defines the two following typedefs:
   - `Hyperplane::BoundType`, which is the type of the bound used by the spill
     tree with this hyperplane, is
     [`BallBound`](binary_space_tree.md#ballbound), or `BallBound<DistanceType>`
     if a custom [`DistanceType`](#template-parameters) is specified.

   - `Hyperplane::ProjVectorType` is `ProjVector<>`, an arbitrary projection
     vector class that wraps an `arma::vec`.
     * If a custom `MatType` is specified, then `Hyperplane::ProjVectorType` is
       `ProjVector<MatType>`, which wraps a vector of the same type as
       `MatType`.
     * For more details, see
       [the source code](/src/mlpack/core/tree/space_split/projection_vector.hpp).

 * A `Hyperplane` object `h` (e.g. returned with `node.Hyperplane()`) has the
   following members:

   - `h.Project(point)` returns a `double` that is the orthogonal projection of
     `point` onto the tangent vector of the hyperplane `h`.

   - `h.Left(point)` returns `true` if `point` is to the left of `h`.

   - `h.Right(point)` returns `true` if `point` is to the right of `h`.

   - `h.Left(bound)` returns `true` if `bound` (an
     `AxisOrthogonalHyperplane::BoundType`; see the bullet point above) is to
     the left of `h`.

   - `h.Right(bound)` returns `true` if `bound` (a `Hyperplane::BoundType`; see
     the bullet point above) is to the right of `h`.

 * A `Hyperplane` object can be serialized with
   [`data::Save()` and `data::Load()`](../../load_save.md#mlpack-objects).

For more details, see the
[the source code](/src/mlpack/core/tree/space_split/hyperplane.hpp).

### Custom `HyperplaneType`s

Custom hyperplane types for a spill tree can be implemented via the
`HyperplaneType` template parameter.  By default, the
[`AxisOrthogonalHyperplane`](#axisorthogonalhyperplane) hyperplane type is used,
but it is also possible to implement and use a custom `HyperplaneType`.  Any
custom `HyperplaneType` class must implement the following signature:

```c++
// NOTE: the custom HyperplaneType class must take two template parameters.
template<typename DistanceType, typename MatType>
class HyperplaneType
{
 public:
  // The hyperplane type must specify these two public typedefs, which are used
  // by the spill tree and the splitting strategy.
  //
  // Substitute HRectBound and ProjVector with your choices.
  using BoundType = mlpack::HRectBound<DistanceType,
                                       typename MatType::elem_type>;
  using ProjVectorType = mlpack::ProjVector<MatType>;

  // Empty constructor, which will construct an empty or default hyperplane.
  HyperplaneType();

  // Construct the HyperplaneType with the given projection vector and split
  // value along that projection.
  HyperplaneType(const ProjVectorType& projVector, double splitVal);

  // Compute the projection of the given point (an `arma::vec` or similar type
  // matching the Armadillo API and element type of `MatType`) onto the vector
  // tangent to the hyperplane.
  template<typename VecType>
  double Project(const VecType& point) const;

  // Return true if the point (an `arma::vec` or similar type matching the
  // Armadillo API and element type of `MatType`) falls to the left of the
  // hyperplane.
  template<typename VecType>
  double Left(const VecType& point) const;

  // Return true if the point (an `arma::vec` or similar type matching the
  // Armadillo API and element type of `MatType`) falls to the right of the
  // hyperplane.
  template<typename VecType>
  double Right(const VecType& point) const;

  // Return true if the given bound is fully to the left of the hyperplane.
  bool Left(const BoundType& bound) const;

  // Return true if the given bound is fully to the right of the hyperplane.
  bool Right(const BoundType& bound) const;

  // Serialize the hyperplane using cereal.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);
};
```

## `SplitType`

The `SplitType` template parameter controls the algorithm used to split each
node of a `SpillTree` while building.  The splitting strategy used can be
entirely arbitrary---the `SplitType` only needs to compute a
[`HyperplaneType`](#hyperplanetype) to split a set of points.

mlpack provides two drop-in choices for `SplitType`, and it is also possible
to write a fully custom split:

 * [`MidpointSpaceSplit`](#midpointspacesplit): split a set of points using a
   hyperplane built on the midpoint (median) of points in a dataset.
 * [`MeanSpaceSplit`](#meanspacesplit): split a set of points using a hyperplane
   built on the mean (average) of points in a dataset.
 * [Custom `SplitType`s](#custom-splittypes): implement a fully custom
   `SplitType` class

### `MidpointSpaceSplit`

The `MidpointSpaceSplit` class is a splitting strategy that can be used by
`SpillTree`.  It is the default strategy for splitting [`SPTree`s](sp_tree.md)
and [`NonOrtSPTree`s](non_ort_sp_tree.md).

The splitting strategy for the `MidpointSpaceSplit` class is, given a set of
points:

 * If [`AxisOrthogonalHyperplane`](#axisorthogonalhyperplane) is being used,
   then select the dimension with the maximum width, and use the midpoint of the
   points' values in that dimension.

 * If [`Hyperplane`](#hyperplane) is being used, then estimate the furthest two
   points in the dataset by random sampling, and use the vector connecting those
   points as the tangent vector to the hyperplane.  The midpoint of the points
   projected onto this hyperplane is used as the split value.

Note that `MidpointSpaceSplit` can only be used with a `HyperplaneType` with
`HyperplaneType::ProjVectorType` as either `AxisAlignedProjVector` or
`ProjVector`.

For implementation details, see
[the source code](/src/mlpack/core/tree/space_split/midpoint_space_split_impl.hpp).

### `MeanSpaceSplit`

The `MeanSpaceSplit` class is a splitting strategy that can be used by
`SpillTree`.  It is the splitting strategy used by the
[`MeanSPTree`](mean_sp_tree.md) and the
[`NonOrtMeanSPTree`](non_ort_mean_sp_tree.md) classes.

The splitting strategy for the `MeanSpaceSplit` class is, given a set of
points:

 * If [`AxisOrthogonalHyperplane`](#axisorthogonalhyperplane) is being used,
   then select the dimension with the maximum width, and use the mean of the
   points' values in that dimension.

 * If [`Hyperplane`](#hyperplane) is being used, then estimate the furthest two
   points in the dataset by random sampling, and use the vector connecting those
   points as the tangent vector to the hyperplane.  The mean of the points
   projected onto this hyperplane is used as the split value.

Note that `MeanSpaceSplit` can only be used with a `HyperplaneType` with
`HyperplaneType::ProjVectorType` as either `AxisAlignedProjVector` or
`ProjVector`.

For implementation details, see
[the source code](/src/mlpack/core/tree/space_split/mean_space_split_impl.hpp).

### Custom `SplitType`s

Custom split strategies for a spill tree can be implemented via the
`SplitType` template parameter.  By default, the
[`MidpointSpaceSplit`](#midpointspacesplit) splitting strategy is used, but it
is also possible to implement and use a custom `SplitType`.  Any custom
`SplitType` class must implement the following signature:

```c++
// NOTE: the custom SplitType class must take two template parameters.
template<typename DistanceType, typename MatType>
class SplitType
{
 public:
  // The SplitType class is only required to provide one static function.

  // Create a splitting hyperplane and store it in the given `HyperplaneType`,
  // using the given data and bounding box `bound`.  `data` will be an Armadillo
  // matrix that is the entire dataset, and `points` are the indices of points
  // in `data` that should be split.
  template<typename HyperplaneType>
  static bool SplitSpace(
      const typename HyperplaneType::BoundType& bound,
      const MatType& data,
      const arma::Col<size_t>& points,
      HyperplaneType& hyp);
};
```

## Example usage

The `SpillTree` class is only really necessary when a custom hyperplane type or
custom splitting strategy is intended to be used.  For simpler use cases, one of
the typedefs of `SpillTree` (such as [`SPTree`](sp_tree.md)) will suffice.

For this reason, all of the examples below explicitly specify all five template
parameters of `SPTree`.
[Writing a custom hyperplane type](#custom-hyperplanetypes) and
[writing a custom splitting strategy](#custom-splittypes) are discussed
in the previous sections.  Each of the parameters in the examples below can be
trivially changed for different behavior.

---

Build a `SpillTree` on the `cloud` dataset and print basic statistics about the
tree.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset, true);

// Build the spill tree with a tau (margin) of 0.2 and a leaf size of 10.
// (This means that nodes are split until they contain 10 or fewer points.)
//
// The std::move() means that `dataset` will be empty after this call, and no
// data will be copied during tree building.
mlpack::SpillTree<mlpack::EuclideanDistance,
                  mlpack::EmptyStatistic,
                  arma::mat,
                  mlpack::AxisOrthogonalHyperplane,
                  mlpack::MidpointSpaceSplit> tree(std::move(dataset), 0.2, 10);

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

// Compute the center of the SpillTree.
arma::vec center;
tree.Center(center);
std::cout << "Center of tree: " << center.t();
```

---

Build two `SpillTree`s on subsets of the corel dataset and compute minimum
and maximum distances between different nodes in the tree.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::mat dataset;
mlpack::data::Load("corel-histogram.csv", dataset, true);

// Convenience typedef for the tree type.
using TreeType = mlpack::SpillTree<mlpack::EuclideanDistance,
                                   mlpack::EmptyStatistic,
                                   arma::mat,
                                   mlpack::AxisOrthogonalHyperplane,
                                   mlpack::MidpointSpaceSplit>;

// Build trees on the first half and the second half of points.  Use a tau
// (overlap) parameter of 0.3, which is tuned to this dataset, and a rho value
// of 0.6 to prevent the trees getting too deep.
TreeType tree1(dataset.cols(0, dataset.n_cols / 2), 0.3, 20, 0.6);
TreeType tree2(dataset.cols(dataset.n_cols / 2 + 1, dataset.n_cols - 1),
    0.3, 20, 0.6);

// Compute the maximum distance between the trees.
std::cout << "Maximum distance between tree root nodes: "
    << tree1.MaxDistance(tree2) << "." << std::endl;

// Get the leftmost grandchild of the first tree's root---if it exists.
if (!tree1.IsLeaf() && !tree1.Child(0).IsLeaf())
{
  TreeType& node1 = tree1.Child(0).Child(0);

  // Get the rightmost grandchild of the second tree's root---if it exists.
  if (!tree2.IsLeaf() && !tree2.Child(1).IsLeaf())
  {
    TreeType& node2 = tree2.Child(1).Child(1);

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

Build a `SpillTree` on 32-bit floating point data and save it to disk.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::fmat dataset;
mlpack::data::Load("corel-histogram.csv", dataset);

// Build the SpillTree using 32-bit floating point data as the matrix type.
// We will still use the default EmptyStatistic and EuclideanDistance
// parameters.
mlpack::SpillTree<mlpack::EuclideanDistance,
                  mlpack::EmptyStatistic,
                  arma::fmat,
                  mlpack::AxisOrthogonalHyperplane,
                  mlpack::MidpointSpaceSplit> tree(
    std::move(dataset), 0.1, 20, 0.95);

// Save the tree to disk with the name 'tree'.
mlpack::data::Save("tree.bin", "tree", tree);

std::cout << "Saved tree with " << tree.Dataset().n_cols << " points to "
    << "'tree.bin'." << std::endl;
```

---

Load a 32-bit floating point `SpillTree` from disk, then traverse it
manually and find the number of nodes whose children overlap.

```c++
// This assumes the tree has already been saved to 'tree.bin' (as in the example
// above).

// This convenient typedef saves us a long type name!
using TreeType = mlpack::SpillTree<mlpack::EuclideanDistance,
                                   mlpack::EmptyStatistic,
                                   arma::fmat,
                                   mlpack::AxisOrthogonalHyperplane,
                                   mlpack::MidpointSpaceSplit>;

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

typedef mlpack::SpillTree<mlpack::EuclideanDistance,
                          mlpack::EmptyStatistic,
                          arma::mat,
                          mlpack::AxisOrthogonalHyperplane,
                          mlpack::MidpointSpaceSplit> TreeType;

// Build two trees, one with a lot of overlap, and one with no overlap
// (e.g. tau = 0).
TreeType tree1(dataset, 0.5, 10), tree2(dataset, 0.0, 10);

// Construct the rule types, and then the traversals.
SpillNearestNeighborRule r1(dataset), r2(dataset);

TreeType::DefeatistSingleTreeTraverser<SpillNearestNeighborRule> t1(r1);
TreeType::DefeatistSingleTreeTraverser<SpillNearestNeighborRule> t2(r2);

// Search for the approximate nearest neighbor of point 3 using both trees.
t1.Traverse(3, tree1);
t2.Traverse(3, tree2);

std::cout << "Approximate nearest neighbor of point 3:" << std::endl;
std::cout << " - Spill tree with overlap 0.5 found: point "
    << r1.NearestNeighbor() << ", distance " << r1.NearestDistance()
    << "." << std::endl;

std::cout << " - Spill tree with no overlap found: point "
    << r2.NearestNeighbor() << ", distance " << r2.NearestDistance()
    << "." << std::endl;

// Now search for point 6.
r1.Reset();
r2.Reset();

t1.Traverse(6, tree1);
t2.Traverse(6, tree2);

std::cout << "Approximate nearest neighbor of point 6:" << std::endl;
std::cout << " - Spill tree with overlap 0.5 found: point "
    << r1.NearestNeighbor() << ", distance " << r1.NearestDistance()
    << "." << std::endl;

std::cout << " - Spill tree with no overlap found: point "
    << r2.NearestNeighbor() << ", distance " << r2.NearestDistance()
    << "." << std::endl;
```
