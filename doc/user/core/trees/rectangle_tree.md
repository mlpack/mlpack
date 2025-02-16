# `RectangleTree`

The `RectangleTree` class represents a generic multidimensional space
partitioning tree.  It is heavily templatized to control splitting behavior and
other behaviors, and is the actual class underlying trees such as the
[`RTree`](r_tree.md).  In general, the `RectangleTree` class is not meant to
be used directly, and instead one of the numerous variants should be used
instead:

 * [`RTree`](r_tree.md)

The `RectangleTree` and its variants are capable of inserting points and
deleting them.  This is different from [`BinarySpaceTree`](binary_space_tree.md)
and other mlpack tree types, where the tree is built entirely in batch at
construction time.  However, this capability comes with a runtime cost, and so
in general the use of `RectangleTree` with mlpack algorithms will be slower than
the batch-construction trees---but, if insert/delete functionality is required,
`RectangleTree` is the only choice.

---

For users who want to use `RectangleTree` directly or with custom behavior,
the full class is still detailed in the subsections below.  `RectangleTree`
supports the [TreeType API](../../../developer/trees.md#the-treetype-api) and
can be used with mlpack's tree-based algorithms, although using custom behavior
may require a template typedef.

 * [Template parameters](#template-parameters)
 * [Constructors](#constructors)
 * [Basic tree properties](#basic-tree-properties)
 * [Bounding distances with the tree](#bounding-distances-with-the-tree)
 * [`StatisticType`](#statistictype) template parameter
 * [`SplitType`](#splittype) template parameter
 * [`DescentType`](#descenttype) template parameter
 * [`AuxiliaryInformationType`](#auxiliaryinformationtype) template parameter
 * [Tree traversals](#tree-traversals)
 * [Example usage](#example-usage)

## See also

<!-- TODO: add links to all distance-based algorithms and other trees? -->

 * [`RTree`](r_tree.md)
 * [R-Tree on Wikipedia](https://en.wikipedia.org/wiki/R-tree)
 * [R-Trees: A Dynamic Index Structure for Spatial Searching (pdf)](http://www-db.deis.unibo.it/courses/SI-LS/papers/Gut84.pdf)
 * [Tree-Independent Dual-Tree Algorithms (pdf)](https://www.ratml.org/pub/pdf/2013tree.pdf)

## Template parameters

The `RectangleTree` class takes six template parameters.  The first three of
these are required by the
[TreeType API](../../../developer/trees.md#template-parameters-required-by-the-treetype-policy)
(see also
[this more detailed section](../../../developer/trees.md#template-parameters)). The
full signature of the class is:

```
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
class RectangleTree;
```

 * `DistanceType`: the [distance metric](../distances.md) to use for distance
   computations.  `RectangleTree` requires that this is
   [`EuclideanDistance`](../distances.md#lmetric), and a compilation error will
   be thrown if any other `DistanceType` is specified.

 * `StatisticType`: this holds auxiliary information in each tree node.  By
   default, [`EmptyStatistic`](#emptystatistic) is used, which holds no
   information.
   - See the [`StatisticType`](#statistictype) section for more details.

 * `MatType`: the type of matrix used to represent points.  Must be a type
   matching the [Armadillo API](../../matrices.md).  By default, `arma::mat` is
   used, but other types such as `arma::fmat` or similar will work just fine.

 * `SplitType`: the class defining how an individual `RectangleTree` node
   should be split.  By default, [`RTreeSplit`](#rtreesplit) is used.
   - See the [`SplitType`](#splittype) section for more details.

 * `DescentType`: the class defining how a child node is chosen for point
   insertion.  By default, [`RTreeDescentHeuristic`](#rtreedescentheuristic) is
   used.
   - See the [`DescentType`](#descenttype) section for more details.

 * `AuxiliaryInformationType`: holds information specific to the variant of the
   `RectangleTree`.  By default, [`NoAuxiliaryInformation`] is used.

Note that the TreeType API requires trees to have only three template
parameters.  In order to use a `RectangleTree` with its six template parameters
with an mlpack algorithm that needs a TreeType, it is easiest to define a
template typedef:

```
template<typename DistanceType, typename StatisticType, typename MatType>
using CustomTree = Rectangle<DistanceType, StatisticType, MatType,
    CustomSplitType, CustomDescentType, CustomAuxiliaryInformationType>
```

Here, `CustomSplitType`, `CustomDescentType`, and
`CustomAuxiliaryInformationType` are the desired splitting and descent
strategies and auxiliary information type.  This is the way that all
`RectangleTree` variants (such as [`RTree`](r_tree.md)) are defined.

## Constructors

`RectangleTree`s are constructed by inserting points in a dataset sequentially.
The dataset is not permuted during the construction process.

---

 * `node = RectangleTree(data)`
 * `node = RectangleTree(data, maxLeafSize=20, minLeafSize=8)`
 * `node = RectangleTree(data, maxLeafSize=20, minLeafSize=8, maxNumChildren=5, minNumChildren=2)`
   - Construct a `RectangleTree` on the given `data` with the given construction
     parameters.
   - Default template parameters are used, meaning that this tree will be a
     [`RTree`](r_tree.md).
   - By default, `data` is copied.  Avoid a copy by using `std::move()` (e.g.
     `std::move(data)`); when doing this, `data` will be set to an empty matrix.

---

 * `node = RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType, AuxiliaryInformationType>(data)`
 * `node = RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType, AuxiliaryInformationType>(data, maxLeafSize=20, minLeafSize=8)`
 * `node = RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType, AuxiliaryInformationType>(data, maxLeafSize=20, minLeafSize=8, maxNumChildren=5, minNumChildren=2)`
   - Construct a `RectangleTree` on the given `data`, using custom template
     parameters to control the behavior of the tree and the given construction
     parameters.
   - By default, `data` is copied.  Avoid a copy by using `std::move()` (e.g.
     `std::move(data)`); when doing this, `data` will be set to an empty matrix.

---

 * `node = RectangleTree(dimensionality)`
   - Construct an empty `RectangleTree` with no children, no points, and default
     template parameters.
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

 - The name `node` is used here for `RectangleTree` objects instead of `tree`,
   because each `RectangleTree` object is a single node in the tree.  The
   constructor returns the node that is the root of the tree.

 - See also the
   [developer documentation on tree constructors](../../../developer/trees.md#constructors-and-destructors).

<!-- TODO: add links to RectangleTree above when it is documented -->

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

Once a `RectangleTree` object is constructed, various properties of the tree
can be accessed or inspected.  Many of these functions are required by the
[TreeType API](../../../developer/trees.md#the-treetype-api).

### Navigating the tree

 * `node.NumChildren()` returns the number of children in `node`.  This is `0`
   if `node` is a leaf, and between the values of `node.MinNumChildren()` and
   `node.MaxNumChildren()` (inclusive) otherwise.

 * `node.IsLeaf()` returns a `bool` indicating whether or not `node` is a leaf.

 * `node.Child(i)` returns a `RectangleTree&` that is the `i`th child.
   - `i` must be less than `node.NumChildren()`.
   - This function should only be called if `node.NumChildren()` is not `0`
     (e.g. if `node` is not a leaf).  Note that this returns a valid
     `RectangleTree&` that can itself be used just like the root node of the
     tree!

 * `node.Parent()` will return a `RectangleTree*` that points to the parent of
   `node`, or `NULL` if `node` is the root of the `RectangleTree`.

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

 * `node.AuxiliaryInfo()` returns an `AuxiliaryInformationType&` that holds any
   auxiliary information required by the node.

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
   - If `node` is not a leaf, this will return `0`, as `RectangleTree` only
     holds points directly in its leaves.
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
   - Point indices are not necessarily contiguous for `RectangleTree`s; that is,
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
     `RectangleTree`s; that is, `node.Descendant(i) + 1` is not necessarily
     `node.Descendant(i + 1)`.

---

### Accessing computed bound quantities of a tree

The following quantities are cached for each node in a `RectangleTree`, and so
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

 * `node.Center(center)` computes the center of the hyperrectangle bounding box
   of `node` and stores it in `center`.
   - `center` should be of type `arma::Col<ElemType>&`, where `ElemType` is the
     element type of the specified `MatType`.
   - `center` will be set to have size equivalent to the dimensionality of the
     dataset held by `node`.
   - This is equivalent to calling `node.Bound().Center(center)`.

 * A `RectangleTree` can be serialized with
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
     furthest from) the `RectangleTree` node `other`, with respect to the
     `MinDistance()` (or `MaxDistance()`) function.
   - If there is a tie, the node with the lowest index is returned.
   - If `node` is a leaf, `0` is returned.

---

 * `node.MinDistance(point)`
 * `node.MinDistance(other)`
   - Return a `double` indicating the minimum possible distance between `node`
     and `point`, or the `RectangleTree` node `other`.
   - This is equivalent to the minimum possible distance between any point
     contained in the bounding hyperrectangle of `node` and `point`, or between
     any point contained in the bounding hyperrectangle of `node` and any point
     contained in the bounding hyperrectangle of `other`.
   - `point` should be a column vector type of the same type as `MatType`.
     (e.g., if `MatType` is `arma::mat`, then `point` should be an `arma::vec`.)

 * `node.MaxDistance(point)`
 * `node.MaxDistance(other)`
   - Return a `double` indicating the maximum possible distance between `node`
     and `point`, or the `RectangleTree` node `other`.
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

Like every mlpack tree, the `RectangleTree` class provides a [single-tree and
dual-tree traversal](../../../developer/trees.md#traversals) that can be paired
with a [`RuleType` class](../../../developer/trees.md#rules) to implement a
single-tree or dual-tree algorithm.

 * `RectangleTree::SingleTreeTraverser`
   - Implements a depth-first single-tree traverser.

 * `RectangleTree::DualTreeTraverser`
   - Implements a dual-depth-first dual-tree traverser.

## `StatisticType`

Each node in a `RectangleTree` holds an instance of the `StatisticType` class.
This class can be used to store additional bounding information or other cached
quantities that a `RectangleTree` does not already compute.

mlpack provides a few existing `StatisticType` classes, and a custom
`StatisticType` can also be easily implemented:

 * [`EmptyStatistic`](#emptystatistic): an empty statistic class that does not
   hold any information
 * [Custom `StatisticType`s](#custom-statistictypes): implement a fully custom
   `StatisticType`

*Note:* this section is still under construction---not all statistic types are
documented yet.

### `EmptyStatistic`

The `EmptyStatistic` class is an empty placeholder class that is used as the
default `StatisticType` template parameter for mlpack trees.

The class ***does not hold any members and provides no functionality***.
[See the implementation.](/src/mlpack/core/tree/statistic.hpp)

### Custom `StatisticType`s

A custom `StatisticType` is trivial to implement.  Only a default constructor
and a constructor taking a `RectangleTree` is necessary.

```
class CustomStatistic
{
 public:
  // Default constructor required by the StatisticType policy.
  CustomStatistic();

  // Construct a CustomStatistic for the given fully-constructed
  // `RectangleTree` node.  Here we have templatized the tree type to make it
  // easy to handle any type of `RectangleTree`.
  template<typename TreeType>
  StatisticType(TreeType& node);

  //
  // Adding any additional precomputed bound quantities can be done; these
  // quantities should be computed in the constructor.  They can then be
  // accessed from the tree with `node.Stat()`.
  //
};
```

*Example*: suppose we wanted to know, for each node, the exact time at which it
was created.  A `StatisticType` could be created that has a
[`std::time_t`](https://en.cppreference.com/w/cpp/chrono/c/time_t) member,
whose value is computed in the constructor.

## `SplitType`

The `SplitType` template parameter controls the algorithm used to split each
node of a `RectangleTree` while building.  The splitting strategy used can be
entirely arbitrary---the `SplitType` simply needs to split a leaf node and a
non-leaf node into children.

mlpack provides several drop-in choices for `SplitType`, and it is also possible
to write a fully custom split:

 * [`RTreeSplit`](#rtreesplit): splits according to a simple binary heuristic
 * [Custom `SplitType`s](#custom-splittypes): implement a fully custom
   `SplitType` class

*Note:* this section is still under construction---not all split types are
documented yet.

### `RTreeSplit`

The `RTreeSplit` class implements the original R-tree splitting strategy and can
be used with the [`RectangleTree`](#rectangletree) class.  This is the splitting
strategy used for the [`RTree`](r_tree.md) class, and is the same strategy
proposed in the [original paper
(pdf)](http://www-db.deis.unibo.it/courses/SI-LS/papers/Gut84.pdf).  The
strategy works as follows:

 * Find the two furthest-apart points (or children if the node is not a leaf).
 * Create two children with each point (or child) as the only point (or child).
 * Iteratively add each remaining point (or child) to the new child whose
   hyperrectangle bound volume increases the least.

For implementation details, see
[the source code](/src/mlpack/core/tree/rectangle_tree/r_tree_split_impl.hpp).

### Custom `SplitType`s

Custom split strategies for a `RectangleTree` can be implemented via the
`SplitType` template parameter.  By default, the [`RTreeSplit`](#rtreesplit)
splitting strategy is used, but it is also possible to implement and use a
custom `SplitType`.  Any custom `SplitType` class must implement the following
signature:

```c++
class SplitType
{
 public:
  // Given the leaf node `tree`, split into multiple nodes.  `TreeType` will be
  // the relevant `RectangleTree` type.  `tree` should be modified directly.
  //
  // `relevels` is an auxiliary array used by some splitting strategies to
  // indicate whether a node needs to be reinserted into the tree.
  template<typename TreeType>
  static void SplitLeafNode(TreeType* tree, std::vector<bool>& relevels);

  // Given the non-leaf node `tree`, split into multiple nodes.  `TreeType` will
  // be the relevant `RectangleTree` type.  `tree` should be modified directly.
  //
  // `relevels` is an auxiliary array used by some splitting strategies to
  // indicate whether a node needs to be reinserted into the tree.
  template<typename TreeType>
  static void SplitNonLeafNode(TreeType* tree, std::vector<bool>& relevels);
};
```

## `DescentType`

The `DescentType` template parameter controls the algorithm used to assign child
points and child nodes to nodes in a `RectangleTree`.  The strategy used can be
arbitrary: the `DescentType` simply needs to return an index of a child to
insert a point or node into.

mlpack provides several drop-in choices for `DescentType`, and it is also
possible to write a fully custom split:

 * [`RTreeDescentHeuristic`](#rtreedescentheuristic): selects the closest child,
   which is the child whose volume will increase the least
 * [Custom `SplitType`s](#custom-splittypes): implement a fully custom
   `SplitType` class

*Note:* this section is still under construction---not all split types are
documented yet.

### `RTreeDescentHeuristic`

The `RTreeDescentHeuristic` is the default descent strategy for the
`RectangleTree` and is used by the [`RTree`](r_tree.md).  The strategy is
simple: the child node whose volume will increase the least is chosen as the
child to insert a point or other node into.

For implementation details, see [the source
code](/src/mlpack/core/tree/rectangle_tree/r_tree_descent_heuristic.hpp).

### Custom `DescentType`s

Custom descent strategies for a `RectangleTree` can be implemented via the
`DescentType` template parameter.  By default, the
[`RTreeDescentHeuristic`](#rtreedescentheuristic) descent strategy is used,
but it is also possible to implement and use a custom `DescentType`.  Any custom
`DescentType` class must implement the following signature:

```c++
class DescentType
{
 public:
  // Return a `size_t` indicating which child of `node` should be chosen to
  // insert `point` in.
  //
  // `TreeType` will be the relevant `RectangleTree` type.
  template<typename TreeType>
  static size_t ChooseDescentNode(const TreeType* node, const size_t point);

  // Return a `size_t` indicating which child of `node` should be chosen to
  // insert `insertedNode` in.
  //
  // `TreeType` will be the relevant `RectangleTree` type.
  template<typename TreeType>
  static size_t ChooseDescentNode(const TreeType* node,
                                  const TreeType* insertedNode);
};
```

## `AuxiliaryInformationType`

The `AuxiliaryInformationType` template parameter holds any auxiliary
information required by the `SplitType` or `DescentType` strategies.  By
default, the `NoAuxiliaryInformation` class is used, which holds nothing.

### Custom `AuxiliaryInformationType`s

Custom `AuxiliaryInformationType`s can be implemented and used with the
`AuxiliaryInformationType` template parameter.  Any custom
`AuxiliaryInformationType` class must implement the following signature:

```c++
// TreeType will be the type of RectangleTree that the auxiliary information
// type is being used in.
template<typename TreeType>
class CustomAuxiliaryInformationType
{
 public:
  // Default constructor is required.
  CustomAuxiliaryInformationType();
  // Construct the object with a tree node that may not yet be constructed.
  CustomAuxiliaryInformationType(TreeType* node);
  // Construct the object with another object and another tree node, optionally
  // making a 'deep copy' instead of just copying pointers where relevant.
  CustomAuxiliaryInformationType(const CustomAuxiliaryInformationType& other,
                                 TreeType* node,
                                 const bool deepCopy = true);

  // Just before a point is inserted into a node, this is called.
  // `node` is the node that will have `node.Dataset().col(point)` inserted into
  // it.
  //
  // Optionally, this method can manipulate `node`.  If so, `true` should be
  // returned to indicate that `node` was changed.  Otherwise, return `false`
  // and the RectangleTree will perform its default behavior.
  bool HandlePointInsertion(TreeType* node, const size_t point);

  // Just before a child node is inserted into a node, this is called.
  // `node` is the node that will have `nodeToInsert` inserted into it as a
  // child.
  //
  // Optionally, this method can manipulate `node`.  If so, `true` should be
  // returned to indicate that `node` was changed.  Otherwise, return `false`
  // and the RectangleTree will perform its default behavior.
  bool HandleNodeInsertion(TreeType* node,
                           TreeType* nodeToInsert,
                           const bool atMaxDepth);

  // Just before a point is deleted from a node, this is called.
  // `node` is the node that will have `node.Dataset().col(point)` deleted from
  // it.
  //
  // Optionally, this method can manipulate `node`.  If so, `true` should be
  // returned to indicate that `node` was changed.  Otherwise, return `false`
  // and the RectangleTree will perform its default behavior.
  bool HandlePointDeletion(TreeType* node, const size_t point);

  // Just before a child node is deleted from a node, this is called.
  // `node` is the node that will have `node.Child(nodeIndex)` deleted from it.
  //
  // Optionally, this method can manipulate `node`.  If so, `true` should be
  // returned to indicate that `node` was changed.  Otherwise, return `false`
  // and the RectangleTree will perform its default behavior.
  bool HandleNodeRemoval(TreeType* node, const size_t nodeIndex);

  // When `node` is changed, this is called so that the auxiliary information
  // can be updated.  If information needs to be propagated upward, return
  // `true` and then `UpdateAuxiliaryInfo(node->Parent())` will be called.
  bool UpdateAuxiliaryInfo(TreeType* node);
};
```

## Example usage

The `RectangleTree` class is only really necessary when a custom split type or
custom descent strategy is intended to be used.  For simpler use cases, one of
the typedefs of `RectangleTree` (such as [`RTree`](r_tree.md)) will suffice.

For this reason, all of the examples below explicitly specify all six template
parameters of `RectangleTree`.
[Writing a custom splitting strategy](#custom-splittypes),
[writing a custom descent strategy](#custom-descenttypes),
and [writing a custom auxiliary information
type](#custom-auxiliaryinformationtypes) are discussed in the previous sections.
Each of the parameters in the examples below can be trivially changed for
different behavior.

---

Build a `RectangleTree` on the `cloud` dataset and print basic statistics
about the tree.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset, true);

// Build the rectangle tree with a leaf size of 10.  (This means that leaf nodes
// cannot contain more than 10 points.)
//
// The std::move() means that `dataset` will be empty after this call, and no
// data will be copied during tree building.
mlpack::RectangleTree<mlpack::EuclideanDistance,
                      mlpack::EmptyStatistic,
                      arma::mat,
                      mlpack::RTreeSplit,
                      mlpack::RTreeDescentHeuristic,
                      mlpack::NoAuxiliaryInformation> tree(std::move(dataset));

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

// Compute the center of the RectangleTree.
arma::vec center;
tree.Center(center);
std::cout << "Center of tree: " << center.t();
```

---

Build two `RectangleTree`s on subsets of the corel dataset and compute minimum
and maximum distances between different nodes in the tree.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::mat dataset;
mlpack::data::Load("corel-histogram.csv", dataset, true);

// Convenience typedef for the tree type.
using TreeType = mlpack::RectangleTree<mlpack::EuclideanDistance,
                                       mlpack::EmptyStatistic,
                                       arma::mat,
                                       mlpack::RTreeSplit,
                                       mlpack::RTreeDescentHeuristic,
                                       mlpack::NoAuxiliaryInformation>;

// Build trees on the first half and the second half of points.
TreeType tree1(dataset.cols(0, dataset.n_cols / 2));
TreeType tree2(dataset.cols(dataset.n_cols / 2 + 1, dataset.n_cols - 1));

// Compute the maximum distance between the trees.
std::cout << "Maximum distance between tree root nodes: "
    << tree1.MaxDistance(tree2) << "." << std::endl;

// Get the leftmost grandchild of the first tree's root---if it exists.
if (!tree1.IsLeaf() && !tree1.Child(0).IsLeaf())
{
  TreeType& node1 = tree1.Child(0).Child(0);

  // Get the leftmost grandchild of the second tree's root---if it exists.
  if (!tree2.IsLeaf() && !tree2.Child(0).IsLeaf())
  {
    TreeType& node2 = tree2.Child(0).Child(0);

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

Build a `RectangleTree` on 32-bit floating point data and save it to disk.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::fmat dataset;
mlpack::data::Load("corel-histogram.csv", dataset);

// Build the RectangleTree using 32-bit floating point data as the matrix
// type.  We will still use the default EmptyStatistic and EuclideanDistance
// parameters.  A leaf size of 100 is used here.
mlpack::RectangleTree<mlpack::EuclideanDistance,
                      mlpack::EmptyStatistic,
                      arma::fmat,
                      mlpack::RTreeSplit,
                      mlpack::RTreeDescentHeuristic,
                      mlpack::NoAuxiliaryInformation> tree(
    std::move(dataset), 100);

// Save the tree to disk with the name 'tree'.
mlpack::data::Save("tree.bin", "tree", tree);

std::cout << "Saved tree with " << tree.Dataset().n_cols << " points to "
    << "'tree.bin'." << std::endl;
```

---

Load a 32-bit floating point `RectangleTree` from disk, then traverse it
manually and find the number of leaf nodes with less than 10 points.

```c++
// This assumes the tree has already been saved to 'tree.bin' (as in the example
// above).

// This convenient typedef saves us a long type name!
using TreeType = mlpack::RectangleTree<mlpack::EuclideanDistance,
                                       mlpack::EmptyStatistic,
                                       arma::fmat,
                                       mlpack::RTreeSplit,
                                       mlpack::RTreeDescentHeuristic,
                                       mlpack::NoAuxiliaryInformation>;

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

Build a `RectangleTree` by iteratively inserting points from the corel dataset,
print some information, and then remove a few randomly chosen points.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::mat dataset;
mlpack::data::Load("corel-histogram.csv", dataset, true);

// This convenient typedef saves us a long type name!
using TreeType = mlpack::RectangleTree<mlpack::EuclideanDistance,
                                       mlpack::EmptyStatistic,
                                       arma::mat,
                                       mlpack::RTreeSplit,
                                       mlpack::RTreeDescentHeuristic,
                                       mlpack::NoAuxiliaryInformation>;

// Create an empty tree of the right dimensionality.
TreeType t(dataset.n_rows);

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
