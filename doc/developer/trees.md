# The TreeType policy in mlpack

Trees are an important data structure in mlpack and are used in a number of the
machine learning algorithms that mlpack implements.  Often, the use of trees can
allow significant acceleration of an algorithm; this is generally done by
pruning away large parts of the tree during computation.

Most mlpack algorithms that use trees are not tied to a specific tree but
instead allow the user to choose a tree via the `TreeType` template parameter.
Any tree passed as a `TreeType` template parameter will need to implement a
certain set of functions.  In addition, a tree may optionally specify some
traits about itself with the `TreeTraits` trait class.

This document aims to clarify the abstractions underlying mlpack trees, list and
describe the required functionality of the `TreeType` policy, and point users
towards existing types of trees.

Although this document is long, there may still be errors and unclear areas.  If
you are having trouble understanding anything, please get in touch on Github
someone will help you (and possibly update the documentation afterwards).

## What is a tree?

In mlpack, we assume that we have some sort of data matrix, which might be
sparse or dense (that is, it could be of type `arma::mat` or `arma::sp_mat`,
or any variant that implements the Armadillo API).  This data matrix corresponds
to a collection of points in some space (usually a Euclidean space).  A tree is
a way of organizing this data matrix in a hierarchical manner---so, points that
are nearby should lie in similar nodes.

We can rigorously define what a tree is, using the definition of *space tree*
introduced in the following paper
([pdf link](https://www.ratml.org/pub/pdf/2013tree.pdf):

```text
R.R. Curtin, W.B. March, P. Ram, D.V. Anderson, A.G. Gray, and C.L. Isbell Jr.,
"Tree-independent dual-tree algorithms," in Proceedings of the 30th
International Conference on Machine Learning (ICML '13), pp. 1435--1443, 2013.
```

The definition is:

A *space tree* on a dataset `S` in `R^(N x d)` is an undirected, connected,
acyclic, rooted simple graph with the following properties:

 - Each node (or vertex) holds a number of points (possibly zero) and is
connected to one parent node and a number of child nodes (possibly zero).

 - There is one node in every space tree with no parent; this is the root node
of the tree.

 - Each point in `S` is contained in at least one node.

 - Each node corresponds to some subset of `R^d` that contains each point in the
   node and also the subsets that correspond to each child of the node.

This is really a quite straightforward definition: a tree is hierarchical, and
each node corresponds to some region of the input space.  Each node may have
some number of children, and may hold some number of points.  However, there is
an important terminology distinction to make: the term *points held by a node*
has a different meaning than the term *descendant points held by a node*.  The
points held in a node are just that---points held only in the node.  The
descendant points of a node are the combination of the points held in a node
with the points held in the node's children and the points held in the node's
children's children (and so forth).  For the purposes of clarity in all
discussions about trees, care is taken to differentiate the terms "descendant
point" and "point".

Now, it's also important to note that a point does not *need* to hold any
children, and that a node *can* hold the same points as its children (or its
parent).  Some types of trees do this.  For instance, each node in the cover
tree holds only one point, and may have a child that holds the same point.  As
another example, the `kd`-tree holds its points only in the leaves (at the
bottom of the tree).  More information on space trees can be found in either the
"Tree-independent dual-tree algorithms" paper or any of the related literature.

So there is a huge amount of possible variety in the types of trees that can
fall into the class of *space trees*.  Therefore, it's important to treat them
abstractly, and the `TreeType` policy allows us to do just that.  All we need
to remember is that a node in a tree can be represented as the combination of
some points held in the node, some child nodes, and some geometric structure
that represents the space that all of the descendant points fall into (this is a
restatement of the fourth part of the definition).

## Template parameters required by the TreeType policy

Most everything in mlpack is decomposed into a series of configurable template
parameters, and trees are no exception.  In order to ease usage of high-level
mlpack algorithms, each `TreeType` itself must be a template class taking three
parameters:

 - `DistanceType` -- the underlying distance metric that the tree will be built
   on (see [the DistanceType policy documentation](distances.md) and
   [pre-implemented distances](../user/core/distances.md#distances))
 - `StatisticType` -- holds any auxiliary information that individual algorithms
   may need
 - `MatType` -- the type of the matrix used to represent the data; see
   [Matrices and data](../user/matrices.md).

The reason that these three template parameters are necessary is so that each
`TreeType` can be used as a template template parameter, which can radically
simplify the required syntax for instantiating mlpack algorithms.  By using
template template parameters, a user needs only to write

```
// The RangeSearch class takes a DistanceType and a TreeType template parameter.

// This code instantiates RangeSearch with the ManhattanDistance and a
// QuadTree.  Note that the QuadTree itself is a template, and takes a
// DistanceType, StatisticType, and MatType, just like the policy requires.

// This example ignores the constructor parameters, for the sake of simplicity.
RangeSearch<ManhattanDistance, QuadTree> rs(...);
```

as opposed to the far more complicated alternative, where the user must specify
the values of each template parameter of the tree type:

```
// This is a much worse alternative, where the user must specify the template
// arguments of their tree.
RangeSearch<ManhattanDistance,
            QuadTree<ManhattanDistance, EmptyStatistic, arma::mat>> rs(...);
```

Unfortunately, the price to pay for this user convenience is that *every*
`TreeType` must have three template parameters, and they must be in exactly
that order.  Fortunately, there is an additional benefit: we are guaranteed that
the tree is built using the same distance metric as the method (that is, a user
can't specify different metric types to the algorithm and to the tree, which
they can without template template parameters).

There are two important notes about this:

 - Not every possible input of `DistanceType`, `StatisticType`, and/or `MatType`
   necessarily need to be valid or work correctly for each type of tree.  For
   instance, the `QuadTree` is limited to Euclidean distance metrics and will
   not work otherwise.  Either compile-time static checks or detailed
   documentation can help keep users from using invalid combinations of template
   arguments.

 - Some types of trees have more template parameters than just these three.  One
   example is the generalized binary space tree, where the bounding shape of
   each node is easily made into a fourth template parameter (the
   [`BinarySpaceTree`](../user/core/trees/binary_space_tree.md) class calls this
   the [`BoundType`](../user/core/trees/binary_space_tree.md#boundtype)
   parameter), and the procedure used to split a node is easily made into a
   fifth template parameter (the `BinarySpaceTree` class calls this the
   [`SplitType`](../user/core/trees/binary_space_tree.md#splittype) parameter).
   However, the syntax of template template parameters *requires* that the class
   only has the correct number of template parameters---no more, no less.
   Fortunately, C++11 allows template typedefs, which can be used to provide
   partial specialization of template classes:

```
// This is the definition of the BinarySpaceTree class, which has five template
// parameters.
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename BoundType,
         typename SplitType>
class BinarySpaceTree;

// The 'using' keyword gives us a template typedef, so we can define the
// MeanSplitKDTree template class, which has three parameters and is a valid
// TreeType policy class.
template<typename DistanceType, typename StatisticType, typename MatType>
using MeanSplitKDTree = BinarySpaceTree<DistanceType,
                                        StatisticType,
                                        MatType,
                                        HRectBound<DistanceType>
                                        MeanSplit<BoundType, DistanceType>>;
```

Now, the [`MeanSplitKDTree`](../user/core/trees/mean_split_kdtree.md) class has
only three template parameters and can be used as a `TreeType` policy class in
various mlpack algorithms.  Many types of trees in mlpack have more than three
template parameters and rely on template typedefs to provide simplified
`TreeType` interfaces.

## The TreeType API

As a result of the definition of *space tree* in the previous section, a
simplified API presents itself quite easily.  However, more complex
functionality is often necessary in mlpack, so this leads to more functions
being necessary for a class to satisfy the `TreeType` policy.  Combining this
with the template parameters required for trees given in the previous section
gives us the complete API required for a class implementing the `TreeType`
policy.  Below is the minimal set of functions required with minor documentation
for each function.  (More extensive documentation and explanation is given
afterwards.)

```
// The three template parameters will be supplied by the user, and are detailed
// in the previous section.
template<typename DistanceType,
         typename StatisticType,
         typename MatType>
class ExampleTree
{
 public:
  // This is the element type held by the matrix.
  // It will generally either be `double`, or `float`.
  typedef typename MatType::elem_type ElemType;

  //////////////////////
  //// Constructors ////
  //////////////////////

  // This batch constructor does not modify the dataset, and builds the entire
  // tree using a default-constructed DistanceType.
  ExampleTree(const MatType& data);

  // This batch constructor does not modify the dataset, and builds the entire
  // tree using the given DistanceType.
  ExampleTree(const MatType& data, DistanceType& distance);

  // Initialize the tree from a given cereal archive.  SFINAE (the
  // second argument) is necessary to ensure that the archive is loading, not
  // saving.
  template<typename Archive>
  ExampleTree(
      Archive& ar,
      const typename std::enable_if_c<typename Archive::is_loading>::type* = 0);

  // Release any resources held by the tree.
  ~ExampleTree();

  // ///////////////////////// //
  // // Basic functionality // //
  // ///////////////////////// //

  // Get the dataset that the tree is built on.
  const MatType& Dataset();

  // Get the distance metric that the tree is built with.
  DistanceType& Distance();

  // Get/modify the StatisticType for this node.
  StatisticType& Stat();

  // Return the parent of the node, or NULL if this is the root.
  ExampleTree* Parent();

  // Return the number of children held by the node.
  size_t NumChildren();
  // Return the i'th child held by the node.
  ExampleTree& Child(const size_t i);

  // Return the number of points held in the node.
  size_t NumPoints();
  // Return the index of the i'th point held in the node.
  size_t Point(const size_t i);

  // Return the number of descendant nodes of this node.
  size_t NumDescendantNodes();
  // Return the i'th descendant node of this node.
  ExampleTree& DescendantNode(const size_t i);

  // Return the number of descendant points of this node.
  size_t NumDescendants();
  // Return the index of the i'th descendant point of this node.
  size_t Descendant(const size_t i);

  // Store the center of the bounding region of the node in the given vector.
  void Center(arma::vec& center);

  // ///////////////////////////////////////////////// //
  // // More complex distance-related functionality // //
  // ///////////////////////////////////////////////// //

  // Return the distance between the center of this node and the center of
  // its parent.
  ElemType ParentDistance();

  // Return an upper bound on the furthest possible distance between the
  // center of the node and any point held in the node.
  ElemType FurthestPointDistance();

  // Return an upper bound on the furthest possible distance between the
  // center of the node and any descendant point of the node.
  ElemType FurthestDescendantDistance();

  // Return a lower bound on the minimum distance between the center and any
  // edge of the node's bounding shape.
  ElemType MinimumBoundDistance();

  // Return a lower bound on the minimum distance between the given point and
  // the node.
  template<typename VecType>
  ElemType MinDistance(VecType& point);

  // Return a lower bound on the minimum distance between the given node and
  // this node.
  ElemType MinDistance(ExampleTree& otherNode);

  // Return an upper bound on the maximum distance between the given point and
  // the node.
  template<typename VecType>
  ElemType MaxDistance(VecType& point);

  // Return an upper bound on the maximum distance between the given node and
  // this node.
  ElemType MaxDistance(ExampleTree& otherNode);

  // Return the combined results of MinDistance() and MaxDistance().
  template<typename VecType>
  RangeType<ElemType> RangeDistance(VecType& point);

  // Return the combined results of MinDistance() and MaxDistance().
  RangeType<ElemType> RangeDistance(ExampleTree& otherNode);

  // //////////////////////////////////// //
  // // Serialization (loading/saving) // //
  // //////////////////////////////////// //

  // Serialize the tree (load from the given archive / save to the given
  // archive, depending on its type).
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

 protected:
  // A default constructor; only meant to be used by cereal.  This
  // must be protected so that cereal will work; it does not need
  // to return a valid tree.
  ExampleTree();

  // Friend access must be given for the default constructor.
  friend class cereal::access;
};
```

Although this is significantly more complex than the four-item definition of
*space tree* might suggest, it turns out many of these methods are not
difficult to implement for most reasonable tree types.  It is also important to
realize that this is a *minimum* API; you may implement more complex tree types
at your leisure (and you may include more template parameters too, though you
will have to use template typedefs to provide versions with three parameters;
see the previous section).

Before diving into the detailed documentation for each function, let us consider
a few important points about the implications of this API:

 - ***Trees are not default-constructible*** and should not (in general) provide
   a default constructor.  This helps prevent invalid trees.  In general, any
   instantiated mlpack object should be valid and ready to use---and a tree
   built on no points is not valid or ready to use.

 - ***Trees only need to provide batch constructors.***  Although many tree
   types do have algorithms for incremental insertions, in mlpack this is not
   required because the tree-based algorithms that mlpack implements generally
   assume fully-built, non-modifiable trees.  For this purpose, batch
   construction is perfectly sufficient.  (It's also worth pointing out that for
   some types of trees, like kd-trees, the cost of a handful of insertions often
   outweighs the cost of completely rebuilding the tree.)

 - **Trees must provide a number of distance bounding functions.**  The utility
   of trees generally stems from the ability to place quick bounds on
   distance-related quantities.  For instance, if all the descendant points of a
   node are bounded by a ball of radius `r` and the center of the node is a
   point `c`, then the minimum distance between some point `p` and any
   descendant point of the node is equal to the distance between `p` and `c`
   minus the radius `r`: `d(p, c) - r`.  This is a fast calculation, and
   (usually) provides a decent bound on the minimum distance between `p` and any
   descendant point of the node.

 - ***Trees need to be able to be serialized.***  mlpack uses the cereal library
   for saving and loading objects.  Trees---which can be a part of machine
   learning models---therefore must have the ability to be saved and loaded.
   Making this all work requires a protected constructor (part of the API) and
   generally makes it impossible to hold references instead of pointers
   internally, because if a tree is loaded from a file then it must own the
   dataset it is built on and the distance metric it uses (this also means that
   a destructor must exist for freeing these resources).

Now, we can consider each part of the API more rigorously.

## Rigorous API documentation

This section is divided into five parts, detailing each of the parts of the API.

### Template parameters

An earlier section discussed the three different template parameters that are
required by the `TreeType` policy.

#### `DistanceType`

The [DistanceType policy](distances.md) provides one method that will be useful
for tree building and other operations:

```
// This function is required by the DistanceType policy.
// Evaluate the distance metric between two points (which may be of different
// types).
template<typename VecTypeA, typename VecTypeB>
double Evaluate(const VecTypeA& a, const VecTypeB& b);
```

Note that this method is not necessarily static, so a `DistanceType` object
should be held internally and its `Evaluate()` method should be called whenever
the distance between two points is required.  *It is generally a bad idea to
hardcode any distance calculation in your tree.*  This will make the tree unable
to generalize to arbitrary distance metrics.  If your tree must depend on
certain assumptions holding about the distance metric (i.e. the distance metric
is a Euclidean metric), then make that clear in the documentation of the tree,
so users do not try to use the tree with an inappropriate distance metric.

#### `StatisticType`

The second template parameter, `StatisticType`, is for auxiliary information
that is required by certain algorithms.  For instance, consider an algorithm
which repeatedly uses the variance of the descendant points of a node.  It might
be tempting to add a `Variance()` method to the required `TreeType` API, but
this quickly leads to code bloat (after all, the API already has quite enough
functions as it is).  Instead, it is better to create a `StatisticType` class
which provides the `Variance()` method, and then call `Stat().Variance()` when
the variance is required.  This also holds true for cached data members.

Each node should have its own instance of a `StatisticType` class.  The
`StatisticType` must provide the following constructors:

```
// Default constructor required by the StatisticType policy.
StatisticType();

// This constructor is required by the StatisticType policy.
template<typename TreeType>
StatisticType(TreeType& node);
```

This constructor should be called with `(*this)` after the node is constructed
(usually, this ends up being the last line in the constructor of a node).

#### `MatType`

The last template parameter is the `MatType` parameter.  This is generally
`arma::mat` or `arma::sp_mat`, but could be any Armadillo type, including
matrices that hold data points of different precisions (such as `float` or even
`int`).  It generally suffices to write `MatType` assuming that `arma::mat`
will be used, since the vast majority of the time this will be what is used.

### Constructors and destructors

The `TreeType` API requires at least three constructors.  Technically, it does
not *require* a destructor, but almost certainly your tree class will be doing
some memory management internally and should have one (though not always).

The first two constructors are variations of the same idea:

```
// This batch constructor does not modify the dataset, and builds the entire
// tree using a default-constructed DistanceType.
ExampleTree(const MatType& data);

// This batch constructor does not modify the dataset, and builds the entire
// tree using the given DistanceType.
ExampleTree(const MatType& data, DistanceType& distance);
```

All that is required here is that a constructor is available that takes a
dataset and optionally an instantiated distance metric.  If no distance metric
is provided, then it should be assumed that the `DistanceType` class has a
default constructor and a default-constructed distance metric should be used.
The constructor *must* return a valid, fully-constructed, ready-to-use tree that
satisfies the definition of *space tree* that was given earlier in the document.

The third constructor requires the tree to be initializable from a `cereal`
archive:

```
// Initialize the tree from a given cereal archive.  SFINAE (the
// second argument) is necessary to ensure that the archive is loading, not
// saving.
template<typename Archive>
ExampleTree(
    Archive& ar,
    const typename std::enable_if_c<typename Archive::is_loading>::type* = 0);
```

This has implications on how the tree must be stored.  In this case, the dataset
is *not yet loaded* and therefore the tree ***may be required to have
ownership of the data matrix***.  This means that realistically the most
reasonable way to represent the data matrix internally in a tree class is not
with a reference but instead with a pointer.  If this is true, then a destructor
will be required:

```
// Release any resources held by the tree.
~ExampleTree();
```

and, if the data matrix is represented internally with a pointer, this
destructor will need to release the memory for the data matrix (in the case that
the tree was created via `cereal`).

Note that these constructors are not necessarily the only constructors that a
`TreeType` implementation can provide.  One important example of when more
constructors are useful is when the tree rearranges points internally; this
might be desired for the sake of speed or memory optimization.  But to do this
with the required constructors would necessarily incur a copy of the data
matrix, because the user will pass a `const MatType&`.  One alternate solution
is to provide a constructor which takes an rvalue reference to a `MatType`:

```
template<typename Archive>
ExampleTree(MatType&& data);
```

(and another overload that takes an instantiated distance metric), and then the
user can use `std::move()` to build the tree without copying the data matrix,
although the data matrix will be modified:

```
ExampleTree exTree(std::move(dataset));
```

It is, of course, possible to add even more constructors if desired.

### Basic tree functionality

The basic functionality of a class implementing the `TreeType` API is quite
straightforward and intuitive.

```
// Get the dataset that the tree is built on.
const MatType& Dataset();
```

This should return a `const` reference to the dataset the tree is built on.  The
fact that this function is required essentially means that each node in the tree
must store a pointer to the dataset (this is not the only option, but it is the
most obvious option).

```
// Get the distance metric that the tree is built with.
DistanceType& Distance();
```

Each node must also store an instantiated distance metric or a pointer to one
(note that this is required even for metrics that have no state and have a
`static` `Evaluate()` function).

```
// Get/modify the StatisticType for this node.
StatisticType& Stat();
```

As discussed earlier, each node must hold a `StatisticType`; this is accessible
through the `Stat()` function.

```
// Return the parent of the node, or NULL if this is the root.
ExampleTree* Parent();

// Return the number of children held by the node.
size_t NumChildren();
// Return the i'th child held by the node.
ExampleTree& Child(const size_t i);

// Return the number of points held in the node.
size_t NumPoints();
// Return the index of the i'th point held in the node.
size_t Point(const size_t i);

// Return the number of descendant nodes of this node.
size_t NumDescendantNodes();
// Return the i'th descendant node of this node.
ExampleTree& DescendantNode(const size_t i);

// Return the number of descendant points of this node.
size_t NumDescendants();
// Return the index of the i'th descendant point of this node.
size_t Descendant(const size_t i);
```

These functions are all fairly self-explanatory.  Most algorithms will use the
`Parent()`, `Children()`, `NumChildren()`, `Point()`, and `NumPoints()`
functions, so care should be taken when implementing those functions to ensure
they will be efficient.  Note that `Point()` and `Descendant()` should return
indices of points, so the actual points can be accessed by calling
`Dataset().col(Point(i))` for some index `i` (or something similar).

An important note about the `Descendant()` function is that each descendant
point should be unique.  So if a node holds the point with index 6 and it has
one child that holds the points with indices 6 and 7, then `NumDescendants()`
should return 2, not 3.  The ordering in which the descendants are returned can
be arbitrary; so, `Descendant(0)` can return 6 *or* 7, and `Descendant(1)`
should return the other index.

```
// Store the center of the bounding region of the node in the given vector.
void Center(arma::vec& center);
```

The last function, `Center()`, should calculate the center of the bounding shape
and store it in the given vector.  So, for instance, if the tree is a ball tree,
then the center is simply the center of the ball.  Algorithm writers would be
wise to try and avoid the use of `Center()` if possible, since it will
necessarily cost a copy of a vector.

### Complex tree functionality and bounds

A node in a tree should also be able to calculate various distance-related
bounds; these are particularly useful in tree-based algorithms.  Note that any
of these bounds does not necessarily need to be maximally tight; generally it is
more important that each bound can be easily calculated.

Details on each bounding function that the `TreeType` API requires are given
below.

```
// Return the distance between the center of this node and the center of
// its parent.
double ParentDistance();
```

Remember that each node corresponds to some region in the space that the dataset
lies in.  For most tree types this shape is often something geometrically
simple: a ball, a cone, a hyperrectangle, a slice, or something similar.  The
`ParentDistance()` function should return the distance between the center of
this node's region and the center of the parent node's region.

In practice this bound is often used in dual-tree (or single-tree) algorithms to
place an easy `MinDistance()` (or `MaxDistance()`) bound for a child node; the
parent's `MinDistance()` (or `MaxDistance()`) function is called and then
adjusted with `ParentDistance()` to provide a possibly loose but efficient bound
on what the result of `MinDistance()` (or `MaxDistance()`) would be with the
child.

```
// Return an upper bound on the furthest possible distance between the
// center of the node and any point held in the node.
double FurthestPointDistance();

// Return an upper bound on the furthest possible distance between the
// center of the node and any descendant point of the node.
double FurthestDescendantDistance();
```

It is often very useful to be able to bound the radius of a node, which is
effectively what `FurthestDescendantDistance()` does.  Often it is easiest to
simply calculate and cache the furthest descendant distance at tree construction
time.  Some trees, such as the cover tree, are able to give guarantees that the
points held in the node will necessarily be closer than the descendant points;
therefore, the `FurthestPointDistance()` function is also useful.

It is permissible to simply have `FurthestPointDistance()` return the result of
`FurthestDescendantDistance()`, and that will still be a valid bound, but
depending on the type of tree it may be possible to have
`FurthestPointDistance()` return a tighter bound.

```
// Return a lower bound on the minimum distance between the center and any
// edge of the node's bounding shape.
double MinimumBoundDistance();
```

This is, admittedly, a somewhat complex and weird quantity.  It is one of the
less important bounding functions, so it is valid to simply return 0...

The bound is a bound on the minimum distance between the center of the node and
any edge of the shape that bounds all of the descendants of the node.  So, if
the bounding shape is a ball (as in a ball tree or a cover tree), then
`MinimumBoundDistance()` should just return the radius of the ball.  If the
bounding shape is a hypercube (as in a generalized octree), then
`MinimumBoundDistance()` should return the side length divided by two.  If the
bounding shape is a hyperrectangle (as in a kd-tree or a spill tree), then
`MinimumBoundDistance()` should return half the side length of the
hyperrectangle's smallest side.

```
// Return a lower bound on the minimum distance between the given point and
// the node.
template<typename VecType>
double MinDistance(VecType& point);

// Return a lower bound on the minimum distance between the given node and
// this node.
double MinDistance(ExampleTree& otherNode);

// Return an upper bound on the maximum distance between the given point and
// the node.
template<typename VecType>
double MaxDistance(VecType& point);

// Return an upper bound on the maximum distance between the given node and
// this node.
double MaxDistance(ExampleTree& otherNode);

// Return the combined results of MinDistance() and MaxDistance().
template<typename VecType>
Range RangeDistance(VecType& point);

// Return the combined results of MinDistance() and MaxDistance().
Range RangeDistance(ExampleTree& otherNode);
```

These six functions are almost without a doubt the most important functionality
of a tree.  Therefore, it is preferable that these methods be implemented as
efficiently as possible, as they may potentially be called many millions of
times in a tree-based algorithm.  It is also preferable that these bounds be as
tight as possible.  In tree-based algorithms, these are used for pruning away
work, and tighter bounds mean that more pruning is possible.

Of these six functions, there are only really two bounds that are desired here:
the *minimum distance* between a node and an object, and the *maximum distance*
between a node and an object.  The object may be either a vector (usually
`arma::vec`) or another tree node.

Consider the first case, where the object is a vector.  The result of
`MinDistance()` needs to be less than or equal to the true minimum distance,
which could be calculated as below:

```
// We assume that we have a vector 'vec', and a tree node 'node'.
double trueMinDist = DBL_MAX;
for (size_t i = 0; i < node.NumDescendants(); ++i)
{
  const double dist = node.Distance().Evaluate(vec,
      node.Dataset().col(node.Descendant(i)));
  if (dist < trueMinDist)
    trueMinDist = dist;
}
// At the end of the loop, trueMinDist will hold the true minimum distance
// between 'vec' and any descendant point of 'node'.
```

Often the bounding shape of a node will allow a quick calculation that will make
a reasonable bound.  For instance, if the node's bounding shape is a ball with
radius `r` and center `ctr`, the calculation is simply
`(node.Distance().Evaluate(vec, ctr) - r)`.  Usually a good `MinDistance()` or
`MaxDistance()` function will make only one call to the `Evaluate()` function of
the distance metric.

The `RangeDistance()` function allows a way for both bounds to be calculated at
once.  It is possible to implement this as a call to `MinDistance()` followed by
a call to `MaxDistance()`, but this may incur more distance metric `Evaluate()`
calls than necessary.  Often calculating both bounds at once can be more
efficient and can be done with fewer `Evaluate()` calls than calling both
`MinDistance()` and `MaxDistance()`.

### Serialization

The last functions that the `TreeType` API requires are for serialization.

```
// Serialize the tree (load from the given archive / save to the given
// archive, depending on its type).
template<typename Archive>
void serialize(Archive& ar, const unsigned int version);

protected:
// A default constructor; only meant to be used by cereal.  This
// must be protected so that cereal will work; it does not need
// to return a valid tree.
ExampleTree();

// Friend access must be given for the default constructor.
friend class cereal::access;
```

On the other hand, the specifics of the functionality required for the
`serialize()` function are somewhat more difficult.  The `serialize()` function
will be called either when a tree is being saved to disk or loaded from disk.
The `cereal` documentation is fairly comprehensive.

An important note is that it is very difficult to use references with `cereal`,
because `serialize()` may be called at any time during the object's lifetime,
and references cannot be re-seated.  In general this will require the use of
pointers, which then require manual memory management.  Therefore, be careful
that `serialize()` (and the tree's destructor) properly handle memory
management!

## The TreeTraits trait class

Some tree-based algorithms can specialize if the tree fulfills certain
conditions.  For instance, if the regions represented by two sibling nodes
cannot overlap, an algorithm may be able to perform a simpler computation.
Based on this reasoning, the `TreeTraits` trait class (much like the
`KernelTraits` class) exists in order to allow a tree to specify (via a `const
static bool`) when these types of conditions are satisfied.  ***Note that a
TreeTraits class is not required***, but may be helpful.

The `TreeTraits` trait class is a template class that takes a `TreeType` as a
parameter, and exposes `const static bool` values that depend on the tree.
Setting these values is achieved by specialization.  The code below shows the
default `TreeTraits` values (these are the values that will be used if no
specialization is provided for a given `TreeType`).

```
template<typename TreeType>
class TreeTraits
{
 public:
  // This is true if the subspaces represented by the children of a node can
  // overlap.
  static const bool HasOverlappingChildren = true;

  // This is true if Point(0) is the centroid of the node.
  static const bool FirstPointIsCentroid = false;

  // This is true if the points contained in the first child of a node
  // (Child(0)) are also contained in that node.
  static const bool HasSelfChildren = false;

  // This is true if the tree rearranges points in the dataset when it is built.
  static const bool RearrangesDataset = false;

  // This is true if the tree always has only two children.
  static const bool BinaryTree = false;
};
```

An example specialization for the `KDTree` class is given below.  Note that
`KDTree` is itself a template class (like every class satisfying the `TreeType`
policy), so we are specializing to a template parameter.

```
template<typename DistanceType,
         typename StatisticType,
         typename MatType>
template<>
class TreeTraits<KDTree<DistanceType, StatisticType, MatType>>
{
 public:
  // The regions represented by the two children of a node may not overlap.
  static const bool HasOverlappingChildren = false;

  // There is no guarantee that the first point of a node is the centroid.
  static const bool FirstPointIsCentroid = false;

  // Points are not contained at multiple levels (only at the leaves).
  static const bool HasSelfChildren = false;

  // Points are rearranged during the building of the tree.
  static const bool RearrangesDataset = true;

  // The tree is always binary.
  static const bool BinaryTree = true;
};
```

Currently, the traits available are each of the five detailed above.  For more
information, see the `TreeTraits` source code for more documentation.

## Traversals

mlpack's tree-based algorithms are all implemented with *traversals*.  Either a
single-tree traversal (which visits nodes in an individual tree) or a dual-tree
traversal (which visits pairs of nodes from two different trees) can be used.

Each tree class is equipped with a `SingleTreeTraverser` and `DualTreeTraverser`
class, with the following signatures:

```
template<typename RuleType>
class SingleTreeTraverser;

template<typename RuleType>
class DualTreeTraverser;
```

The `RuleType` template parameter is a class that defines the action to be taken
at each step of the traversal; this is detailed in [the next section](#rules).

---

The `SingleTreeTraverser` for each tree implements two functions, matching the
API below:

```
template<typename RuleType>
class SingleTreeTraverser
{
 public:
  // Create the SingleTreeTraverser with the given rules.
  SingleTreeTraverser(RuleType& rule);

  // Perform a full traversal of the tree, with the given query index.
  // The RuleType's implementation will be given the query index during the
  // traversal, and this class does not use `queryIndex` directly.
  //
  // Here `TreeType` is the type of the tree that is being used.  Generally
  // `referenceNode` should refer to the root of the tree being traversed, but
  // it does not have to.
  void Traverse(const size_t queryIndex, TreeType& referenceNode);
};
```

---

The `DualTreeTraverser` for each tree also implements two similar functions,
matching the API below:

```
template<typename RuleType>
class DualTreeTraverser
{
 public:
  // Create the DualTreeTraverser with the given rules.
  DualTreeTraverser(RuleType& rule);

  // Perform a full dual-tree traversal, visiting combinations of nodes from the
  // query tree and the reference tree.  Here `TreeType` is the type of the tree
  // that is being used.  Generally, `queryNode` and `referenceNode` should reer
  // to the roots of the trees being traversed, but it does not have to.
  //
  // It is ok if `queryNode` and `referenceNode` refer to the same tree.
  void Traverse(TreeType& queryNode, TreeType& referenceNode);
};
```

---

In essence, the `SingleTreeTraverser` and `DualTreeTraverser` classes are
relatively simple classes that define the order of traversal of each node in the
tree.  For a simple example, the [`KDTree`](../user/core/trees/kdtree.md)
class's `SingleTreeTraverser` is a simple depth-first traversal.  Suppose that
we have the `KDTree` drawn below as ASCII art:

```text
       0
      / \
     /   \
    /     \
   /       \
  1         2
 / \       / \
3   4     5   6
   / \   / \
  7   8 9   10
```

This `KDTree` has 10 nodes total; for a simple depth-first traversal, we will
visit nodes in this order:

```text
0, 1, 3, 4, 7, 8, 2, 5, 9, 10, 6
```

Dual-tree traversals are more complicated, as we are visiting combinations of
nodes from two different trees.  Consider two even simpler `KDTree`s drawn below
as ASCII art:

```text
  query tree                       reference tree

       q0                                 r0
      /  \                               /  \
     /    \                             /    \
    /      \                           /      \
   q1      q2                         r1      r2
  /  \    /  \                       /  \    /  \
q3    q4 q5   q6                   r3    r4 r5   r6
```

The default `KDTree` dual-tree traversal is a dual depth-first traversal.  This
means that the order of visitation is:

```
// Indentation shows the level of recursion.
(q0, r0),
    (q1, r1),
        (q3, r3),
        (q3, r4),
        (q4, r3),
        (q4, r4),
    (q1, r2),
        (q3, r5),
        (q3, r6),
        (q4, r5),
        (q4, r6),
    (q2, r1),
        (q5, r3),
        (q5, r4),
        (q6, r3),
        (q6, r4),
    (q2, r2),
        (q5, r5),
        (q5, r6),
        (q6, r5),
        (q6, r6)
```

## Rules

The third part of a tree-based algorithm are the rules for when nodes in the
[traversal](#traversals) can be pruned.  This is where the speedup comes from:
consider, for instance, the dual-tree traversal example directly above.  If we
*pruned* the node combination `(q1, r1)`, we could then simply skip the child
node combinations `(q3, r3)`, `(q3, r4)`, `(q4, r3)`, and `(q4, r4)`.  In the
single-tree example above, if we could prune the node `1`, then we would not
even need to visit the nodes `3`, `4`, `7`, or `8`.

Pruning is the key to acceleration, and good pruning rules allow highly
accelerated algorithms because they can simply skip most of the nodes (or node
combinations) in a tree traversal.

mlpack's traversers take a `RuleType` template parameter.  The `RuleType`
defines the rules of the traversal, telling the traversal when to prune nodes,
and performing the actual work to be done during the traversal.  Any `RuleType`
must implement the following class signature:

```
//
// A RuleType that can be used with a SingleTreeTraverser or DualTreeTraverser.
// In the example code below, `TreeType` refers to the type fo tree being used.
// This could be hard-coded to a specific tree type, or could be a template
// parameter to the `Rules` class, or could be a template parameter to the
// functions themselves---any of these three options will work with mlpack's
// traversers.
//
class Rules
{
 public:
  // Any RuleType must implement the base case: this is the operation to be done
  // between two points held in tree nodes.  `queryIndex` is the index of the
  // point in the query tree, and `referenceIndex` is the index of the point in
  // the reference tree.  This should return a `double` representing the base
  // case result (typically the result of the operation between two points).
  double BaseCase(const size_t queryIndex, const size_t referenceIndex);

  //
  // Functions required by a SingleTreeTraverser.
  //

  // Determine whether to prune the node `referenceNode`.  If so, return
  // `DBL_MAX`.  Otherwise, return a numeric score indicating how "promising"
  // the node is (lower scores are better).
  double Score(const size_t queryIndex,
               TreeType& referenceNode);

  // Check again if `referenceNode` can be pruned, returning `DBL_MAX` if so.
  // This can be useful as bounding quantities can change during traversal and
  // may have been updated since `Score()` was called.
  double Rescore(const size_t queryIndex,
                 TreeType& referenceNode,
                 const double oldScore);

  //
  // Functions required by a DualTreeTraverser.
  //

  // Determine whether to prune the node combination
  // `(queryNode, referenceNode)`.  If so, return `DBL_MAX`.  Otherwise, return
  // a numeric score indicating how "promising" the node combination is (lower
  // scores are better).
  double Score(TreeType& queryNode,
               TreeType& referenceNode);

  // Check again if the combination `(queryNode, referenceNode)` can be pruned,
  // returning `DBL_MAX` if so.  This can be useful as bounding quantities can
  // change during traversal and may have been updated since `Score()` was
  // called.
  double Rescore(TreeType& queryNode,
                 TreeType& referenceNode,
                 const double oldScore);

  // The traversal information is auxiliary information held by the dual-tree
  // traverser and can be used to cache quantities during the traversal.  More
  // details in the section below the example.  If no traversal information is
  // needed for the functions above, you can use an empty class here:
  //
  //    class TraversalInfoType {};
  //
  // This class can be omitted when only single-tree traversers will be used.
  class TraversalInfoType; // This can also be a typedef to an external class.

  // Access or modify the auxiliary traversal information.
  // This will be used by dual-tree traversers and can be omitted when only
  // single-tree traversers will be used.
  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

 private:
  TraversalInfoType traversalInfo;

  // In general, a Rules class must also hold a query and reference dataset (or
  // references to datasets), as it will receive `size_t` indexes like
  // `queryIndex` and `referenceIndex`, and these generally need to be mapped
  // back to the data point they represent.
};
```

In essence, the `RuleType` classes define the following behaviors:

 * what to do with two points
 * whether we can prune a node in single-tree mode
 * whether we can prune a node combination in dual-tree mode

### An example `RuleType` class

To better understand these functions, let us design a very simple `RuleType`
class that collects all points that have distance less than 1 from each other.

```c++
// Here we have chosen to templatize the `TreeType` at the class level.
// This set of rules will simply print out whenever two points have a distance
// of less than 1.
template<typename TreeType>
class LessThanOneDistRules
{
 public:
  // The constructor accepts a query dataset and a reference dataset.
  LessThanOneDistRules(const arma::mat& querySet,
                       const arma::mat& referenceSet) :
      querySet(querySet), referenceSet(referenceSet) { }

  // The base case compares whether two points have a distance less than one,
  // and prints to the screen if so.
  double BaseCase(const size_t queryIndex, const size_t referenceIndex)
  {
    // Compute the distance between the two points.
    const double dist = mlpack::EuclideanDistance::Evaluate(
        querySet.col(queryIndex), referenceSet.col(referenceIndex));
    if (dist <= 1.0)
    {
      std::cout << " - Query point " << queryIndex << " and reference point "
          << referenceIndex << " have distance " << dist << "!" << std::endl;
    }

    return dist; // Return the base case value; it is used by some traversers.
  }

  // The single-tree Score() function computes whether the query point could
  // have distance less than 1 to any descendant points of the reference node.
  double Score(const size_t queryIndex, TreeType& referenceNode)
  {
    const double minDist = referenceNode.MinDistance(querySet.col(queryIndex));

    // Return DBL_MAX if minDist is greater than 1, because then a descendant
    // point of referenceNode cannot have distance less than 1 to the query
    // point.
    return (minDist <= 1.0) ? minDist : DBL_MAX;
  }

  // In this simple example, we are not computing any bounds, so Rescore() will
  // not be able to prune anything new.  Therefore we just return the old score.
  double Rescore(const size_t queryIndex,
                 TreeType& referenceNode,
                 const double oldScore)
  {
    return oldScore;
  }

  // The dual-tree Score() function is the same as the single-tree version, but
  // generalized to node combinations.
  double Score(TreeType& queryNode, TreeType& referenceNode)
  {
    const double minDist = referenceNode.MinDistance(queryNode);

    // Return DBL_MAX if minDist is greater than 1, because then a descendant
    // point of referenceNode cannot have distance less than 1 to any descendant
    // point of queryNode.
    return (minDist <= 1.0) ? minDist : DBL_MAX;
  }

  // In this simple example, we are not computing any bounds, so Rescore() will
  // not be able to prune anything new.  Therefore we just return the old score.
  double Rescore(TreeType& queryNode,
                 TreeType& referenceNode,
                 const double oldScore)
  {
    return oldScore;
  }

  // We are not using any traversal info, so define it as an empty class.
  class TraversalInfoType { };

  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

 private:
  const arma::mat& querySet;
  const arma::mat& referenceSet;

  TraversalInfoType traversalInfo;
};
```

We can now put this rule set into action, using
[`KDTree`](../user/core/trees/kdtree.md)s as the tree type.

```c++
// Create random Gaussian-distributed sets with 50 points in 3 dimensions, and
// moderate covariance (so that not many point pairs have distance less than 1).
arma::mat querySet     = 1.75 * arma::randn<arma::mat>(3, 50);
arma::mat referenceSet = 1.75 * arma::randn<arma::mat>(3, 50);

// Create KDTrees on the data, using a minimum leaf size of 5.
//
// (Note that KDTrees rearrange points in the dataset, so the printed indices
// from BaseCase() are not the same as what they would be if we computed
// distances in a double for-loop on querySet and referenceSet; see the KDTree
// documentation for more details.)
mlpack::KDTree<> queryTree(std::move(querySet), 5);
mlpack::KDTree<> referenceTree(std::move(referenceSet), 5);

// Construct a RuleType.
LessThanOneDistRules<mlpack::KDTree<>> rule(queryTree.Dataset(),
                                            referenceTree.Dataset());

// Now perform a single-tree traversal for point 0 (this will find all reference
// points with distance less than 1 to query point 0).
std::cout << "Single-tree traversal for query point 0:" << std::endl;
mlpack::KDTree<>::SingleTreeTraverser st(rule);
st.Traverse(0, referenceTree);
std::cout << std::endl;

// Now perform a dual-tree traversal (this will find all pairs of points in the
// query set and reference set with distance less than 1).
std::cout << "Dual-tree traversal:" << std::endl;
mlpack::KDTree<>::DualTreeTraverser dt(rule);
dt.Traverse(queryTree, referenceTree);
```

Of course, this example is very simple.  Far more complex sets of rules are
possible.  For a step up in complexity,
[`RangeSearchRules`](/src/mlpack/methods/range_search/range_search_rules_impl.hpp)
is similar to this example.  Even more complex is the
[`NeighborSearchRules`](/src/mlpack/methods/neighbor_search/neighbor_search_rules_impl.hpp);
that set of rules maintains bounds in a custom `StatisticType` and uses these
bounds in `Score()` and `Rescore()`.

### `TraversalInfoType`

For dual-tree traversals, a `RuleType` must be equipped with a
`TraversalInfoType` class.  This `TraversalInfoType` class is a way to store
auxiliary bounding information.  Any mlpack traversal will use the
`TraversalInfo()` accessor before calling `BaseCase()`, `Score()`, or
`Rescore()` to reset the `TraversalInfoType`'s state to its state when the
parent combination was called.

So, for instance, consider the following dual-tree recursion order (from the
example in the [traversals section](#traversals)):

```text
// Indentation levels indicate the level of recursion.
(q0, r0),
    (q1, r1),
        (q3, r3),
        (q3, r4),
        (q4, r3),
        (q4, r4),
...
```

When visiting the combination `(q3, r3)`, the traversal info object will have
the same state as just after `Score()` was called with the combination `(q1,
r1)`.  When `Score()` on `(q3, r3)` (and any `BaseCase()`s) are called, this
traversal info object may be modified---but before the traversal calls `Score()`
on the combination `(q3, r4)`, the traversal info object will have its state
reset to the state it had just after `Score()` was called for `(q1, r1)`.

A simple `TraversalInfoType` is available with the
[`TraversalInfo`](/src/mlpack/core/tree/traversal_info.hpp) class, which
contains members to track the last query and reference nodes, last `Score()`
function result, and last base case result.  Setting these members is not done
automatically, and must be done in the `RuleType`'s `BaseCase()`, `Score()`, and
`Rescore()` functions.

## A list of trees in mlpack and more information

mlpack contains several ready-to-use implementations of trees that satisfy the
TreeType policy API:

 - [`KDTree`](../user/core/trees/kdtree.md)
 - [`MeanSplitKDTree`](../user/core/trees/mean_split_kdtree.md)
 - [`BallTree`](../user/core/trees/ball_tree.md)
 - [`MeanSplitBallTree`](../user/core/trees/mean_split_ball_tree.md)
 - `RTree`
 - `RStarTree`
 - `StandardCoverTree`

Often, these are template typedefs of more flexible tree classes:

 - [`BinarySpaceTree`](../user/core/trees/binary_space_tree.md) -- binary trees,
   such as the KD-tree and ball tree
 - `RectangleTree` -- the R tree and variants
 - `CoverTree` -- the cover tree and variants
