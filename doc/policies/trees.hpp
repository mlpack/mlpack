/*! @page trees The TreeType policy in mlpack

@section treeintro Introduction

Trees are an important data structure in mlpack and are used in a number of the
machine learning algorithms that mlpack implements.  Often, the use of trees can
allow significant acceleration of an algorithm; this is generally done by
pruning away large parts of the tree during computation.

Most mlpack algorithms that use trees are not tied to a specific tree but
instead allow the user to choose a tree via the \c TreeType template parameter.
Any tree passed as a \c TreeType template parameter will need to implement a
certain set of functions.  In addition, a tree may optionally specify some
traits about itself with the \c TreeTraits trait class.

This document aims to clarify the abstractions underlying mlpack trees, list and
describe the required functionality of the \c TreeType policy, and point users
towards existing types of trees.  A table of contents is below:

 - \ref treeintro
 - \ref whatistree
 - \ref treetype_template_params
 - \ref treetype_api
 - \ref treetype_rigorous
   - \ref treetype_rigorous_template
   - \ref treetype_rigorous_constructor
   - \ref treetype_rigorous_basic
   - \ref treetype_rigorous_complex
   - \ref treetype_rigorous_serialization
 - \ref treetype_traits
 - \ref treetype_more

Although this document is long, there may still be errors and unclear areas.  If
you are having trouble understanding anything, please get in touch on Github or
on the mailing list and someone will help you (and possibly update the
documentation afterwards).

@section whatistree What is a tree?

In mlpack, we assume that we have some sort of data matrix, which might be
sparse or dense (that is, it could be of type \c arma::mat or \c arma::sp_mat,
or any variant that implements the Armadillo API).  This data matrix corresponds
to a collection of points in some space (usually a Euclidean space).  A tree is
a way of organizing this data matrix in a hierarchical manner---so, points that
are nearby should lie in similar nodes.

We can rigorously define what a tree is, using the definition of **space tree**
introduced in the following paper:

@quote
R.R. Curtin, W.B. March, P. Ram, D.V. Anderson, A.G. Gray, and C.L. Isbell Jr.,
"Tree-independent dual-tree algorithms," in Proceedings of the 30th
International Conference on Machine Learning (ICML '13), pp. 1435--1443, 2013.
@endquote

The definition is:

A **space tree** on a dataset \f$ S \in \mathcal{R}^{N \times d} \f$ is an
undirected, connected, acyclic, rooted simple graph with the following
properties:

 - Each node (or vertex) holds a number of points (possibly zero) and is
connected to one parent node and a number of child nodes (possibly zero).

 - There is one node in every space tree with no parent; this is the root node
of the tree.

 - Each point in \f$S\f$ is contained in at least one node.

 - Each node corresponds to some subset of \f$\mathcal{R}^d\f$ that contains
each point in the node and also the subsets that correspond to each child of the
node.

This is really a quite straightforward definition: a tree is hierarchical, and
each node corresponds to some region of the input space.  Each node may have
some number of children, and may hold some number of points.  However, there is
an important terminology distinction to make: the term **points held by a node**
has a different meaning than the term **descendant points held by a node**.  The
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
another example, the \f$kd\f$-tree holds its points only in the leaves (at the
bottom of the tree).  More information on space trees can be found in either the
"Tree-independent dual-tree algorithms" paper or any of the related literature.

So there is a huge amount of possible variety in the types of trees that can
fall into the class of *space trees*.  Therefore, it's important to treat them
abstractly, and the \c TreeType policy allows us to do just that.  All we need
to remember is that a node in a tree can be represented as the combination of
some points held in the node, some child nodes, and some geometric structure
that represents the space that all of the descendant points fall into (this is a
restatement of the fourth part of the definition).

@section treetype_template_params Template parameters required by the TreeType policy

Most everything in mlpack is decomposed into a series of configurable template
parameters, and trees are no exception.  In order to ease usage of high-level
mlpack algorithms, each \c TreeType itself must be a template class taking three
parameters:

 - \c MetricType -- the underlying metric that the tree will be built on (see
\ref metrics "the MetricType policy documentation")
 - \c StatisticType -- holds any auxiliary information that individual
algorithms may need
 - \c MatType -- the type of the matrix used to represent the data

The reason that these three template parameters are necessary is so that each
\c TreeType can be used as a template template parameter, which can radically
simplify the required syntax for instantiating mlpack algorithms.  By using
template template parameters, a user needs only to write

@code
// The RangeSearch class takes a MetricType and a TreeType template parameter.

// This code instantiates RangeSearch with the ManhattanDistance and a
// QuadTree.  Note that the QuadTree itself is a template, and takes a
// MetricType, StatisticType, and MatType, just like the policy requires.

// This example ignores the constructor parameters, for the sake of simplicity.
RangeSearch<ManhattanDistance, QuadTree> rs(...);
@endcode

as opposed to the far more complicated alternative, where the user must specify
the values of each template parameter of the tree type:

@code
// This is a much worse alternative, where the user must specify the template
// arguments of their tree.
RangeSearch<ManhattanDistance,
            QuadTree<ManhattanDistance, EmptyStatistic, arma::mat>> rs(...);
@endcode

Unfortunately, the price to pay for this user convenience is that *every*
\c TreeType must have three template parameters, and they must be in exactly
that order.  Fortunately, there is an additional benefit: we are guaranteed that
the tree is built using the same metric as the method (that is, a user can't
specify different metric types to the algorithm and to the tree, which they can
without template template parameters).

There are two important notes about this:

 - Not every possible input of MetricType, StatisticType, and/or MatType
necessarily need to be valid or work correctly for each type of tree.  For
instance, the QuadTree is limited to Euclidean metrics and will not work
otherwise.  Either compile-time static checks or detailed documentation can help
keep users from using invalid combinations of template arguments.

 - Some types of trees have more template parameters than just these three.  One
example is the generalized binary space tree, where the bounding shape of each
node is easily made into a fourth template parameter (the \c BinarySpaceTree
class calls this the \c BoundType parameter), and the procedure used to split a
node is easily made into a fifth template parameter (the \c BinarySpaceTree
class calls this the \c SplitType parameter).  However, the syntax of template
template parameters *requires* that the class only has the correct number of
template parameters---no more, no less.  Fortunately, C++11 allows template
typedefs, which can be used to provide partial specialization of template
classes:

@code
// This is the definition of the BinarySpaceTree class, which has five template
// parameters.
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         typename BoundType,
         typename SplitType>
class BinarySpaceTree;

// The 'using' keyword gives us a template typedef, so we can define the
// MeanSplitKDTree template class, which has three parameters and is a valid
// TreeType policy class.
template<typename MetricType, typename StatisticType, typename MatType>
using MeanSplitKDTree = BinarySpaceTree<MetricType,
                                        StatisticType,
                                        MatType,
                                        HRectBound<MetricType>
                                        MeanSplit<BoundType, MetricType>>;
@endcode

Now, the \c MeanSplitKDTree class has only three template parameters and can be
used as a \c TreeType policy class in various mlpack algorithms.  Many types of
trees in mlpack have more than three template parameters and rely on template
typedefs to provide simplified \c TreeType interfaces.

@section treetype_api The TreeType API

As a result of the definition of *space tree* in the previous section, a
simplified API presents itself quite easily.  However, more complex
functionality is often necessary in mlpack, so this leads to more functions
being necessary for a class to satisfy the \c TreeType policy.  Combining this
with the template parameters required for trees given in the previous section
gives us the complete API required for a class implementing the \c TreeType
policy.  Below is the minimal set of functions required with minor
documentation for each function.  (More extensive documentation and explanation
is given afterwards.)

@code
// The three template parameters will be supplied by the user, and are detailed
// in the previous section.
template<typename MetricType,
         typename StatisticType,
         typename MatType>
class ExampleTree
{
 public:
  //////////////////////
  //// Constructors ////
  //////////////////////

  // This batch constructor does not modify the dataset, and builds the entire
  // tree using a default-constructed MetricType.
  ExampleTree(const MatType& data);

  // This batch constructor does not modify the dataset, and builds the entire
  // tree using the given MetricType.
  ExampleTree(const MatType& data, MetricType& metric);

  // Initialize the tree from a given boost::serialization archive.  SFINAE (the
  // second argument) is necessary to ensure that the archive is loading, not
  // saving.
  template<typename Archive>
  ExampleTree(
      Archive& ar,
      const typename boost::enable_if<typename Archive::is_loading>::type* = 0);

  // Release any resources held by the tree.
  ~ExampleTree();

  // ///////////////////////// //
  // // Basic functionality // //
  // ///////////////////////// //

  // Get the dataset that the tree is built on.
  const MatType& Dataset();

  // Get the metric that the tree is built with.
  MetricType& Metric();

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
  double ParentDistance();

  // Return an upper bound on the furthest possible distance between the
  // center of the node and any point held in the node.
  double FurthestPointDistance();

  // Return an upper bound on the furthest possible distance between the
  // center of the node and any descendant point of the node.
  double FurthestDescendantDistance();

  // Return a lower bound on the minimum distance between the center and any
  // edge of the node's bounding shape.
  double MinimumBoundDistance();

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
  math::Range RangeDistance(VecType& point);

  // Return the combined results of MinDistance() and MaxDistance().
  math::Range RangeDistance(ExampleTree& otherNode);

  // //////////////////////////////////// //
  // // Serialization (loading/saving) // //
  // //////////////////////////////////// //

  // Return a string representation of the tree.
  std::string ToString() const;

  // Serialize the tree (load from the given archive / save to the given
  // archive, depending on its type).
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int version);

 protected:
  // A default constructor; only meant to be used by boost::serialization.  This
  // must be protected so that boost::serialization will work; it does not need
  // to return a valid tree.
  ExampleTree();

  // Friend access must be given for the default constructor.
  friend class boost::serialization::access;
};
@endcode

Although this is significantly more complex than the four-item definition of
*space tree* might suggest, it turns out many of these methods are not
difficult to implement for most reasonable tree types.  It is also important to
realize that this is a *minimum* API; you may implement more complex tree types
at your leisure (and you may include more template parameters too, though you
will have to use template typedefs to provide versions with three parameters;
see \ref treetype_template_params "the previous section").

Before diving into the detailed documentation for each function, let us consider
a few important points about the implications of this API:

 - **Trees are not default-constructible** and should not (in general) provide
a default constructor.  This helps prevent invalid trees.  In general, any
instantiated mlpack object should be valid and ready to use---and a tree built
on no points is not valid or ready to use.

 - **Trees only need to provide batch constructors.**  Although many tree types
do have algorithms for incremental insertions, in mlpack this is not required
because the tree-based algorithms that mlpack implements generally assume
fully-built, non-modifiable trees.  For this purpose, batch construction is
perfectly sufficient.  (It's also worth pointing out that for some types of
trees, like kd-trees, the cost of a handful of insertions often outweighs the
cost of completely rebuilding the tree.)

 - **Trees must provide a number of distance bounding functions.**  The utility
of trees generally stems from the ability to place quick bounds on
distance-related quantities.  For instance, if all the descendant points of a
node are bounded by a ball of radius \f$\lambda\f$ and the center of the node
is a point \f$c\f$, then the minimum distance between some point \f$p\f$ and any
descendant point of the node is equal to the distance between \f$p\f$ and
\f$c\f$ minus the radius \f$\lambda\f$: \f$d(p, c) - \lambda\f$.  This is a fast
calculation, and (usually) provides a decent bound on the minimum distance
between \f$p\f$ and any descendant point of the node.

 - **Trees need to be able to be serialized.**  mlpack uses the
boost::serialization library for saving and loading objects.  Trees---which can
be a part of machine learning models---therefore must have the ability to be
saved and loaded.  Making this all work requires a protected constructor (part
of the API) and generally makes it impossible to hold references instead of
pointers internally, because if a tree is loaded from a file then it must own
the dataset it is built on and the metric it uses (this also means that a
destructor must exist for freeing these resources).

Now, we can consider each part of the API more rigorously.

@section treetype_rigorous Rigorous API documentation

This section is divided into five parts:

 - \ref treetype_rigorous_template
 - \ref treetype_rigorous_constructor
 - \ref treetype_rigorous_basic
 - \ref treetype_rigorous_complex
 - \ref treetype_rigorous_serialization

@subsection treetype_rigorous_template Template parameters

\ref treetype_template_param "An earlier section" discussed the three different
template parameters that are required by the \c TreeType policy.

The \ref metrics "MetricType policy" provides one method that will be useful for
tree building and other operations:

@code
// This function is required by the MetricType policy.
// Evaluate the metric between two points (which may be of different types).
template<typename VecTypeA, typename VecTypeB>
double Evaluate(const VecTypeA& a, const VecTypeB& b);
@endcode

Note that this method is not necessarily static, so a \c MetricType object
should be held internally and its \c Evaluate() method should be called whenever
the distance between two points is required.  **It is generally a bad idea to
hardcode any distance calculation in your tree.**  This will make the tree
unable to generalize to arbitrary metrics.  If your tree must depend on certain
assumptions holding about the metric (i.e. the metric is a Euclidean metric),
then make that clear in the documentation of the tree, so users do not try to
use the tree with an inappropriate metric.

The second template parameter, \c StatisticType, is for auxiliary information
that is required by certain algorithms.  For instance, consider an algorithm
which repeatedly uses the variance of the descendant points of a node.  It might
be tempting to add a \c Variance() method to the required \c TreeType API, but
this quickly leads to code bloat (after all, the API already has quite enough
functions as it is).  Instead, it is better to create a \c StatisticType class
which provides the \c Variance() method, and then call \c Stat().Variance() when
the variance is required.  This also holds true for cached data members.

Each node should have its own instance of a \c StatisticType class.  The
\c StatisticType must provide the following constructor:

@code
// This constructor is required by the StatisticType policy.
template<typename TreeType>
StatisticType(TreeType& node);
@endcode

This constructor should be called with \c (*this) after the node is constructed
(usually, this ends up being the last line in the constructor of a node).

The last template parameter is the \c MatType parameter.  This is generally
\c arma::mat or \c arma::sp_mat, but could be any Armadillo type, including
matrices that hold data points of different precisions (such as \c float or even
\c int).  It generally suffices to write \c MatType assuming that \c arma::mat
will be used, since the vast majority of the time this will be what is used.

@subsection treetype_rigorous_constructor Constructors and destructors

The \c TreeType API requires at least three constructors.  Technically, it does
not *require* a destructor, but almost certainly your tree class will be doing
some memory management internally and should have one (though not always).

The first two constructors are variations of the same idea:

@code
// This batch constructor does not modify the dataset, and builds the entire
// tree using a default-constructed MetricType.
ExampleTree(const MatType& data);

// This batch constructor does not modify the dataset, and builds the entire
// tree using the given MetricType.
ExampleTree(const MatType& data, MetricType& metric);
@endcode

All that is required here is that a constructor is available that takes a
dataset and optionally an instantiated metric.  If no metric is provided, then
it should be assumed that the \c MetricType class has a default constructor and
a default-constructed metric should be used.  The constructor *must* return a
valid, fully-constructed, ready-to-use tree that satisfies the definition
of *space tree* that was \ref whatistree "given earlier".

It is possible to implement both these constructors as one by using \c
boost::optional.

The third constructor requires the tree to be initializable from a \c
boost::serialization archive:

@code
// Initialize the tree from a given boost::serialization archive.  SFINAE (the
// second argument) is necessary to ensure that the archive is loading, not
// saving.
template<typename Archive>
ExampleTree(
    Archive& ar,
    const typename boost::enable_if<typename Archive::is_loading>::type* = 0);
@endcode

This has implications on how the tree must be stored.  In this case, the dataset
is *not yet loaded* and therefore the tree **may be required to have
ownership of the data matrix**.  This means that realistically the most
reasonable way to represent the data matrix internally in a tree class is not
with a reference but instead with a pointer.  If this is true, then a destructor
will be required:

@code
// Release any resources held by the tree.
~ExampleTree();
@endcode

and, if the data matrix is represented internally with a pointer, this
destructor will need to release the memory for the data matrix (in the case that
the tree was created via \c boost::serialization ).

Note that these constructors are not necessarily the only constructors that a
\c TreeType implementation can provide.  One important example of when more
constructors are useful is when the tree rearranges points internally; this
might be desired for the sake of speed or memory optimization.  But to do this
with the required constructors would necessarily incur a copy of the data
matrix, because the user will pass a \c "const MatType&".  One alternate
solution is to provide a constructor which takes an rvalue reference to a
\c MatType:

@code
template<typename Archive>
ExampleTree(MatType&& data);
@endcode

(and another overload that takes an instantiated metric), and then the user can
use \c std::move() to build the tree without copying the data matrix, although
the data matrix will be modified:

@code
ExampleTree exTree(std::move(dataset));
@endcode

It is, of course, possible to add even more constructors if desired.

@subsection treetype_rigorous_basic Basic tree functionality

The basic functionality of a class implementing the \c TreeType API is quite
straightforward and intuitive.

@code
// Get the dataset that the tree is built on.
const MatType& Dataset();
@endcode

This should return a \c const reference to the dataset the tree is built on.
The fact that this function is required essentially means that each node in the
tree must store a pointer to the dataset (this is not the only option, but it is
the most obvious option).

@code
// Get the metric that the tree is built with.
MetricType& Metric();
@endcode

Each node must also store an instantiated metric or a pointer to one (note that
this is required even for metrics that have no state and have a \c static \c
Evaluate() function).

@code
// Get/modify the StatisticType for this node.
StatisticType& Stat();
@endcode

As discussed earlier, each node must hold a \c StatisticType; this is accessible
through the \c Stat() function.

@code
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
@endcode

These functions are all fairly self-explanatory.  Most algorithms will use the
\c Parent(), \c Children(), \c NumChildren(), \c Point(), and \c NumPoints()
functions, so care should be taken when implementing those functions to ensure
they will be efficient.  Note that \c Point() and \c Descendant() should return
indices of points, so the actual points can be accessed by calling
\c "Dataset().col(Point(i))" for some index \c i (or something similar).

An important note about the \c Descendant() function is that each descendant
point should be unique.  So if a node holds the point with index 6 and it has
one child that holds the points with indices 6 and 7, then \c NumDescendants()
should return 2, not 3.  The ordering in which the descendants are returned can
be arbitrary; so, \c Descendant(0) can return 6 \b or 7, and \c Descendant(1)
should return the other index.

@code
// Store the center of the bounding region of the node in the given vector.
void Center(arma::vec& center);
@endcode

The last function, \c Center(), should calculate the center of the bounding
shape and store it in the given vector.  So, for instance, if the tree is a ball
tree, then the center is simply the center of the ball.  Algorithm writers would
be wise to try and avoid the use of \c Center() if possible, since it will
necessarily cost a copy of a vector.

@subsection treetype_rigorous_complex Complex tree functionality and bounds

A node in a tree should also be able to calculate various distance-related
bounds; these are particularly useful in tree-based algorithms.  Note that any
of these bounds does not necessarily need to be maximally tight; generally it is
more important that each bound can be easily calculated.

Details on each bounding function that the \c TreeType API requires are given
below.

@code
// Return the distance between the center of this node and the center of
// its parent.
double ParentDistance();
@endcode

Remember that each node corresponds to some region in the space that the dataset
lies in.  For most tree types this shape is often something geometrically
simple: a ball, a cone, a hyperrectangle, a slice, or something similar.  The
\c ParentDistance() function should return the distance between the center of
this node's region and the center of the parent node's region.

In practice this bound is often used in dual-tree (or single-tree) algorithms to
place an easy \c MinDistance() (or \c MaxDistance() ) bound for a child node;
the parent's \c MinDistance() (or \c MaxDistance() ) function is called and then
adjusted with \c ParentDistance() to provide a possibly loose but efficient
bound on what the result of \c MinDistance() (or \c MaxDistance() ) would be
with the child.

@code
// Return an upper bound on the furthest possible distance between the
// center of the node and any point held in the node.
double FurthestPointDistance();

// Return an upper bound on the furthest possible distance between the
// center of the node and any descendant point of the node.
double FurthestDescendantDistance();
@endcode

It is often very useful to be able to bound the radius of a node, which is
effectively what \c FurthestDescendantDistance() does.  Often it is easiest to
simply calculate and cache the furthest descendant distance at tree construction
time.  Some trees, such as the cover tree, are able to give guarantees that the
points held in the node will necessarily be closer than the descendant points;
therefore, the \c FurthestPointDistance() function is also useful.

It is permissible to simply have \c FurthestPointDistance() return the result of
\c FurthestDescendantDistance(), and that will still be a valid bound, but
depending on the type of tree it may be possible to have \c
FurthestPointDistance() return a tighter bound.

@code
// Return a lower bound on the minimum distance between the center and any
// edge of the node's bounding shape.
double MinimumBoundDistance();
@endcode

This is, admittedly, a somewhat complex and weird quantity.  It is one of the
less important bounding functions, so it is valid to simply return 0...

The bound is a bound on the minimum distance between the center of the node and
any edge of the shape that bounds all of the descendants of the node.  So, if
the bounding shape is a ball (as in a ball tree or a cover tree), then
\c MinimumBoundDistance() should just return the radius of the ball.  If the
bounding shape is a hypercube (as in a generalized octree), then
\c MinimumBoundDistance() should return the side length divided by two.  If the
bounding shape is a hyperrectangle (as in a kd-tree or a spill tree), then
\c MinimumBoundDistance() should return half the side length of the
hyperrectangle's smallest side.

@code
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
math::Range RangeDistance(VecType& point);

// Return the combined results of MinDistance() and MaxDistance().
math::Range RangeDistance(ExampleTree& otherNode);
@endcode

These six functions are almost without a doubt the most important functionality
of a tree.  Therefore, it is preferable that these methods be implemented as
efficiently as possible, as they may potentially be called many millions of
times in a tree-based algorithm.  It is also preferable that these bounds be as
tight as possible.  In tree-based algorithms, these are used for pruning away
work, and tighter bounds mean that more pruning is possible.

Of these six functions, there are only really two bounds that are desired here:
the *minimum distance* between a node and an object, and the *maximum distance*
between a node and an object.  The object may be either a vector (usually \c
arma::vec ) or another tree node.

Consider the first case, where the object is a vector.  The result of
\c MinDistance() needs to be less than or equal to the true minimum distance,
which could be calculated as below:

@code
// We assume that we have a vector 'vec', and a tree node 'node'.
double trueMinDist = DBL_MAX;
for (size_t i = 0; i < node.NumDescendants(); ++i)
{
  const double dist = node.Metric().Evaluate(vec,
      node.Dataset().col(node.Descendant(i)));
  if (dist < trueMinDist)
    trueMinDist = dist;
}
// At the end of the loop, trueMinDist will hold the true minimum distance
// between 'vec' and any descendant point of 'node'.
@endcode

Often the bounding shape of a node will allow a quick calculation that will make
a reasonable bound.  For instance, if the node's bounding shape is a ball with
radius \c r and center \c ctr, the calculation is simply
\c "(node.Metric().Evaluate(vec, ctr) - r)".  Usually a good \c MinDistance() or
\c MaxDistance() function will make only one call to the \c Evaluate() function
of the metric.

The \c RangeDistance() function allows a way for both bounds to be calculated at
once.  It is possible to implement this as a call to \c MinDistance() followed
by a call to \c MaxDistance(), but this may incur more metric \c Evaluate()
calls than necessary.  Often calculating both bounds at once can be more
efficient and can be done with fewer \c Evaluate() calls than calling both
\c MinDistance() and \c MaxDistance().

@subsection treetype_rigorous_serialization Serialization

The last two public functions that the \c TreeType API requires are related to
serialization and printing.

@code
// Return a string representation of the tree.
std::string ToString() const;
@endcode

There are few restrictions on the precise way that the \c ToString() function
should operate, but generally it should behave similarly to the \c ToString()
function in other mlpack methods.  Generally, a user will call \c ToString()
when they want to inspect the object and see what it looks like.  For a tree,
printing the entire tree may be way more information than the user was
expecting, so it may be a better option to print either only the node itself or
the node plus one or two levels of children.

@code
// Serialize the tree (load from the given archive / save to the given
// archive, depending on its type).
template<typename Archive>
void Serialize(Archive& ar, const unsigned int version);

protected:
// A default constructor; only meant to be used by boost::serialization.  This
// must be protected so that boost::serialization will work; it does not need
// to return a valid tree.
ExampleTree();

// Friend access must be given for the default constructor.
friend class boost::serialization::access;
@endcode

On the other hand, the specifics of the functionality required for the
\c Serialize() function are somewhat more difficult.  The \c Serialize()
function will be called either when a tree is being saved to disk or loaded from
disk.  The \c boost::serialization documentation is fairly comprehensive, but
when writing a \c Serialize() method for mlpack trees you should use
\c data::CreateNVP() instead of \c BOOST_SERIALIZATION_NVP().  This is because
mlpack classes implement \c Serialize() instead of \c serialize() in order to
conform to the mlpack style guidelines, and making this work requires some
interesting shim code, which is hidden inside of \c data::CreateNVP().  It may
be useful to look at other \c Serialize() methods contained in other mlpack
classes as an example.

An important note is that it is very difficult to use references with
\c boost::serialization, because \c Serialize() may be called at any time during
the object's lifetime, and references cannot be re-seated.  In general this will
require the use of pointers, which then require manual memory management.
Therefore, be careful that \c Serialize() (and the tree's destructor) properly
handle memory management!

@section treetype_traits The TreeTraits trait class

Some tree-based algorithms can specialize if the tree fulfills certain
conditions.  For instance, if the regions represented by two sibling nodes
cannot overlap, an algorithm may be able to perform a simpler computation.
Based on this reasoning, the \c TreeTraits trait class (much like the
mlpack::kernel::KernelTraits class) exists in order to allow a tree to specify
(via a \c const \c static \c bool) when these types of conditions are
satisfied.  **Note that a TreeTraits class is not required,** but may be
helpful.

The \c TreeTraits trait class is a template class that takes a \c TreeType as a
parameter, and exposes \c const \c static \c bool values that depend on the
tree.  Setting these values is achieved by specialization.  The code below shows
the default \c TreeTraits values (these are the values that will be used if no
specialization is provided for a given \c TreeType).

@code
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
@endcode

An example specialization for the \ref mlpack::tree::KDTree class is given
below.  Note that \ref mlpack::tree::KDTree is itself a template class (like
every class satisfying the \c TreeType policy), so we are specializing to a
template parameter.

@code
template<typename MetricType,
         typename StatisticType,
         typename MatType>
template<>
class TreeTraits<KDTree<MetricType, StatisticType, MatType>>
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
@endcode

Currently, the traits available are each of the five detailed above.  For more
information, see the \ref mlpack::tree::TreeTraits documentation.

@section treetype_more A list of trees in mlpack and more information

mlpack contains several ready-to-use implementations of trees that satisfy the
TreeType policy API:

 - mlpack::tree::KDTree
 - mlpack::tree::MeanSplitKDTree
 - mlpack::tree::BallTree
 - mlpack::tree::MeanSplitBallTree
 - mlpack::tree::RTree
 - mlpack::tree::RStarTree
 - mlpack::tree::StandardCoverTree

Often, these are template typedefs of more flexible tree classes:

 - mlpack::tree::BinarySpaceTree -- binary trees, such as the KD-tree and ball
   tree
 - mlpack::tree::RectangleTree -- the R tree and variants
 - mlpack::tree::CoverTree -- the cover tree and variants

*/
