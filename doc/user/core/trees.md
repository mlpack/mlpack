# Trees

mlpack includes a number of space partitioning trees and other trees for its
geometric techniques.  All of mlpack's trees implement
the [same API](../../developer/trees.md), allowing easy plug-and-play usage of
different trees.  The following tree types are available in mlpack:

 * [`KDTree`](trees/kdtree.md)
 * [`MeanSplitKDTree`](trees/mean_split_kdtree.md)
 * [`MeanSplitBallTree`](trees/mean_split_ball_tree.md)
 * [`BinarySpaceTree`](trees/binary_space_tree.md)
 * [`BallTree`](trees/ball_tree.md)

*Note:* this documentation is a work in progress.  Not all trees are documented
yet.

---

In general, it is not necessary to create an mlpack tree directly, but instead
to simply specify the type of tree a particular algorithm should use via a
template parameter.  For instance, all of the algorithms below use mlpack trees
and can have the type of tree specified via template parameters:

<!-- TODO: document these! -->

 * [`NeighborSearch`](/src/mlpack/methods/neighbor_search/neighbor_search.hpp)
   (for k-nearest-neighbor and k-furthest-neighbor)
 * [`RangeSearch`](/src/mlpack/methods/range_search/range_search.hpp)
 * [`KDE`](/src/mlpack/methods/kde/kde.hpp)
 * [`FastMKS`](/src/mlpack/methods/fastmks/fastmks.hpp)
 * [`DTB`](/src/mlpack/methods/emst/dtb.hpp) (for computing Euclidean minimum
   spanning trees)
 * [`KRANN`](/src/mlpack/methods/rann/rann.hpp)

---

***Note:*** if you are looking for documentation on **decision trees**, see the
documentation for the [`DecisionTree`](../methods/decision_tree.md) class.
