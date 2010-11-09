#ifndef SINGLE_TREE_H
#define SINGLE_TREE_H
#include <limits>
#include <boost/foreach.hpp>
#include "matching.h"

MATCHING_NAMESPACE_BEGIN;

/** Single tree algorithm to calculate nearest neighbors
  */
template <typename P, typename T>
    class SingleTree
{
public:
  typedef P                   point_type;
  typedef T                   node_type;

  double nearestNeighbor(const point_type& q, node_type& ref, int &minIndex, double &minDistance);

  // distance of q from point index in ref
  double distance(const point_type& q, node_type& ref, int index);

  // distance of q to bounding box of ref
  double distance(const point_type& q, node_type& ref);
};

template <typename P, typename T>
    double SingleTree<P,T>::nearestNeighbor(const point_type &q, node_type& ref,
                                            int &minIndex, double &minDistance)
{
  if (distance(q, ref) >= minDistance)
  {
//    printf("pruned %d\n", ref.n_points());
    return ref.n_points();
  }
  if (ref.isLeaf())   // at leaf, brute force
  {
    for (int i = 0; i < ref.n_points(); i++)
    {
      double val = distance(q, ref, i);
      if (val < minDistance)
      {
        minDistance = val;
        minIndex = i;
      }
    }
    return 0;
  }
  else                // not a leaf, recursive to children
  {
    double pruned = 0;
    for (int i = 0; i < ref.n_children(); i++)
    {
      node_type* child = (node_type*)ref.child(i);
      pruned += nearestNeighbor(q, *child, minIndex, minDistance);
    }
    return pruned;
  }
//  if (minIndex == -1)  // no hint
//    minDistance = std::numeric_limits<double>::infinity();
//  for (int i = 0; i < ref.n_points(); i++)
//  {
//    double val = distance(q, ref, i);
//    if (val < minDistance)
//    {
//      minDistance = val;
//      minIndex = i;
//    }
//  }
}

MATCHING_NAMESPACE_END;

#endif // SINGLE_TREE_H
