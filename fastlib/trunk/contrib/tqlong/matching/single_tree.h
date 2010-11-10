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

  static double nearestNeighbor(const point_type& q, node_type& ref, int &minIndex, double &minDistance);
  static double kNearestNeighbor(const point_type& q, node_type& ref, std::vector<int>& minIndexes, std::vector<double>& minDistances);

  // distance of q from point index in ref
  static double distance(const point_type& q, node_type& ref, int index);

  // distance of q to bounding box of ref
  static double distance(const point_type& q, node_type& ref);
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
        minIndex = ref.index(i); // return index from root view
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

template <typename P, typename T>
    double SingleTree<P,T>::kNearestNeighbor(const point_type &q, node_type &ref,
                                             std::vector<int> &minIndexes, std::vector<double> &minDistances)
{
  int k = (int) minIndexes.size();
  if (distance(q, ref) >= minDistances[k-1])
  {
    return ref.n_points();
  }
  if (ref.isLeaf())
  {
    for (int i = 0; i < ref.n_points(); i++)
    {
      double val = distance(q, ref, i);
      for (int order = 0; order < k; order++)
      {
        if (val < minDistances[order])
        {
          for (int k_reverse = k-1; k_reverse > order; k_reverse--)
          {
            minDistances[k_reverse] = minDistances[k_reverse-1];
            minIndexes[k_reverse] = minIndexes[k_reverse-1];
          }
          minDistances[order] = val;
          minIndexes[order] = ref.index(i); // return index from root view
          break;
        }
      }
    }
    return 0;
  }
  else
  {
    double pruned = 0;
    for (int i = 0; i < ref.n_children(); i++)
    {
      node_type* child = (node_type*)ref.child(i);
      pruned += kNearestNeighbor(q, *child, minIndexes, minDistances);
    }
    return pruned;
  }
}

MATCHING_NAMESPACE_END;

#endif // SINGLE_TREE_H
