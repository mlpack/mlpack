/**
 * @file methods/neighbor_search/sort_policies/furthest_neighbor_sort.hpp
 * @author Ryan Curtin
 *
 * Implementation of the SortPolicy class for NeighborSearch; in this case, the
 * furthest neighbors are those that are most important.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_FURTHEST_NEIGHBOR_SORT_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_FURTHEST_NEIGHBOR_SORT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This class implements the necessary methods for the SortPolicy template
 * parameter of the NeighborSearch class.  The sorting policy here is that the
 * minimum distance is the best (so, when used with NeighborSearch, the output
 * is furthest neighbors).
 */
class FurthestNS
{
 public:
  /**
   * Return whether or not value is "better" than ref.  In this case, that means
   * that the value is greater than or equal to the reference.
   *
   * @param value Value to compare
   * @param ref Value to compare with
   *
   * @return bool indicating whether or not (value >= ref).
   */
  static inline bool IsBetter(const double value, const double ref)
  {
    return (value >= ref);
  }

  /**
   * Return the best possible distance between two nodes.  In our case, this is
   * the maximum distance between the two tree nodes using the given distance
   * function.
   */
  template<typename TreeType>
  static double BestNodeToNodeDistance(const TreeType* queryNode,
                                       const TreeType* referenceNode);

  /**
   * Return the best possible distance between two nodes, given that the
   * distance between the centers of the two nodes has already been calculated.
   * This is used in conjunction with trees that have self-children (like cover
   * trees).
   */
  template<typename TreeType>
  static double BestNodeToNodeDistance(const TreeType* queryNode,
                                       const TreeType* referenceNode,
                                       const double centerToCenterDistance);

  /**
   * Return the best possible distance between the query node and the reference
   * child node given the base case distance between the query node and the
   * reference node.  TreeType::ParentDistance() must be implemented to use
   * this.
   *
   * @param queryNode Query node.
   * @param referenceNode Reference node.
   * @param referenceChildNode Child of reference node which is being inspected.
   * @param centerToCenterDistance Distance between centers of query node and
   *     reference node.
   */
  template<typename TreeType>
  static double BestNodeToNodeDistance(const TreeType* queryNode,
                                       const TreeType* referenceNode,
                                       const TreeType* referenceChildNode,
                                       const double centerToCenterDistance);

  /**
   * Return the best possible distance between a node and a point.  In our case,
   * this is the maximum distance between the tree node and the point using the
   * given distance function.
   */
  template<typename VecType, typename TreeType>
  static double BestPointToNodeDistance(const VecType& queryPoint,
                                        const TreeType* referenceNode);

  /**
   * Return the best possible distance between a point and a node, given that
   * the distance between the point and the center of the node has already been
   * calculated.  This is used in conjunction with trees that have
   * self-children (like cover trees).
   */
  template<typename VecType, typename TreeType>
  static double BestPointToNodeDistance(const VecType& queryPoint,
                                        const TreeType* referenceNode,
                                        const double pointToCenterDistance);

  /**
   * Return the best child according to this sort policy. In this case it will
   * return the one with the maximum distance.
   */
  template<typename VecType, typename TreeType>
  static size_t GetBestChild(const VecType& queryPoint, TreeType& referenceNode)
  {
    return referenceNode.GetFurthestChild(queryPoint);
  };

  /**
   * Return the best child according to this sort policy. In this case it will
   * return the one with the maximum distance.
   */
  template<typename TreeType>
  static size_t GetBestChild(const TreeType& queryNode, TreeType& referenceNode)
  {
    return referenceNode.GetFurthestChild(queryNode);
  };

  /**
   * Return what should represent the worst possible distance with this
   * particular sort policy.  In our case, this should be the minimum possible
   * distance, 0.
   *
   * @return 0
   */
  static inline double WorstDistance() { return 0; }

  /**
   * Return what should represent the best possible distance with this
   * particular sort policy.  In our case, this should be the maximum possible
   * distance, DBL_MAX.
   *
   * @return DBL_MAX
   */
  static inline double BestDistance() { return DBL_MAX; }

  /**
   * Return the best combination of the two distances.
   */
  static inline double CombineBest(const double a, const double b)
  {
    if (a == DBL_MAX || b == DBL_MAX)
      return DBL_MAX;
    return a + b;
  }

  /**
   * Return the worst combination of the two distances.
   */
  static inline double CombineWorst(const double a, const double b)
  { return std::max(a - b, 0.0); }

  /**
   * Return the given value relaxed.
   *
   * @param value Value to relax.
   * @param epsilon Relative error (non-negative).
   *
   * @return double Value relaxed.
   */
  static inline double Relax(const double value, const double epsilon)
  {
    if (value == 0)
      return 0;
    if (value == DBL_MAX || epsilon >= 1)
      return DBL_MAX;
    return (1 / (1 - epsilon)) * value;
  }

  /**
   * Convert the given distance to a score.  Lower scores are better, but for
   * furthest neighbor search, larger distances are better.  Therefore we must
   * invert the given distance.
   */
  static inline double ConvertToScore(const double distance)
  {
    if (distance == DBL_MAX)
      return 0.0;
    else if (distance == 0.0)
      return DBL_MAX;
    else
      return (1.0 / distance);
  }

  /**
   * Convert the given score back to a distance.  This is the inverse of the
   * operation of converting a distance to a score, and again, for furthest
   * neighbor search, corresponds to inverting the value.
   */
  static inline double ConvertToDistance(const double score)
  {
    return ConvertToScore(score);
  }
};

// Due to an internal MinGW compiler bug (string table overflow) we have to
// truncate the class name. For backward compatibility we setup an alias here.
using FurthestNeighborSort = FurthestNS;

} // namespace mlpack

// Include implementation of templated functions.
#include "furthest_neighbor_sort_impl.hpp"

#endif
