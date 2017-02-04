/**
 * @file r_plus_tree_split_policy.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition and implementation of the RPlusTreeSplitPolicy class, a class that
 * helps to determine the subtree into which we should insert an intermediate
 * node.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_POLICY_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_POLICY_HPP

namespace mlpack {
namespace tree {

/**
 * The RPlusPlusTreeSplitPolicy helps to determine the subtree into which
 * we should insert a child of an intermediate node that is being split.
 * This class is designed for the R+ tree.
 */
class RPlusTreeSplitPolicy
{
 public:
  //! Indicate that the child should be split.
  static const int SplitRequired = 0;
  //! Indicate that the child should be inserted to the first subtree.
  static const int AssignToFirstTree = 1;
  //! Indicate that the child should be inserted to the second subtree.
  static const int AssignToSecondTree = 2;

  /**
   * This method returns SplitRequired if a child of an intermediate node should
   * be split, AssignToFirstTree if the child should be inserted to the first
   * subtree, AssignToSecondTree if the child should be inserted to the second
   * subtree. The method makes desicion according to the minimum bounding
   * rectangle of the child, the axis along which the intermediate node is being
   * split and the coordinate at which the node is being split.
   *
   * @param child A child of the node that is being split.
   * @param axis The axis along which the node is being split.
   * @param cut The coordinate at which the node is being split.
   */
  template<typename TreeType>
  static int GetSplitPolicy(const TreeType& child,
                            const size_t axis,
                            const typename TreeType::ElemType cut)
  {
    if (child.Bound()[axis].Hi() <= cut)
      return AssignToFirstTree;
    else if (child.Bound()[axis].Lo() >= cut)
      return AssignToSecondTree;

    return SplitRequired;
  }

  /**
   * Return the minimum bounding rectangle of the node.
   * This method should always return the bound that is used for the
   * decision-making in GetSplitPolicy().
   * 
   * @param node The node whose bound is requested.
    */
  template<typename TreeType>
  static const
      bound::HRectBound<metric::EuclideanDistance, typename TreeType::ElemType>&
          Bound(const TreeType& node)
  {
    return node.Bound();
  }
};

} // namespace tree
} // namespace mlpack

#endif // MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_POLICY_HPP
