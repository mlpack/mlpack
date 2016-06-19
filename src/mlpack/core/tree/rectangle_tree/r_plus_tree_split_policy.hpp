/**
 * @file r_plus_tree_split_policy.hpp
 * @author Mikhail Lozhnikov
 *
 * Defintion and implementation of the RPlusTreeSplitPolicy class, a class that
 * helps to determine the node into which we should insert an intermediate node.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_POLICY_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_POLICY_HPP

namespace mlpack {
namespace tree {

class RPlusTreeSplitPolicy
{
 public:
  static const int SplitRequired = 0;
  static const int AssignToFirstTree = 1;
  static const int AssignToSecondTree = 2;

  template<typename TreeType>
  static int GetSplitPolicy(const TreeType* child, size_t axis,
      typename TreeType::ElemType cut)
  {
    if (child->Bound()[axis].Hi() <= cut)
      return AssignToFirstTree;
    else if (child->Bound()[axis].Lo() >= cut)
      return AssignToSecondTree;

    return SplitRequired;
  }

  template<typename TreeType>
  static const
      bound::HRectBound<metric::EuclideanDistance, typename TreeType::ElemType>&
          Bound(const TreeType* node)
  {
    return node->Bound();
  }
};

} //  namespace tree
} //  namespace mlpack
#endif //  MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_POLICY_HPP


