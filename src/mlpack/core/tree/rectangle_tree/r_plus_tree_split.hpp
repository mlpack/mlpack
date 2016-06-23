/**
 * @file r_plus_tree_split.hpp
 * @author Mikhail Lozhnikov
 *
 * Defintion of the RPlusTreeSplit class, a class that splits the nodes of an R
 * tree, starting at a leaf node and moving upwards if necessary.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

template<typename SplitPolicyType,
         template<typename> class SweepType>
class RPlusTreeSplit
{
 public:
  typedef SplitPolicyType SplitPolicy;
  /**
   * Split a leaf node using the "default" algorithm.  If necessary, this split
   * will propagate upwards through the tree.
   * @param node. The node that is being split.
   * @param relevels Not used.
   */
  template<typename TreeType>
  static void SplitLeafNode(TreeType *tree,std::vector<bool>& relevels);

  /**
   * Split a non-leaf node using the "default" algorithm.  If this is a root
   * node, the tree increases in depth.
   * @param node. The node that is being split.
   * @param relevels Not used.
   */
  template<typename TreeType>
  static bool SplitNonLeafNode(TreeType *tree,std::vector<bool>& relevels);



 private:

  template<typename ElemType>
  struct SortStruct
  {
    ElemType d;
    int n;
  };

  template<typename TreeType>
  static void SplitLeafNodeAlongPartition(TreeType* tree, TreeType* treeOne,
      TreeType* treeTwo, size_t cutAxis, typename TreeType::ElemType cut);

  template<typename TreeType>
  static void SplitNonLeafNodeAlongPartition(TreeType* tree, TreeType* treeOne,
      TreeType* treeTwo, size_t cutAxis, typename TreeType::ElemType cut);

  template<typename TreeType>
  static void AddFakeNodes(const TreeType* tree, TreeType* emptyTree);

  template<typename TreeType>
  static bool PartitionNode(const TreeType* node, size_t& minCutAxis,
      typename TreeType::ElemType& minCut);

  template<typename TreeType>
  static void InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode);

};

} // namespace tree
} // namespace mlpack

// Include implementation
#include "r_plus_tree_split_impl.hpp"

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_HPP

