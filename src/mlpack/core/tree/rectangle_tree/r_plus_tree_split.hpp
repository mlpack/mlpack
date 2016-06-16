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

const double fillFactorFraction = 0.5;

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

class RPlusTreeSplit
{
 public:
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

  template<typename ElemType>
  static bool StructComp(const SortStruct<ElemType>& s1,
                         const SortStruct<ElemType>& s2)
  {
    return s1.d < s2.d;
  }

  template<typename TreeType>
  static void SplitLeafNodeAlongPartition(TreeType* tree,
      TreeType* treeOne, TreeType* treeTwo, size_t cutAxis, double cut);

  template<typename TreeType>
  static void SplitNonLeafNodeAlongPartition(TreeType* tree,
      TreeType* treeOne, TreeType* treeTwo, size_t cutAxis, double cut);

  template<typename TreeType>
  static bool PartitionNode(const TreeType* node, size_t fillFactor,
      size_t& minCutAxis, double& minCut);

  template<typename TreeType>
  static double SweepLeafNode(size_t axis, const TreeType* node,
      size_t fillFactor, double& axisCut);

  template<typename TreeType>
  static double SweepNonLeafNode(size_t axis, const TreeType* node,
      size_t fillFactor, double& axisCut);

  template<typename TreeType>
  static void InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode);

  template<typename TreeType>
  static bool CheckNonLeafSweep(const TreeType* node,
      size_t cutAxis, double cut);

  template<typename TreeType>
  static bool CheckLeafSweep(const TreeType* node, size_t cutAxis, double cut);
};

} // namespace tree
} // namespace mlpack

// Include implementation
#include "r_plus_tree_split_impl.hpp"

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_HPP

