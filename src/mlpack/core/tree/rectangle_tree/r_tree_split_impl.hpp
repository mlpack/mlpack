/**
 * @file r_tree_split_impl.hpp
 * @author Andrew Wells
 *
 * Implementation of class (RTreeSplit) to split a RectangleTree.
 */
#ifndef __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_SPLIT_IMPL_HPP
#define __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_SPLIT_IMPL_HPP

#include "r_tree_split.hpp"

namespace mlpack {
namespace tree {

template<typename MatType>
bool RTreeSplit<MatType>::SplitLeafNode(const RectangleTree& tree)
{

}

bool RTreeSplit<MatType>::SplitNonLeafNode(const RectangleTree& tree)
{

}

}; // namespace tree
}; // namespace mlpack

#endif
