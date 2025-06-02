/**
 * @file methods/det/dt_utils.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * This file implements functions to perform different tasks with the Density
 * Tree class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DET_DT_UTILS_HPP
#define MLPACK_METHODS_DET_DT_UTILS_HPP

#include <mlpack/core.hpp>
#include "dtree.hpp"

namespace mlpack {

/**
 * Print the membership of leaves of a density estimation tree given the labels
 * and number of classes.  Optionally, pass the name of a file to print this
 * information to (otherwise stdout is used).
 *
 * @param dtree Tree to print membership of.
 * @param data Dataset tree is built upon.
 * @param labels Class labels of dataset.
 * @param numClasses Number of classes in dataset.
 * @param leafClassMembershipFile Name of file to print to (optional).
 */
template <typename MatType, typename TagType>
void PrintLeafMembership(DTree<MatType, TagType>* dtree,
                         const MatType& data,
                         const arma::Mat<size_t>& labels,
                         const size_t numClasses,
                         const std::string& leafClassMembershipFile = "");

/**
 * Print the variable importance of each dimension of a density estimation tree.
 * Optionally, pass the name of a file to print this information to (otherwise
 * stdout is used).
 *
 * @param dtree Density tree to use.
 * @param viFile Name of file to print to (optional).
 */
template <typename MatType, typename TagType>
void PrintVariableImportance(const DTree<MatType, TagType>* dtree,
                             const std::string viFile = "");

/**
 * Train the optimal decision tree using cross-validation with the given number
 * of folds.  Optionally, give a filename to print the unpruned tree to.  This
 * initializes a tree on the heap, so you are responsible for deleting it.
 *
 * @param dataset Dataset for the tree to use.
 * @param folds Number of folds to use for cross-validation.
 * @param useVolumeReg If true, use volume regularization.
 * @param maxLeafSize Maximum number of points allowed in a leaf.
 * @param minLeafSize Minimum number of points allowed in a leaf.
 * @param unprunedTreeOutput Filename to print unpruned tree to (optional).
 * @param skipPruning Set true to skip pruning.
 */
template <typename MatType, typename TagType>
DTree<MatType, TagType>* Trainer(MatType& dataset,
                                 const size_t folds,
                                 const bool useVolumeReg = false,
                                 const size_t maxLeafSize = 10,
                                 const size_t minLeafSize = 5,
                                 const std::string unprunedTreeOutput = "",
                                 const bool skipPruning = false,
                                 util::Timers& timers = IO::GetTimers());

/**
 * This class is responsible for caching the path to each node of the tree. Its
 * instance is provided to EnumerateTree() utility ONCE and it caches the paths
 * to all the leafs and then easily (and quickly) retrieves these paths for each
 * test entry.
 */
class PathCacher
{
 public:
  /**
   * Possible formats to use for output.
   */
  enum PathFormat
  {
    //! Print only whether we went left or right.
    FormatLR,
    //! Print the direction, then the tag of the node.
    FormatLR_ID,
    //! Print the tag of the node, then the direction.
    FormatID_LR
  };

  /**
   * Construct a PathCacher object on the given tree with the given format.
   *
   * @param fmt Format to use for output.
   * @param tree Tree to cache paths in.
   */
  template<typename MatType>
  PathCacher(PathFormat fmt, DTree<MatType, int>* tree);

  /**
   * Enter a given node.
   */
  template<typename MatType>
  void Enter(const DTree<MatType, int>* node,
             const DTree<MatType, int>* parent);

  /**
   * Leave the given node.
   */
  template<typename MatType>
  void Leave(const DTree<MatType, int>* node,
             const DTree<MatType, int>* parent);

  /**
   * Return the constructed path for a given tag.
   */
  inline const std::string& PathFor(int tag) const;

  /**
   * Get the parent tag of a given tag.
   */
  inline int ParentOf(int tag) const;

  /**
   * Get the number of nodes in the path cache.
   */
  size_t NumNodes() const { return pathCache.size(); }

 protected:
  using PathType = std::list<std::pair<bool, int>>;
  using PathCacheType = std::vector<std::pair<int, std::string>>;

  PathType      path;
  PathFormat    format;
  PathCacheType pathCache;

  inline std::string   BuildString();
};

} // namespace mlpack

#include "dt_utils_impl.hpp"

#endif // MLPACK_METHODS_DET_DT_UTILS_HPP
