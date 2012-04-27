/**
 * @file cover_tree.hpp
 * @author Ryan Curtin
 *
 * Definition of CoverTree, which can be used in place of the BinarySpaceTree.
 */
#ifndef __MLPACK_CORE_TREE_COVER_TREE_HPP
#define __MLPACK_CORE_TREE_COVER_TREE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

template<typename StatisticType = EmptyStatistic>
class CoverTree
{
 public:
  /**
   * Create the cover tree.
   */
  CoverTree(const arma::mat& dataset,
            const double expansionConstant = 2.0);

  CoverTree(const arma::mat& dataset,
            const double expansionConstant,
            const size_t pointIndex,
            const int scale,
            arma::Col<size_t>& indices,
            arma::vec& distances,
            size_t nearSetSize,
            size_t& farSetSize,
            size_t& usedSetSize);

  ~CoverTree();

  //! Get a reference to the dataset.
  const arma::mat& Dataset() const { return dataset; }

  //! Get the index of the point which this node represents.
  size_t Point() const { return point; }

  //! Get a particular child node.
  const CoverTree& Child(const size_t index) const { return *children[index]; }
  //! Modify a particular child node.
  CoverTree& Child(const size_t index) { return *children[index]; }

  //! Get the number of children.
  size_t NumChildren() const { return children.size(); }

  //! Get the scale of this node.
  int Scale() const { return scale; }
  //! Modify the scale of this node.  Be careful...
  int& Scale() { return scale; }

  //! Get the expansion constant.
  double ExpansionConstant() const { return expansionConstant; }
  //! Modify the expansion constant; don't do this, you'll break everything.
  double& ExpansionConstant() { return expansionConstant; }

 private:
  //! Reference to the matrix which this tree is built on.
  const arma::mat& dataset;

  //! Index of the point in the matrix which this node represents.
  size_t point;

  //! The distance to the furthest descendant.
  double furthestDistance; // Better name?

  //! The distance to the parent.
  double parentDistance; // Better name?

  //! The list of children; the first is the "self child" (what is this?).
  std::vector<CoverTree*> children;

  //! Depth of the node in terms of scale (in terms of what?).
//  size_t scaleDepth;
  //! Scale level of the node.
  int scale;

  //! The expansion constant used to construct the tree.
  double expansionConstant;

  //! The instantiated statistic.
  StatisticType stat;
};

// Utility functions; these should probably be incorporated into the class
// itself sometime soon.

size_t SplitNearFar(arma::Col<size_t>& indices,
                    arma::vec& distances,
                    const double bound,
                    const size_t pointSetSize);

void BuildDistances(const arma::mat& dataset,
                    const size_t pointIndex,
                    const arma::Col<size_t>& indices,
                    arma::vec& distances,
                    const size_t pointSetSize);

size_t SortPointSet(arma::Col<size_t>& indices,
                    arma::vec& distances,
                    const size_t childFarSetSize,
                    const size_t childUsedSetSize,
                    const size_t farSetSize);

};
};

// Include implementation.
#include "cover_tree_impl.hpp"

#endif
