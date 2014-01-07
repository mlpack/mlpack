/**
 * @file cosine_tree_builder.hpp
 * @author Mudit Raj Gupta
 *
 * Helper class to build the cosine tree
 *
 * This file is part of MLPACK 1.0.8.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_TREE_COSINE_TREE_COSINE_TREE_BUILDER_HPP
#define __MLPACK_CORE_TREE_COSINE_TREE_COSINE_TREE_BUILDER_HPP

#include <mlpack/core.hpp>
#include "cosine_tree.hpp"

using namespace mlpack::tree;

namespace mlpack {
namespace tree /** tree-building procedures. */ {

class CosineTreeBuilder
{
 private:
  /**
   * Length Square Sampling method for sampling rows
   * of the matrix
   *
   * @param A Matrix for which probabilities are calculated
   * @param prob Reference to the probability vector
   */
  void LSSampling(arma::mat A, arma::vec& prob);

  /**
   * Calculates the centroid of the matrix
   *
   * @param A Matrix for which the centroid has to be calculated
   */
  arma::rowvec CalculateCentroid(arma::mat A) const;

  /**
   * Calculates the Pivot for splitting
   *
   * @param prob Probability for a point to act as the pivot
   */
  size_t GetPivot(arma::vec prob);

  /**
   * Splits the points into the root node into children nodes
   *
   * @param c Array of Cosin Similarities
   * @param ALeft Matrix for storing the points in Left Node
   * @param ARight Matrix for storing the points in Right Node
   * @param A All points
   */
  void SplitData(std::vector<double> c, arma::mat& ALeft,
                 arma::mat& Aright, arma::mat A);

  /**
   * Creates Cosine Similarity Array
   *
   * @param c Array of Cosine Similarity
   * @param A All points
   * @param pivot pivot point
   */
  void CreateCosineSimilarityArray(std::vector<double>& c,
                                   arma::mat A, size_t pivot);

  /**
   * Calculates Maximum Cosine Similarity
   *
   * @param c Array of Cosine Similarities
   */
  double GetMaxSimilarity(std::vector<double> c);

  /**
   * Calculates Maximum Cosine Similarity
   *
   * @param c Array of Cosine Similarities
   */
  double GetMinSimilarity(std::vector<double> c);

 public:
  //! Empty Constructor
  CosineTreeBuilder();
  //! Destructor
  ~CosineTreeBuilder();

  /**
   * Creates a new cosine tree node
   *
   * @param A Data for constructing the node
   * @param root Reference to the constructed node
   */
  void CTNode(arma::mat A, CosineTree& root);

  /**
   * Splits a cosine tree node
   *
   * @param root Node to be split
   * @param right reference to the right child
   * @param left reference to the left child
   */
  void CTNodeSplit(CosineTree& root, CosineTree& left, CosineTree& right);
};
}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "cosine_tree_builder_impl.hpp"

#endif
