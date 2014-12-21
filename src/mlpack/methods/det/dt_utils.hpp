/**
 * @file dt_utils.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * This file implements functions to perform different tasks with the Density
 * Tree class.
 *
 * This file is part of MLPACK 1.0.9.
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
#ifndef __MLPACK_METHODS_DET_DT_UTILS_HPP
#define __MLPACK_METHODS_DET_DT_UTILS_HPP

#include <mlpack/core.hpp>
#include "dtree.hpp"

namespace mlpack {
namespace det {

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
void PrintLeafMembership(DTree* dtree,
                         const arma::mat& data,
                         const arma::Mat<size_t>& labels,
                         const size_t numClasses,
                         const std::string leafClassMembershipFile = "");

/**
 * Print the variable importance of each dimension of a density estimation tree.
 * Optionally, pass the name of a file to print this information to (otherwise
 * stdout is used).
 *
 * @param dtree Density tree to use.
 * @param viFile Name of file to print to (optional).
 */
void PrintVariableImportance(const DTree* dtree,
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
 */
DTree* Trainer(arma::mat& dataset,
               const size_t folds,
               const bool useVolumeReg = false,
               const size_t maxLeafSize = 10,
               const size_t minLeafSize = 5,
               const std::string unprunedTreeOutput = "");

}; // namespace det
}; // namespace mlpack

#endif // __MLPACK_METHODS_DET_DT_UTILS_HPP
