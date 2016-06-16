/**
 * @file gini_impurity.hpp
 * @author Ryan Curtin
 *
 * The GiniImpurity class, which is a fitness function (FitnessFunction) for
 * streaming decision trees.
 *
 * This file is part of mlpack 2.0.2.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef MLPACK_METHODS_HOEFFDING_TREES_GINI_INDEX_HPP
#define MLPACK_METHODS_HOEFFDING_TREES_GINI_INDEX_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

class GiniImpurity
{
 public:
  static double Evaluate(const arma::Mat<size_t>& counts)
  {
    // We need to sum over the difference between the un-split node and the
    // split nodes.  First we'll calculate the number of elements in each split
    // and total.
    size_t numElem = 0;
    arma::vec splitCounts(counts.n_cols);
    for (size_t i = 0; i < counts.n_cols; ++i)
    {
      splitCounts[i] = arma::accu(counts.col(i));
      numElem += splitCounts[i];
    }

    // Corner case: if there are no elements, the impurity is zero.
    if (numElem == 0)
      return 0.0;

    arma::Col<size_t> classCounts = arma::sum(counts, 1);

    // Calculate the Gini impurity of the un-split node.
    double impurity = 0.0;
    for (size_t i = 0; i < classCounts.n_elem; ++i)
    {
      const double f = ((double) classCounts[i] / (double) numElem);
      impurity += f * (1.0 - f);
    }

    // Now calculate the impurity of the split nodes and subtract them from the
    // overall impurity.
    for (size_t i = 0; i < counts.n_cols; ++i)
    {
      if (splitCounts[i] > 0)
      {
        double splitImpurity = 0.0;
        for (size_t j = 0; j < counts.n_rows; ++j)
        {
          const double f = ((double) counts(j, i) / (double) splitCounts[i]);
          splitImpurity += f * (1.0 - f);
        }

        impurity -= ((double) splitCounts[i] / (double) numElem) *
            splitImpurity;
      }
    }

    return impurity;
  }

  /**
   * Return the range of the Gini impurity for the given number of classes.
   * (That is, the difference between the maximum possible value and the minimum
   * possible value.)
   */
  static double Range(const size_t numClasses)
  {
    // The best possible case is that only one class exists, which gives a Gini
    // impurity of 0.  The worst possible case is that the classes are evenly
    // distributed, which gives n * (1/n * (1 - 1/n)) = 1 - 1/n.
    return 1.0 - (1.0 / double(numClasses));
  }
};

} // namespace tree
} // namespace mlpack

#endif
