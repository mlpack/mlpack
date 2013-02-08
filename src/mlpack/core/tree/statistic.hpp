/**
 * @file statistic.hpp
 *
 * Definition of the policy type for the statistic class.
 *
 * You should define your own statistic that looks like EmptyStatistic.
 *
 * This file is part of MLPACK 1.0.4.
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

#ifndef __MLPACK_CORE_TREE_STATISTIC_HPP
#define __MLPACK_CORE_TREE_STATISTIC_HPP

namespace mlpack {
namespace tree {

/**
 * Empty statistic if you are not interested in storing statistics in your
 * tree.  Use this as a template for your own.
 */
class EmptyStatistic
{
  public:
    EmptyStatistic() {}
    ~EmptyStatistic() {}

    /**
     * This constructor is called when a leaf is created.
     *
     * @param dataset Matrix that the tree is being built on.
     * @param begin Starting index corresponding to this leaf.
     * @param count Number of points held in this leaf.
     */
    template<typename MatType>
    EmptyStatistic(const MatType& /* dataset */,
                   const size_t /* begin */,
                   const size_t /* count */)
    { }

    /**
     * This constructor is called when a non-leaf node is created.
     * This lets you build fast bottom-up statistics when building trees.
     *
     * @param dataset Matrix that the tree is being built on.
     * @param begin Starting index corresponding to this leaf.
     * @param count Number of points held in this leaf.
     * @param leftStat EmptyStatistic object of the left child node.
     * @param rightStat EmptyStatistic object of the right child node.
     */
    template<typename MatType>
    EmptyStatistic(const MatType& /* dataset */,
                   const size_t /* start */,
                   const size_t /* count */,
                   const EmptyStatistic& /* leftStat */,
                   const EmptyStatistic& /* rightStat */)
    { }
};

}; // namespace tree
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_STATISTIC_HPP
