/**
 * @file statistic.hpp
 *
 * Definition of the policy type for the statistic class.
 *
 * You should define your own statistic that looks like EmptyStatistic.
 *
 * This file is part of MLPACK 1.0.10.
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
    EmptyStatistic() { }
    ~EmptyStatistic() { }

    /**
     * This constructor is called when a node is finished being created.  The
     * node is finished, and its children are finished, but it is not
     * necessarily true that the statistics of other nodes are initialized yet.
     *
     * @param node Node which this corresponds to.
     */
    template<typename TreeType>
    EmptyStatistic(TreeType& /* node */) { }

  public:
    /**
     * Returns a string representation of this object.
     */
    std::string ToString() const
    {
      std::stringstream convert;
      convert << "EmptyStatistic [" << this << "]" << std::endl;
      return convert.str();
    }
};

}; // namespace tree
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_STATISTIC_HPP
