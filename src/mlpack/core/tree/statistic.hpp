/**
 * @file statistic.hpp
 *
 * Definition of the policy type for the statistic class.
 *
 * You should define your own statistic that looks like EmptyStatistic.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
