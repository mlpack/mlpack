/**
 * @file statistic.hpp
 *
 * Definition of the policy type for the statistic class.
 *
 * You should define your own statistic that looks like EmptyStatistic.
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

    /**
     * Serialize the statistic (there's nothing to be saved).
     */
    template<typename Archive>
    void Serialize(Archive& /* ar */, const unsigned int /* version */)
    { }

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

} // namespace tree
} // namespace mlpack

#endif // __MLPACK_CORE_TREE_STATISTIC_HPP
