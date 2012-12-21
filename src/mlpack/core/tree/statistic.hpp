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
