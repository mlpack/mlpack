/**
 * @file statistic.h
 *
 * Home for the concept of tree statistics.
 *
 * You should define your own statistic that looks like EmptyStatistic.
 *
 * @experimental
 */

#ifndef __MLPACK_CORE_TREE_STATISTIC_HPP
#define __MLPACK_CORE_TREE_STATISTIC_HPP

/**
 * Empty statistic if you are not interested in storing statistics in your
 * tree.  Use this as a template for your own.
 *
 * @experimental
 */
class EmptyStatistic
{
  public:
    EmptyStatistic() {}
    ~EmptyStatistic() {}

    /**
     * Initializes by taking statistics on raw data.
     */
    void Init(const arma::mat& dataset, size_t start, size_t count) { }

    /**
     * Initializes by combining statistics of two partitions.
     *
     * This lets you build fast bottom-up statistics when building trees.
     */
    void Init(const arma::mat& dataset, size_t start, size_t count,
        const EmptyStatistic& left_stat, const EmptyStatistic& right_stat) { }
};

#endif // __MLPACK_CORE_TREE_STATISTIC_HPP
