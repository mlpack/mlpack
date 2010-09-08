/**
 * @file statistic.h
 *
 * Home for the concept of tree statistics.
 *
 * You should define your own statistic that looks like EmptyStatistic.
 */

#ifndef CORE_TREE_STATISTIC_H
#define CORE_TREE_STATISTIC_H

#include <armadillo>

/**
 * Empty statistic if you are not interested in storing statistics in your
 * tree.  Use this as a template for your own.
 */

namespace core {
namespace tree {
class AbstractStatistic {
  public:

    virtual ~AbstractStatistic() {
    }

    /**
     * Initializes by taking statistics on raw data.
     */
    template<typename TreeIteratorType>
    void Init(TreeIteratorType &it) {
    }

    /**
     * Initializes by combining statistics of two partitions.
     *
     * This lets you build fast bottom-up statistics when building trees.
     */
    template<typename TreeIteratorType>
    void Init(
      TreeIteratorType &it,
      const core::tree::AbstractStatistic *left_stat,
      const core::tree::AbstractStatistic *right_stat) {
    }
};
};
};

#endif
