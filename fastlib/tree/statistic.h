// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file statistic.h
 *
 * Home for the concept of tree statistics.
 *
 * You should define your own statistic that looks like EmptyStatistic.
 */

#ifndef TREE_STATISTIC_H
#define TREE_STATISTIC_H

/**
 * Empty statistic if you are not interested in storing statistics in your
 * tree.  Use this as a template for your own.
 */
template<class TDataset>
class EmptyStatistic {
 public:
  EmptyStatistic() {}
  ~EmptyStatistic() {}
  
  /**
   * Initializes by taking statistics on raw data.
   */
  void Init(const TDataset& dataset, index_t start, index_t count) {
  }

  /**
   * Initializes by combining statistics of two partitions.
   *
   * This lets you build fast bottom-up statistics when building trees.
   */
  void Init(const TDataset& dataset, index_t start, index_t count,
      const EmptyStatistic& left_stat, const EmptyStatistic& right_stat) {
  }
};

#endif
