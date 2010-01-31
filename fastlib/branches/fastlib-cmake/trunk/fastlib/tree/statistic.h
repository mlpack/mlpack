/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file statistic.h
 *
 * Home for the concept of tree statistics.
 *
 * You should define your own statistic that looks like EmptyStatistic.
 *
 * @experimental
 */

#ifndef TREE_STATISTIC_H
#define TREE_STATISTIC_H

/**
 * Empty statistic if you are not interested in storing statistics in your
 * tree.  Use this as a template for your own.
 *
 * @experimental
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
