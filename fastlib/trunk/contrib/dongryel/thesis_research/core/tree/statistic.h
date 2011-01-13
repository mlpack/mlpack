/** @file statistic.h
 *
 *  Home for the concept of tree statistics.
 *
 *  You should define your own statistic that looks like EmptyStatistic.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_STATISTIC_H
#define CORE_TREE_STATISTIC_H

#include <boost/serialization/string.hpp>

namespace core {
namespace tree {

/** @brief Empty statistic if you are not interested in storing
 *         statistics in your tree.  Use this as a template for your
 *         own.
 */
class AbstractStatistic {
  private:

    friend class boost::serialization::access;

  public:

    /** @brief Copies another abstract statistics (does not do anything).
     */
    void Copy(const AbstractStatistic &stat_in) {
    }

    /** @brief Resets the statistics.
     */
    void SetZero() {
    }

    /** @brief Serialization/deserialization does not save anything
     *         for abstract stats.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
    }

    /** @brief Initializes by taking statistics on raw data.
     */
    template<typename TreeIteratorType>
    void Init(TreeIteratorType &it) {
    }

    /** @brief Initializes by combining statistics of two partitions.
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
