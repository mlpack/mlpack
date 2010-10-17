/**
 * @file general_spacetree.h
 *
 * Generalized space partitioning tree.
 *
 */

#ifndef CORE_TREE_GENERAL_SPACETREE_H
#define CORE_TREE_GENERAL_SPACETREE_H

#include <armadillo>
#include <boost/serialization/string.hpp>
#include "core/table/dense_matrix.h"
#include "statistic.h"

/**
 * A binary space partitioning tree, such as KD or ball tree.
 *
 * This particular tree forbids you from having more children.
 *
 * @param TBound the bounding type of each child
 * @param TStatistic extra data in the node
 *
 */
namespace core {
namespace tree {
template < class TBound >
class GeneralBinarySpaceTree {
  public:
    typedef TBound BoundType;

    /** @brief The bound for the node.
     */
    BoundType bound_;

    /** @brief The pointer to the left node.
     */
    GeneralBinarySpaceTree *left_;

    /** @brief The pointer to the right node.
     */
    GeneralBinarySpaceTree *right_;

    /** @brief The beginning index.
     */
    int begin_;

    /** @brief The number of points contained within the node.
     */
    int count_;

    /** @brief The statistics for the points owned within the node.
     */
    core::tree::AbstractStatistic *stat_;

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & bound_;
      ar & begin_;
      ar & count_;
      ar & left_;
      ar & right_;
    }

    ~GeneralBinarySpaceTree() {
      if(left_ != NULL) {
        delete left_;
        left_ = NULL;
      }
      if(right_ != NULL) {
        delete right_;
        right_ = NULL;
      }
      if(stat_ != NULL) {
        delete stat_;
        stat_ = NULL;
      }
    }

    GeneralBinarySpaceTree() {
      left_ = NULL;
      right_ = NULL;
      begin_ = -1;
      count_ = -1;
      stat_ = NULL;
    }

  public:

    void Init(int begin_in, int count_in) {
      begin_ = begin_in;
      count_ = count_in;
    }

    /**
     * Find a node in this tree by its begin and count.
     *
     * Every node is uniquely identified by these two numbers.
     * This is useful for communicating position over the network,
     * when pointers would be invalid.
     *
     * @param begin_q the begin() of the node to find
     * @param count_q the count() of the node to find
     * @return the found node, or NULL
     */
    const GeneralBinarySpaceTree* FindByBeginCount(
      int begin_q, int count_q) const {

      if(begin_ == begin_q && count_ == count_q) {
        return this;
      }
      else if(is_leaf()) {
        return NULL;
      }
      else if(begin_q < right_->begin_) {
        return left_->FindByBeginCount(begin_q, count_q);
      }
      else {
        return right_->FindByBeginCount(begin_q, count_q);
      }
    }

    /**
     * Find a node in this tree by its begin and count (const).
     *
     * Every node is uniquely identified by these two numbers.
     * This is useful for communicating position over the network,
     * when pointers would be invalid.
     *
     * @param begin_q the begin() of the node to find
     * @param count_q the count() of the node to find
     * @return the found node, or NULL
     */
    GeneralBinarySpaceTree* FindByBeginCount(
      int begin_q, int count_q) {

      if(begin_ == begin_q && count_ == count_q) {
        return this;
      }
      else if(is_leaf()) {
        return NULL;
      }
      else if(begin_q < right_->begin_) {
        return left_->FindByBeginCount(begin_q, count_q);
      }
      else {
        return right_->FindByBeginCount(begin_q, count_q);
      }
    }

    /**
     * Used only when constructing the tree.
     */
    void set_children(
      const core::table::DenseMatrix& data, GeneralBinarySpaceTree *left_in,
      GeneralBinarySpaceTree *right_in) {

      left_ = left_in;
      right_ = right_in;
    }

    const BoundType &bound() const {
      return bound_;
    }

    BoundType &bound() {
      return bound_;
    }

    const core::tree::AbstractStatistic *&stat() const {
      return stat_;
    }

    core::tree::AbstractStatistic *&stat() {
      return stat_;
    }

    bool is_leaf() const {
      return !left_;
    }

    /**
     * Gets the left branch of the tree.
     */
    const GeneralBinarySpaceTree *left() const {
      return left_;
    }

    GeneralBinarySpaceTree *&left() {
      return left_;
    }

    /**
     * Gets the right branch.
     */
    const GeneralBinarySpaceTree *right() const {
      return right_;
    }

    GeneralBinarySpaceTree *&right() {
      return right_;
    }

    /**
     * Gets the index of the begin point of this subset.
     */
    int begin() const {
      return begin_;
    }

    int &begin() {
      return begin_;
    }

    /**
     * Gets the index one beyond the last index in the series.
     */
    int end() const {
      return begin_ + count_;
    }

    /**
     * Gets the number of points in this subset.
     */
    int count() const {
      return count_;
    }

    int &count() {
      return count_;
    }

    void Print() const {
      if(!is_leaf()) {
        printf("internal node: %d to %d: %d points total\n",
               begin_, begin_ + count_ - 1, count_);
      }
      else {
        printf("leaf node: %d to %d: %d points total\n",
               begin_, begin_ + count_ - 1, count_);
      }

      if(!is_leaf()) {
        left_->Print();
        right_->Print();
      }
    }
};
};
};

#endif
