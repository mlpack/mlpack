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
#include <deque>
#include "core/table/dense_matrix.h"
#include "core/table/memory_mapped_file.h"
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
template < class TreeSpecType >
class GeneralBinarySpaceTree {
  public:
    typedef typename TreeSpecType::BoundType BoundType;

    typedef core::tree::GeneralBinarySpaceTree<TreeSpecType> TreeType;

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

    static void SplitTree(
      const core::metric_kernels::AbstractMetric &metric_in,
      core::table::DenseMatrix& matrix,
      TreeType *node,
      int leaf_size,
      int max_num_leaf_nodes,
      int *current_num_leaf_nodes,
      std::vector<int> *old_from_new,
      int *num_nodes,
      core::table::MemoryMappedFile *m_file_in) {

      TreeType *left = NULL;
      TreeType *right = NULL;

      // If the node is just too small or we have reached the maximum
      // number of leaf nodes allowed, then do not split.
      if(node->count() < leaf_size ||
          (*current_num_leaf_nodes) >= max_num_leaf_nodes) {
        TreeSpecType::MakeLeafNode(
          metric_in, matrix, node->begin(), node->count(), &(node->bound()));
      }

      // Otherwise, attempt to split.
      else {
        bool can_cut = TreeSpecType::AttemptSplitting(
                         metric_in, matrix, node, &left, &right,
                         leaf_size, old_from_new, m_file_in);

        if(can_cut) {
          (*current_num_leaf_nodes)++;
          (*num_nodes) = (*num_nodes) + 2;
          SplitTree(
            metric_in, matrix, left, leaf_size, max_num_leaf_nodes,
            current_num_leaf_nodes, old_from_new, num_nodes, m_file_in);
          SplitTree(
            metric_in, matrix, right, leaf_size, max_num_leaf_nodes,
            current_num_leaf_nodes, old_from_new, num_nodes, m_file_in);
          TreeSpecType::CombineBounds(metric_in, matrix, node, left, right);
        }
        else {
          TreeSpecType::MakeLeafNode(
            metric_in, matrix, node->begin(), node->count(), &(node->bound()));
        }
      }

      // Set children information appropriately.
      node->set_children(matrix, left, right);
    }

    /**
     * Creates a tree from data.
     *
     * This requires you to pass in two unitialized ArrayLists which
     * will contain index mappings so you can account for the
     * re-ordering of the matrix.
     *
     * @param metric_in the metric to be used.
     * @param matrix data where each column is a point, WHICH WILL BE RE-ORDERED
     * @param leaf_size the maximum points in a leaf
     * @param max_num_leaf_nodes the number of maximum leaf nodes this tree
     *        should have.
     * @param old_from_new pointer to an unitialized vector; it will map
     *        new indices to original
     * @param new_from_old pointer to an unitialized vector; it will map
     *        original indexes to new indices
     * @param num_nodes the number of nodes constructed in total.
     */
    static TreeType *MakeTree(
      const core::metric_kernels::AbstractMetric &metric_in,
      core::table::DenseMatrix& matrix, int leaf_size,
      int max_num_leaf_nodes = std::numeric_limits<int>::max(),
      std::vector<int> *old_from_new = NULL,
      std::vector<int> *new_from_old = NULL,
      int *num_nodes = NULL,
      core::table::MemoryMappedFile *m_file_in = NULL) {

      TreeType *node = (m_file_in) ?
                       (TreeType *) m_file_in->Allocate(sizeof(TreeType)) :
                       new TreeType();
      std::vector<int> *old_from_new_ptr;

      if(old_from_new) {
        old_from_new->resize(matrix.n_cols());

        for(int i = 0; i < matrix.n_cols(); i++) {
          (*old_from_new)[i] = i;
        }
        old_from_new_ptr = old_from_new;
      }
      else {
        old_from_new_ptr = NULL;
      }

      int num_nodes_in = 1;
      node->Init(0, matrix.n_cols());
      node->bound().center().Init(matrix.n_rows());
      int current_num_leaf_nodes = 1;
      SplitTree(
        metric_in, matrix, node, leaf_size, max_num_leaf_nodes,
        &current_num_leaf_nodes, old_from_new_ptr, &num_nodes_in, m_file_in);

      if(num_nodes) {
        *num_nodes = num_nodes_in;
      }
      if(new_from_old) {
        new_from_old->resize(matrix.n_cols());
        for(int i = 0; i < matrix.n_cols(); i++) {
          (*new_from_old)[(*old_from_new)[i]] = i;
        }
      }
      return node;
    }

    static int MatrixPartition(
      const core::metric_kernels::AbstractMetric &metric_in,
      core::table::DenseMatrix& matrix, int first, int count,
      BoundType &left_bound, BoundType &right_bound,
      std::vector<int> *old_from_new) {

      int end = first + count;
      int left_count = 0;

      std::deque<bool> left_membership;
      left_membership.resize(count);

      for(int left = first; left < end; left++) {

        // Make alias of the current point.
        core::table::DenseConstPoint point;
        matrix.MakeColumnVector(left, &point);

        // Compute the distances from the two pivots.
        double distance_from_left_pivot =
          metric_in.Distance(point, left_bound.center());
        double distance_from_right_pivot =
          metric_in.Distance(point, right_bound.center());

        // We swap if the point is further away from the left pivot.
        if(distance_from_left_pivot > distance_from_right_pivot) {
          left_membership[left - first] = false;
        }
        else {
          left_membership[left - first] = true;
          left_count++;
        }
      }

      int left = first;
      int right = first + count - 1;

      // At any point: everything < left is correct
      //               everything > right is correct
      for(;;) {
        while(left_membership[left - first] && left <= right) {
          left++;
        }

        while(!left_membership[right - first] && left <= right) {
          right--;
        }

        if(left > right) {

          // left == right + 1
          break;
        }

        // Swap the left vector with the right vector.
        matrix.swap_cols(left, right);
        std::swap(left_membership[left - first], left_membership[right - first]);

        if(old_from_new) {
          std::swap((*old_from_new)[left], (*old_from_new)[right]);
        }
        right--;
      }

      return left_count;
    }
};
};
};

#endif
