/** @file general_spacetree.h
 *
 *  Generalized space partitioning tree.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_GENERAL_SPACETREE_H
#define CORE_TREE_GENERAL_SPACETREE_H

#include <boost/interprocess/offset_ptr.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/level.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/tracking_enum.hpp>
#include <deque>

#ifdef _OPENMP_
#include <omp.h>
#endif

#include <queue>
#include "core/parallel/distributed_tree_util.h"
#include "core/table/dense_matrix.h"
#include "core/table/memory_mapped_file.h"
#include "statistic.h"
#include "core/table/memory_mapped_file.h"

namespace core {
namespace table {
extern core::table::MemoryMappedFile *global_m_file_;
}
}

namespace core {
namespace tree {

/** @brief The utility to initialize old_from_new indices.
 */
template<typename IndexType>
class IndexInitializer {
  public:
    static void PostProcessOldFromNew(
      const core::table::DenseMatrix &matrix_in, IndexType *old_from_new_out);

    static void OldFromNew(
      const core::table::DenseMatrix &matrix_in,
      int rank_in,
      IndexType *old_from_new_out);

    static void NewFromOld(
      const core::table::DenseMatrix &matrix_in,
      IndexType *old_from_new_in,
      int *new_from_old_out);
};

/** @brief The template specialization of IndexInitializer for the
 *         distributed table setting.
 */
template<>
class IndexInitializer< std::pair<int, std::pair<int, int> > > {
  public:
    static void PostProcessOldFromNew(
      const core::table::DenseMatrix &matrix_in,
      std::pair<int, std::pair<int, int> > *old_from_new_out) {
      for(int i = 0; i < matrix_in.n_cols(); i++) {
        old_from_new_out[i].second.second = i;
      }
    }

    static void OldFromNew(
      const core::table::DenseMatrix &matrix_in, int rank_in,
      std::pair<int, std::pair<int, int> > *old_from_new_out) {
      for(int i = 0; i < matrix_in.n_cols(); i++) {
        old_from_new_out[i] = std::pair<int, std::pair<int, int> >(
                                rank_in, std::pair<int, int>(i, 0));
      }
    }

    static void NewFromOld(
      const core::table::DenseMatrix &matrix_in,
      std::pair<int, std::pair<int, int> > *old_from_new_in,
      int *new_from_old_out) {
      for(int i = 0; i < matrix_in.n_cols(); i++) {
        new_from_old_out[old_from_new_in[i].second.second] = i;
      }
    }
};

/** @brief The template specialization of IndexInitializer for the
 *         ordinary table setting.
 */
template<>
class IndexInitializer< int > {
  public:
    static void PostProcessOldFromNew(
      const core::table::DenseMatrix &matrix_in,
      int *old_from_new_out) {

      // Do nothing.
    }

    static void OldFromNew(
      const core::table::DenseMatrix &matrix_in, int rank_in,
      int *old_from_new_out) {
      for(int i = 0; i < matrix_in.n_cols(); i++) {
        old_from_new_out[i] = i;
      }
    }

    static void NewFromOld(
      const core::table::DenseMatrix &matrix_in,
      int *old_from_new_in,
      int *new_from_old_out) {
      for(int i = 0; i < matrix_in.n_cols(); i++) {
        new_from_old_out[old_from_new_in[i]] = i;
      }
    }
};

/** @brief The general binary space partioning tree.
 */
template < class IncomingTreeSpecType >
class GeneralBinarySpaceTree {
  private:

    // For BOOST serialization.
    friend class boost::serialization::access;

  public:

    /** @brief The specification of the tree.
     */
    typedef IncomingTreeSpecType TreeSpecType;

    /** @brief The bound type used in the tree.
     */
    typedef typename TreeSpecType::BoundType BoundType;

    /** @brief The tree type.
     */
    typedef core::tree::GeneralBinarySpaceTree<TreeSpecType> TreeType;

    /** @brief The statistics type stored in the node.
     */
    typedef typename TreeSpecType::StatisticType StatisticType;

  private:

    /** @brief The class used for prioritizing a node by its size.
     */
    class PrioritizeNodesBySize_:
      public std::binary_function <
        const TreeType *, const TreeType *, bool > {
      public:
        bool operator()(
          const TreeType *a, const TreeType *b) const {
          return a->count() < b->count();
        }
    };

    /** @brief The bound for the node.
     */
    BoundType bound_;

    /** @brief The beginning index.
     */
    int begin_;

    /** @brief The number of points contained within the node.
     */
    int count_;

    /** @brief The statistics for the points owned within the node.
     */
    StatisticType stat_;

    /** @brief The pointer to the left node.
     */
    boost::interprocess::offset_ptr<GeneralBinarySpaceTree> left_;

    /** @brief The pointer to the right node.
     */
    boost::interprocess::offset_ptr<GeneralBinarySpaceTree> right_;

  private:

    /** @brief Finds the depth of the tree rooted at a given node.
     */
    int depth_private_(const GeneralBinarySpaceTree *node) const {
      if(node->is_leaf()) {
        return 1;
      }
      else {
        return 1 + std::max(
                 depth_private_(node->left()), depth_private_(node->right()));
      }
    }

    template<typename MetricType>
    static void SplitTreeBreadthFirstPostProcess_(
      const MetricType &metric_in,
      core::table::DenseMatrix& matrix,
      TreeType *node,
      int serial_depth_first_limit) {

      if(! node->is_leaf()) {
        if(node->count() > serial_depth_first_limit) {
          SplitTreeBreadthFirstPostProcess_(
            metric_in, matrix, node->left(), serial_depth_first_limit);
          SplitTreeBreadthFirstPostProcess_(
            metric_in, matrix, node->right(), serial_depth_first_limit);
        }

        TreeSpecType::CombineBounds(
          metric_in, matrix, node, node->left(), node->right());
      }
    }

  public:

    /** @brief Finds the depth of the tree at this node.
     */
    int depth() const {
      return depth_private_(this);
    }

    /** @brief Gets the list of begin and count pair for at most $k$
     *         frontier nodes of the subtree rooted at this node.
     */
    void get_frontier_nodes(
      int max_num_points_subtree_size,
      std::vector < TreeType * > *frontier_nodes) const {

      // The priority queue type.
      typedef std::priority_queue <
      const TreeType *,
            std::vector<const TreeType *>,
            typename TreeType::PrioritizeNodesBySize_ > PriorityQueueType;
      PriorityQueueType queue;
      queue.push(const_cast<TreeType *>(this));

      while(queue.size() > 0) {
        TreeType *dequeued_node = const_cast<TreeType *>(queue.top());
        queue.pop();
        if(dequeued_node->is_leaf() ||
            dequeued_node->count() <= max_num_points_subtree_size) {
          frontier_nodes->push_back(dequeued_node);
        }
        else {
          queue.push(dequeued_node->left());
          queue.push(dequeued_node->right());
        }
      }
    }

    void get_frontier_nodes_bounded_by_number(
      int max_num_nodes,
      std::vector < TreeType * > *frontier_nodes) const {

      // The priority queue type.
      typedef std::priority_queue <
      const TreeType *,
            std::vector<const TreeType *>,
            typename TreeType::PrioritizeNodesBySize_ > PriorityQueueType;
      PriorityQueueType queue;
      queue.push(const_cast<TreeType *>(this));

      while(static_cast<int>(queue.size()) +
            static_cast<int>(frontier_nodes->size()) <
            max_num_nodes) {
        TreeType *dequeued_node = const_cast<TreeType *>(queue.top());
        queue.pop();
        if(dequeued_node->is_leaf()) {
          frontier_nodes->push_back(dequeued_node);
        }
        else {
          queue.push(dequeued_node->left());
          queue.push(dequeued_node->right());
        }
      }
      while(queue.size() > 0) {
        TreeType *dequeued_node = const_cast<TreeType *>(queue.top());
        frontier_nodes->push_back(dequeued_node);
        queue.pop();
      }
    }

    /** @brief A method for copying a node without its children.
     */
    void CopyWithoutChildren(const GeneralBinarySpaceTree &node) {
      bound_.Copy(node.bound());
      begin_ = node.begin();
      count_ = node.count();
      stat_.Copy(node.stat());
      left_ = NULL;
      right_ = NULL;
    }

    /** @brief The assignment operator that copies a node without its
     *         children.
     */
    void operator=(const GeneralBinarySpaceTree &node) {
      CopyWithoutChildren(node);
    }

    /** @brief The copy constructor that copies a node without its
     *         children.
     */
    GeneralBinarySpaceTree(const GeneralBinarySpaceTree &node) {
      CopyWithoutChildren(node);
    }

    /** @brief A method for serializing a node. This method does not
     *         save the children.
     */
    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {
      ar & bound_;
      ar & begin_;
      ar & count_;
      ar & stat_;
    }

    /** @brief A method for unserializing a node. This does not
     *         recover its children though.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {
      ar & bound_;
      ar & begin_;
      ar & count_;
      ar & stat_;
      left_ = NULL;
      right_ = NULL;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    /** @brief The destructor.
     */
    ~GeneralBinarySpaceTree() {
      if(left_.get() != NULL) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(left_.get());
        }
        else {
          delete left_.get();
        }
        left_ = NULL;
      }
      if(right_.get() != NULL) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(right_.get());
        }
        else {
          delete right_.get();
        }
        right_ = NULL;
      }
    }

    /** @brief The default constructor.
     */
    GeneralBinarySpaceTree() {
      left_ = NULL;
      right_ = NULL;
      begin_ = -1;
      count_ = -1;
    }

    /** @brief Initializes the node with the beginning index and the
     *         count of the points inside.
     */
    void Init(int begin_in, int count_in) {
      begin_ = begin_in;
      count_ = count_in;
    }

    /** @brief Find a node in this tree by its begin and count.
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

    /** @brief Find a node in this tree by its begin and count.
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

    /** @brief Sets the left child of the node with the given node.
     */
    void set_left_child(
      const core::table::DenseMatrix& data, GeneralBinarySpaceTree *left_in) {
      left_ = left_in;
    }

    /** @brief Sets the right child of the node with the given node.
     */
    void set_right_child(
      const core::table::DenseMatrix& data, GeneralBinarySpaceTree *right_in) {
      right_ = right_in;
    }

    /** @brief Sets the child nodes of the node with the given
     *         children.
     */
    void set_children(
      const core::table::DenseMatrix& data, GeneralBinarySpaceTree *left_in,
      GeneralBinarySpaceTree *right_in) {

      left_ = left_in;
      right_ = right_in;
    }

    /** @brief Gets the bound.
     */
    const BoundType &bound() const {
      return bound_;
    }

    /** @brief Gets the bound.
     */
    BoundType &bound() {
      return bound_;
    }

    /** @brief Returns the statistics.
     */
    const StatisticType &stat() const {
      return stat_;
    }

    /** @brief Returns the statistics.
     */
    StatisticType &stat() {
      return stat_;
    }

    /** @brief Returns whether the node is leaf or not.
     */
    bool is_leaf() const {
      return !left_;
    }

    /** @brief Gets the left branch of the tree.
     */
    const GeneralBinarySpaceTree *left() const {
      return left_.get();
    }

    /** @brief Gets the left branch.
     */
    GeneralBinarySpaceTree *left() {
      return left_.get();
    }

    /** @brief Gets the right branch.
     */
    const GeneralBinarySpaceTree *right() const {
      return right_.get();
    }

    /** @brief Gets the right branch.
     */
    GeneralBinarySpaceTree *right() {
      return right_.get();
    }

    /** @brief Gets the index of the begin point of this subset.
     */
    int begin() const {
      return begin_;
    }

    /** @brief Gets the index one beyond the last index in the series.
     */
    int end() const {
      return begin_ + count_;
    }

    /** @brief Gets the number of points in this subset.
     */
    int count() const {
      return count_;
    }

    /** @brief Recursively prints the tree underneath.
     */
    void Print() const {
      if(!is_leaf()) {
        printf("internal node: %d to %d: %d points total\n",
               begin_, begin_ + count_ - 1, count_);
      }
      else {
        printf("leaf node: %d to %d: %d points total\n",
               begin_, begin_ + count_ - 1, count_);
      }
      bound_.Print();

      if(!is_leaf()) {
        left_->Print();
        right_->Print();
      }
    }

    /** @brief Attempts to split a kd-tree node and reshuffles the
     *         data accordingly and creates two child nodes.
     */
    template<typename MetricType, typename TreeType, typename IndexType>
    static bool AttemptSplitting(
      const MetricType &metric_in,
      core::table::DenseMatrix &matrix,
      core::table::DenseMatrix &weights,
      TreeType *node, TreeType **left,
      TreeType **right, int leaf_size,
      IndexType *old_from_new,
      core::table::MemoryMappedFile *m_file_in) {

      bool can_cut = TreeSpecType::AttemptSplitting(
                       metric_in, matrix, weights, node, left, right,
                       leaf_size, old_from_new, m_file_in);

      // In case splitting fails, then free memory.
      if(! can_cut) {
        if((*left) != NULL) {
          if(m_file_in != NULL) {
            m_file_in->DestroyPtr(* left) ;
          }
          else {
            delete(*left);
            *left = NULL;
          }
        }
        if((*right) != NULL) {
          if(m_file_in != NULL) {
            m_file_in->DestroyPtr(*right);
          }
          else {
            delete(*right);
            *right = NULL;
          }
        }
      }

      return can_cut;
    }

    /** @brief Attempts to split a set of points and re-distribute
     *         them across a set of MPI processes.
     */
    template<typename MetricType>
    static bool AttemptSplitting(
      boost::mpi::communicator &comm,
      const MetricType &metric_in,
      const BoundType &bound,
      const core::table::DenseMatrix &matrix_in,
      std::vector< std::vector<int> > *assigned_point_indices,
      std::vector<int> *membership_counts_per_process) {

      // Perform tree-spec specific task first.
      int left_count;
      std::deque<bool> left_membership;
      if(!  TreeSpecType::AttemptSplitting(
            comm, metric_in, bound, matrix_in,
            &left_count, &left_membership)) {
        return false;
      }

      // Post-process the assignments here.
      // The assigned point indices per process and per-process counts
      // will be outputted.
      assigned_point_indices->resize(comm.size());
      membership_counts_per_process->resize(comm.size());

      // Loop through the membership vectors and assign to the right
      // process partner.
      int left_destination, right_destination;
      core::parallel::DistributedTreeExtraUtil::left_and_right_destinations(
        comm, &left_destination, &right_destination, (int *) NULL);
      for(unsigned int i = 0; i < left_membership.size(); i++) {
        if(left_membership[i]) {
          (*assigned_point_indices)[left_destination].push_back(i);
          (*membership_counts_per_process)[left_destination]++;
        }
        else {
          (*assigned_point_indices)[right_destination].push_back(i);
          (*membership_counts_per_process)[right_destination]++;
        }
      }

      // Check whether the local assignment is ok.
      bool assignment_is_ok =
        (comm.rank() < comm.size() / 2) ?
        (left_count > 0) : (left_count < matrix_in.n_cols());

      // Do an all-reduction to check whether the assignment is ok.
      bool global_assignment_is_ok = true;
      boost::mpi::all_reduce(
        comm, assignment_is_ok, global_assignment_is_ok,
        std::logical_and<bool>());

      return global_assignment_is_ok;
    }

#ifdef _OPENMP_

    /** @brief Recursively splits a given node creating its children
     *         in a breadth-first manner. This function is better for
     *         multi-threading.
     */
    template<typename MetricType, typename IndexType>
    static void SplitTreeBreadthFirst(
      const MetricType &metric_in,
      core::table::DenseMatrix& matrix,
      core::table::DenseMatrix &weights,
      TreeType *node,
      int leaf_size,
      int max_num_leaf_nodes,
      int *current_num_leaf_nodes,
      IndexType *old_from_new,
      int *num_nodes) {

      // Get the number of available OpenMP threads.
      int num_threads = omp_get_max_threads();

      // If the number of points fall under this threshold, then we
      // call the serial depth-first.
      int serial_depth_first_limit = matrix.n_cols() / num_threads;

      // The list of active nodes on the current level.
      std::deque< TreeType * > active_nodes;
      active_nodes.push_back(node);

      do {

        // The number of active nodes on the current round.
        int num_items_on_level = static_cast<int>(active_nodes.size());

#pragma omp parallel for
        for(int i = 0; i < num_items_on_level; i++) {

          // Set the number of threads in the inner level.
          omp_set_num_threads(std::max(1, num_threads / num_items_on_level));

          // If the i-th node contains more than the depth first limit
          // threshold, then split the node and put the children nodes
          // at the end.
          if(active_nodes[i]->count() > serial_depth_first_limit) {
            if(active_nodes[i]->count() > leaf_size) {

              TreeType *left = NULL;
              TreeType *right = NULL;
              bool can_cut = AttemptSplitting(
                               metric_in, matrix, weights,
                               active_nodes[i], &left, &right,
                               leaf_size, old_from_new,
                               core::table::global_m_file_);

              // Set children information appropriately.
              active_nodes[i]->set_children(matrix, left, right);

              if(can_cut) {
#pragma omp critical
                {
                  // Start of critical section.
                  active_nodes.push_back(active_nodes[i]->left());
                  active_nodes.push_back(active_nodes[i]->right());
                  (*current_num_leaf_nodes)++;
                  (*num_nodes) += 2;
                }  // End of critical section.
              }
              else {
                TreeSpecType::MakeLeafNode(
                  metric_in, matrix, active_nodes[i]->begin(),
                  active_nodes[i]->count(), &(active_nodes[i]->bound()));
              }
            } // end of splitting a non-leaf node.
            else {
              TreeSpecType::MakeLeafNode(
                metric_in, matrix, active_nodes[i]->begin(),
                active_nodes[i]->count(), &(active_nodes[i]->bound()));
            }
          }
          else {
            SplitTreeDepthFirst(
              metric_in, matrix, weights, active_nodes[i],
              leaf_size, max_num_leaf_nodes, current_num_leaf_nodes,
              old_from_new, num_nodes);
          }
        } // end of omp parallel for.

        // Clean up the previous level.
        for(int i = 0; i < num_items_on_level; i++) {
          active_nodes.pop_front();
        }
      }
      while(active_nodes.size() > 0);

      // Correct the bounds later. Start from the root again.
      SplitTreeBreadthFirstPostProcess_(
        metric_in, matrix, node, serial_depth_first_limit);
    }

#endif

    /** @brief Recursively splits a given node creating its children
     *         in a depth-first manner.
     */
    template<typename MetricType, typename IndexType>
    static void SplitTreeDepthFirst(
      const MetricType &metric_in,
      core::table::DenseMatrix& matrix,
      core::table::DenseMatrix &weights,
      TreeType *node,
      int leaf_size,
      int max_num_leaf_nodes,
      int *current_num_leaf_nodes,
      IndexType *old_from_new,
      int *num_nodes) {

      TreeType *left = NULL;
      TreeType *right = NULL;

      // If the node is just too small or we have reached the maximum
      // number of leaf nodes allowed, then do not split.
      bool constructed_leaf_node = false;
#pragma omp critical
      {
        constructed_leaf_node =
          (node->count() <= leaf_size ||
           (*current_num_leaf_nodes) >= max_num_leaf_nodes);
      }

      if(constructed_leaf_node) {
        TreeSpecType::MakeLeafNode(
          metric_in, matrix, node->begin(), node->count(), &(node->bound()));
      }

      // Otherwise, attempt to split.
      else {
        bool can_cut = AttemptSplitting(
                         metric_in, matrix, weights, node, &left, &right,
                         leaf_size, old_from_new, core::table::global_m_file_);

        if(can_cut) {
#pragma omp critical
          {
            (*current_num_leaf_nodes)++;
            (*num_nodes) = (*num_nodes) + 2;
          }
          SplitTreeDepthFirst(
            metric_in, matrix, weights, left, leaf_size, max_num_leaf_nodes,
            current_num_leaf_nodes, old_from_new, num_nodes);
          SplitTreeDepthFirst(
            metric_in, matrix, weights, right, leaf_size, max_num_leaf_nodes,
            current_num_leaf_nodes, old_from_new, num_nodes);
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

    /** @brief Creates a tree from data.
     */
    template<typename MetricType, typename IndexType>
    static TreeType *MakeTree(
      const MetricType &metric_in,
      core::table::DenseMatrix& matrix,
      core::table::DenseMatrix &weights,
      int leaf_size,
      IndexType *old_from_new,
      int *new_from_old,
      int max_num_leaf_nodes = std::numeric_limits<int>::max(),
      int *num_nodes = NULL,
      int rank_in = 0) {

      // Enable OpenMP nested parallelization.
#ifdef _OPENMP_
      omp_set_nested(true);
#endif

      // Postprocess old_from_new indices before building the tree.
      IndexInitializer<IndexType>::PostProcessOldFromNew(matrix, old_from_new);

      TreeType *node = (core::table::global_m_file_) ?
                       core::table::global_m_file_->Construct<TreeType>() :
                       new TreeType();

      int num_nodes_in = 1;
      node->Init(0, matrix.n_cols());
      node->bound().Init(matrix.n_rows());
      TreeSpecType::FindBoundFromMatrix(
        metric_in, matrix, 0, matrix.n_cols(), &node->bound());

      int current_num_leaf_nodes = 1;

#ifdef _OPENMP_
      SplitTreeBreadthFirst(
        metric_in, matrix, weights, node, leaf_size, max_num_leaf_nodes,
        &current_num_leaf_nodes, old_from_new, &num_nodes_in);
#else
      SplitTreeDepthFirst(
        metric_in, matrix, weights, node, leaf_size, max_num_leaf_nodes,
        &current_num_leaf_nodes, old_from_new, &num_nodes_in);
#endif

      if(num_nodes) {
        *num_nodes = num_nodes_in;
      }

      // Finalize the new_from_old mapping from old_from_new mapping.
      IndexInitializer<IndexType>::NewFromOld(
        matrix, old_from_new, new_from_old);
      return node;
    }

    /** @brief Reshuffles the matrix such that the left points occupy
     *         the left contiguous blocks of the matrix and right
     *         points occupy the right contiguous blocks of the
     *         matrix.
     */
    template<typename MetricType, typename IndexType>
    static int MatrixPartition(
      const MetricType &metric_in,
      core::table::DenseMatrix &matrix,
      core::table::DenseMatrix &weights,
      int first, int count,
      BoundType &left_bound, BoundType &right_bound,
      IndexType *old_from_new) {

      int end = first + count;
      int left_count = 0;

      std::deque<bool> left_membership;
      left_membership.resize(count);

      // Compute the required memberships.
      TreeSpecType::ComputeMemberships(
        metric_in, matrix, first, end, left_bound, right_bound,
        &left_count, &left_membership);

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
          break;
        }

        // Swap the left vector with the right vector.
        matrix.swap_cols(left, right);
        weights.swap_cols(left, right);
        std::swap(
          left_membership[left - first], left_membership[right - first]);

        if(old_from_new) {
          std::swap(old_from_new[left], old_from_new[right]);
        }
        right--;
      }

      return left_count;
    }
};
}
}

namespace boost {
namespace serialization {
template<>
template<typename IncomingTreeSpecType>
struct tracking_level <
    core::tree::GeneralBinarySpaceTree<IncomingTreeSpecType> > {
  typedef mpl::integral_c_tag tag;
  typedef mpl::int_< boost::serialization::track_never > type;
  BOOST_STATIC_CONSTANT(
    int,
    value = tracking_level::type::value
  );
  BOOST_STATIC_ASSERT((
                        mpl::greater <
                        implementation_level< core::tree::GeneralBinarySpaceTree<IncomingTreeSpecType> >,
                        mpl::int_<primitive_type>
                        >::value
                      ));
};
}
}

#endif
