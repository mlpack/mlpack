/** @file general_spacetree.h
 *
 *  Generalized space partitioning tree.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_GENERAL_SPACETREE_H
#define CORE_TREE_GENERAL_SPACETREE_H

#include <boost/serialization/split_free.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/level.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/tracking_enum.hpp>
#include <deque>
#include <queue>
#include "core/table/dense_matrix.h"
#include "core/table/memory_mapped_file.h"
#include "statistic.h"
#include <boost/interprocess/offset_ptr.hpp>
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

    /** @brief A helper function for
     *         get_frontier_node_begin_count_pairs.
     */
    void get_frontier_node_begin_count_pairs_private_(
      const GeneralBinarySpaceTree *start_node,
      int max_num_frontier_nodes,
      int *num_frontier_nodes_encountered,
      std::vector <
      std::pair<int, int> > *frontier_node_begin_count_pairs,
      std::vector< BoundType > *bounds) const {

      // The priority queue type.
      typedef std::priority_queue <
      const TreeType *,
            std::vector<const TreeType *>,
            typename TreeType::PrioritizeNodesBySize_ > PriorityQueueType;
      PriorityQueueType queue;
      queue.push(start_node);

      while((*num_frontier_nodes_encountered) < max_num_frontier_nodes &&
            queue.size() > 0) {
        const TreeType *dequeued_node = queue.top();
        queue.pop();
        if(dequeued_node->is_leaf()) {
          frontier_node_begin_count_pairs->push_back(
            std::pair<int, int>(
              dequeued_node->begin(), dequeued_node->count()));
          bounds->resize(bounds->size() + 1);
          bounds->back().Copy(dequeued_node->bound());
        }
        else {
          if((*num_frontier_nodes_encountered) < max_num_frontier_nodes) {
            (*num_frontier_nodes_encountered)++;
            queue.push(dequeued_node->left());
            queue.push(dequeued_node->right());
          }
          else {
            frontier_node_begin_count_pairs->push_back(
              std::pair<int, int>(
                dequeued_node->begin(), dequeued_node->count()));
            bounds->resize(bounds->size() + 1);
            bounds->back().Copy(dequeued_node->bound());
          }
        }
      }
      while(
        static_cast<int>(frontier_node_begin_count_pairs->size()) <
        max_num_frontier_nodes && queue.size() > 0) {

        const TreeType *dequeued_node = queue.top();
        frontier_node_begin_count_pairs->push_back(
          std::pair<int, int>(
            dequeued_node->begin(), dequeued_node->count()));
        bounds->resize(bounds->size() + 1);
        bounds->back().Copy(dequeued_node->bound());
        queue.pop();
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
    void get_frontier_node_begin_count_pairs(
      int max_num_frontier_nodes,
      std::vector <
      std::pair<int, int> > *frontier_node_begin_count_pairs,
      std::vector< BoundType > *bounds) const {
      int num_frontier_nodes_encountered = 1;

      get_frontier_node_begin_count_pairs_private_(
        this, max_num_frontier_nodes,
        &num_frontier_nodes_encountered,
        frontier_node_begin_count_pairs, bounds);
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

    /** @brief Recursively splits a given node creating its children.
     */
    template<typename MetricType, typename IndexType>
    static void SplitTree(
      const MetricType &metric_in,
      core::table::DenseMatrix& matrix,
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
      if(node->count() <= leaf_size ||
          (*current_num_leaf_nodes) >= max_num_leaf_nodes) {
        TreeSpecType::MakeLeafNode(
          metric_in, matrix, node->begin(), node->count(), &(node->bound()));
      }

      // Otherwise, attempt to split.
      else {
        bool can_cut = TreeSpecType::AttemptSplitting(
                         metric_in, matrix, node, &left, &right,
                         leaf_size, old_from_new, core::table::global_m_file_);

        if(can_cut) {
          (*current_num_leaf_nodes)++;
          (*num_nodes) = (*num_nodes) + 2;
          SplitTree(
            metric_in, matrix, left, leaf_size, max_num_leaf_nodes,
            current_num_leaf_nodes, old_from_new, num_nodes);
          SplitTree(
            metric_in, matrix, right, leaf_size, max_num_leaf_nodes,
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
     *
     * This requires you to pass in two unitialized ArrayLists which
     * will contain index mappings so you can account for the
     * re-ordering of the matrix.
     */
    template<typename MetricType, typename IndexType>
    static TreeType *MakeTree(
      const MetricType &metric_in,
      core::table::DenseMatrix& matrix, int leaf_size,
      IndexType *old_from_new,
      int *new_from_old,
      int max_num_leaf_nodes = std::numeric_limits<int>::max(),
      int *num_nodes = NULL,
      int rank_in = 0) {

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
      SplitTree(
        metric_in, matrix, node, leaf_size, max_num_leaf_nodes,
        &current_num_leaf_nodes, old_from_new, &num_nodes_in);

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
      core::table::DenseMatrix& matrix, int first, int count,
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
