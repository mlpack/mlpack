#ifndef CFMM_TREE_H
#define CFMM_TREE_H

#include "fastlib/fastlib.h"

#include "cfmm_tree_impl.h"


namespace proximity {

// Forward declaration...
template<typename TStatistics> class CFmmTree;

template<typename TStatistics>
class CFmmWellSeparatedTree {

  public:

    /** @brief The type of statistics stored in each node.
     */
    typedef TStatistics Statistics;

    /** @brief The beginning index of the point for each dataset.
     */
    ArrayList<index_t> begin_;

    /** @brief The number of points contained in this node for each
     *         dataset.
     */
    ArrayList<index_t> count_;

    /** @brief The well-separated index for each dataset, i.e. the
     *         maximum WS index for the points for each group.
     */
    GenVector<int> well_separated_indices_;

    /** @brief The total number of points for all the datasets.
     */
    index_t total_count_;

    bool init_flag_;

    /** @brief The stored statistics for this node.
     */
    Statistics stat_;

    /** @brief The pointers to the children nodes.
     */
    ArrayList<CFmmTree<TStatistics> *> children_;

    /** @brief The pointer to the parent so that it can access the
     *         parent's bounding box information.
     */
    CFmmTree<TStatistics> *parent_;

  public:

    CFmmWellSeparatedTree() {
    }

    ~CFmmWellSeparatedTree() {
      if (children_.size() > 0) {
        for (index_t i = 0; i < children_.size(); i++) {
          delete children_[i];
        }
      }
    }

    index_t num_children() const {
      return children_.size();
    }

    /** @brief Gets the index of the begin point of this subset.
     */
    index_t begin(index_t particle_set_number) const {
      return begin_[particle_set_number];
    }

    /** @brief Gets the index one beyond the last index in the series.
     */
    index_t end(index_t particle_set_number) const {
      return begin_[particle_set_number] + count_[particle_set_number];
    }

    index_t count(index_t particle_set_number) const {
      return count_[particle_set_number];
    }

    index_t count() const {
      return total_count_;
    }

    CFmmTree<TStatistics> *get_child(int index) const {
      return children_[index];
    }

    /** @brief Tests whether it is a leaf node or not.
     */
    bool is_leaf() const {
      return children_.size() == 0;
    }

    void Init(index_t number_of_particle_sets, index_t dimension,
              CFmmTree<TStatistics> *parent_in) {
      begin_.Init(number_of_particle_sets);
      count_.Init(number_of_particle_sets);
      total_count_ = 0;
      well_separated_indices_.Init(number_of_particle_sets);
      well_separated_indices_.SetZero();
      children_.Init();
      parent_ = parent_in;
      init_flag_ = false;
    }

    void Init(index_t particle_set_number, index_t begin_in,
              index_t count_in) {

      begin_[particle_set_number] = begin_in;
      count_[particle_set_number] = count_in;
      total_count_ += count_in;
    }

    CFmmTree<TStatistics> *AllocateNewChild(index_t number_of_particle_sets,
                                            index_t dimension,
                                            unsigned int node_index_in) {

      CFmmTree<TStatistics> *new_node = new CFmmTree<TStatistics>();
      *(children_.PushBackRaw()) = new_node;

      new_node->Init(number_of_particle_sets, dimension, this);
      new_node->node_index_ = node_index_in;

      return new_node;
    }

    void Print() const {
      if (!is_leaf()) {
        printf("internal node: %d points total\n", total_count_);
        for (index_t i = 0; i < begin_.size(); i++) {
          printf("   set %d: %d to %d: %d points total\n", i,
                 begin_[i], begin_[i] + count_[i] - 1, count_[i]);
        }
        for (index_t c = 0; c < children_.size(); c++) {
          children_[c]->Print();
        }
      }
      else {
        printf("leaf node: %d points total\n", total_count_);
        for (index_t i = 0; i < begin_.size(); i++) {
          printf("   set %d: %d to %d: %d points total\n", i,
                 begin_[i], begin_[i] + count_[i] - 1, count_[i]);
        }
      }
    }
};

template<typename TStatistics>
class CFmmTree {

  public:

    /** @brief The bounding box type is the rectangle bound.
     */
    typedef DHrectBound<2> Bound;

    /** @brief The type of the dataset.
     */
    typedef Matrix Dataset;

    /** @brief The type of statistics stored in each node.
     */
    typedef TStatistics Statistics;

    /** @brief The bounding box.
     */
    Bound bound_;

    /** @brief The beginning index of the point for each dataset.
     */
    ArrayList<index_t> begin_;

    /** @brief The number of points contained in this node for each
     *         dataset.
     */
    ArrayList<index_t> count_;

    /** @brief The well-separated index for each dataset, i.e. the
     *         maximum WS index for the points for each group.
     */
    GenVector<int> well_separated_indices_;

    /** @brief The total number of points for all the datasets.
     */
    index_t total_count_;

    /** @brief The current level of the tree.
     */
    index_t level_;

    /** @brief The global index of the node.
     */
    unsigned int node_index_;

    /** @brief The stored statistics for this node.
     */
    Statistics stat_;

    bool init_flag_;

    /** @brief The divided group based on the well-separated
     *         indices. This generalizes the 2-way branchings used in
     *         the CFMM paper.
     */
    ArrayList<CFmmWellSeparatedTree<TStatistics> *>
    partitions_based_on_ws_indices_;

    /** @brief The parent partition that owns this node. This is NULL
     *         for the root node.
     */
    CFmmWellSeparatedTree<Statistics> *parent_;

  public:

    CFmmTree() {
    }

    ~CFmmTree() {
      if (partitions_based_on_ws_indices_.size() > 0) {

        for (index_t i = 0; i < partitions_based_on_ws_indices_.size(); i++) {
          delete partitions_based_on_ws_indices_[i];
        }
      }

    }

    const Statistics& stat() const {
      return stat_;
    }

    Statistics& stat() {
      return stat_;
    }

    /** @brief Tests whether the current node is a leaf node
     *         (childless).
     */
    bool is_leaf() const {
      if (partitions_based_on_ws_indices_.size() > 0) {
        bool flag = true;
        for (index_t i = 0; i < partitions_based_on_ws_indices_.size() && flag;
             i++) {
          flag = flag && (partitions_based_on_ws_indices_[i]->is_leaf());
        }
        return flag;
      }
      else {
        return true;
      }
    }

    void Init(index_t number_of_particle_sets, index_t dimension,
              CFmmWellSeparatedTree<Statistics> *parent_in) {

      begin_.Init(number_of_particle_sets);
      count_.Init(number_of_particle_sets);
      total_count_ = 0;
      node_index_ = 0;
      partitions_based_on_ws_indices_.Init();
      well_separated_indices_.Init(number_of_particle_sets);
      well_separated_indices_.SetZero();
      parent_ = parent_in;
      init_flag_ = false;
    }

    void Init(index_t particle_set_number, index_t begin_in,
              index_t count_in) {

      begin_[particle_set_number] = begin_in;
      count_[particle_set_number] = count_in;
      total_count_ += count_in;
    }

    double side_length() const {
      const DRange &range = bound_.get(0);
      return range.hi - range.lo;
    }

    const Bound& bound() const {
      return bound_;
    }

    Bound& bound() {
      return bound_;
    }

    /**
     * Gets the index of the begin point of this subset.
     */
    index_t begin(index_t particle_set_number) const {
      return begin_[particle_set_number];
    }

    /**
     * Gets the index one beyond the last index in the series.
     */
    index_t end(index_t particle_set_number) const {
      return begin_[particle_set_number] + count_[particle_set_number];
    }

    unsigned int node_index() const {
      return node_index_;
    }

    void set_level(index_t level) {
      level_ = level;
    }

    index_t level() {
      return level_;
    }

    /**
     * Gets the number of points in this subset.
     */
    index_t count(index_t particle_set_number) const {
      return count_[particle_set_number];
    }

    index_t count() const {
      return total_count_;
    }

    CFmmWellSeparatedTree<TStatistics> *AllocateNewPartition() {

      CFmmWellSeparatedTree<TStatistics> *new_partition =
        new CFmmWellSeparatedTree<TStatistics>();
      *(partitions_based_on_ws_indices_.PushBackRaw()) = new_partition;

      new_partition->Init(begin_.size(), bound_.dim(), this);
      return new_partition;
    }

    void Print() const {
      if (!is_leaf()) {
        printf("internal node: %d points total on level %d\n", total_count_,
               level_);
        printf("  bound:\n");
        for (index_t i = 0; i < bound_.dim(); i++) {
          printf("%g %g\n", bound_.get(i).lo, bound_.get(i).hi);
        }
        for (index_t i = 0; i < begin_.size(); i++) {
          printf("   set %d: %d to %d: %d points total\n", i,
                 begin_[i], begin_[i] + count_[i] - 1, count_[i]);
        }
        for (index_t c = 0; c < partitions_based_on_ws_indices_.size(); c++) {
          partitions_based_on_ws_indices_[c]->Print();
        }
      }
      else {
        printf("leaf node: %d points total on level %d\n", total_count_,
               level_);
        printf("  bound:\n");
        for (index_t i = 0; i < bound_.dim(); i++) {
          printf("%g %g\n", bound_.get(i).lo, bound_.get(i).hi);
        }
        for (index_t i = 0; i < begin_.size(); i++) {
          printf("   set %d: %d to %d: %d points total\n", i,
                 begin_[i], begin_[i] + count_[i] - 1, count_[i]);
        }
      }
    }

};

/** @brief Creates a continuous FMM tree (high-dimensional
 *         generalization of quad-tree, octree) from data with
 *         additional partitions on a target value associated with
 *         each point. I imagine this could be a useful tree for
 *         general supervised learning.
 *
 * @experimental
 *
 * This requires you to pass in two unitialized ArrayLists which
 * will contain index mappings so you can account for the
 * re-ordering of the matrix.  (By unitialized I mean don't call
 * Init on it)
 *
 * @param matrix data where each column is a point, WHICH WILL BE
 * RE-ORDERED
 *
 * @param targets the labels associated with each data point for
 * each group, WHICH WILL BE RE-ORDERED.
 *
 * @param leaf_size the maximum points in a leaf
 *
 * @param min_required_ws_index the minimum value of well-separation
 * index required for multipole-multipole interaction.
 *
 * @param old_from_new pointer to an unitialized arraylist; it
 * will map new indices to original
 *
 * @param new_from_old pointer to an unitialized arraylist; it
 * will map original indexes to new indices
 */
template<typename TStatistic>
CFmmTree<TStatistic> *MakeCFmmTree
(ArrayList<Matrix *> &matrices, ArrayList<Vector *> &targets,
 index_t leaf_size, index_t min_required_ws_index, index_t max_tree_depth,
 ArrayList< ArrayList<CFmmTree<TStatistic> *> >
 *nodes_in_each_level,
 ArrayList< ArrayList<index_t> > *old_from_new = NULL,
 ArrayList< ArrayList<index_t> > *new_from_old = NULL) {


  proximity::CFmmTree<TStatistic> *node = new proximity::CFmmTree<TStatistic>();
  if (old_from_new) {
    old_from_new->Init(matrices.size());

    for (index_t j = 0; j < matrices.size(); j++) {
      (*old_from_new)[j].Init(matrices[j]->n_cols());

      for (index_t i = 0; i < matrices[j]->n_cols(); i++) {
        (*old_from_new)[j][i] = i;
      }
    }
  }

  // Initialize the global list of nodes.
  nodes_in_each_level->Init(max_tree_depth + 1);
  for (index_t i = 0; i < nodes_in_each_level->size(); i++) {
    ((*nodes_in_each_level)[i]).Init();
  }

  // Initialize the root node.
  node->Init(matrices.size(), matrices[0]->n_rows(),
             (proximity::CFmmWellSeparatedTree<TStatistic> *) NULL);
  node->set_level(0);
  for (index_t i = 0; i < matrices.size(); i++) {
    node->Init(i, 0, matrices[i]->n_cols());
  }

  // Make the tightest cube bounding box you can fit around the
  // current set of points.
  tree_cfmm_tree_private::ComputeBoundingHypercube(matrices, node);

  // Put the root node into the initial list of level 0.
  *(((*nodes_in_each_level)[0]).PushBackRaw()) = node;

  tree_cfmm_tree_private::SplitCFmmTree
  (matrices, targets, node, leaf_size, min_required_ws_index,
   max_tree_depth, nodes_in_each_level, old_from_new, 0);

  // Index shuffling business...
  if (new_from_old) {
    new_from_old->Init(matrices.size());

    for (index_t j = 0; j < matrices.size(); j++) {
      (*new_from_old)[j].Init(matrices[j]->n_cols());
      for (index_t i = 0; i < matrices[j]->n_cols(); i++) {
        (*new_from_old)[j][(*old_from_new)[j][i]] = i;
      }
    }
  }

  return node;

}

};

#endif
