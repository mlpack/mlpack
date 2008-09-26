#ifndef GEN_HYPERCUBE_TREE_H
#define GEN_HYPERCUBE_TREE_H

#include "fastlib/fastlib.h"
#include "fastlib/tree/statistic.h"

namespace proximity {

  template<class TStatistic=EmptyStatistic<Matrix> >
  class GenHypercubeTree {

   public:

    typedef DHrectBound<2> Bound;
    typedef Matrix Dataset;
    typedef TStatistic Statistic;
    
    Bound bound_;
    ArrayList<GenHypercubeTree *> children_;
    ArrayList<index_t> begin_;
    ArrayList<index_t> count_;
    index_t total_count_;
    index_t level_;
    index_t node_index_;
    Statistic stat_;
    
    OT_DEF(GenHypercubeTree) {
      OT_MY_OBJECT(bound_);
      OT_MY_OBJECT(children_);
      OT_MY_OBJECT(begin_);
      OT_MY_OBJECT(count_);
      OT_MY_OBJECT(total_count_);
      OT_MY_OBJECT(level_);
      OT_MY_OBJECT(node_index_);
      OT_MY_OBJECT(stat_);
    }
    
   public:
    
    const Statistic& stat() const {
      return stat_;
    }
    
    Statistic& stat() {
      return stat_;
    }

    /** @brief Tests whether the current node is a leaf node
     *         (childless).
     *
     *  @return true if childless, false otherwise.
     */
    bool is_leaf() {
      return children_ == NULL;
    }

    void Init(index_t number_of_particle_sets) {
      begin_.Init(number_of_particle_sets);
      count_.Init(number_of_particle_sets);
      total_count_ = 0;
      children_.Init();
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

    GenHypercubeTree *get_child(int index) const {
      return children_[index];
    }

    void set_level(index_t level) {
      level_ = level;
    }

    GenHypercubeTree *AllocateNewChild(index_t number_of_particle_sets,
				       index_t node_index_in) {
      
      GenHypercubeTree *new_node = new GenHypercubeTree();
      children_.PushBackCopy(new_node);
      new_node->Init(number_of_particle_sets);
      node_index_ = node_index_in;

      return new_node;
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
    
    index_t node_index() const {
      return node_index_;
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

    index_t level() const {
      return level_;
    }

    /**
     * Gets the number of children.
     */
    index_t num_children() const {
      return children_.size();
    }

  };

#include "gen_hypercube_tree_impl.h"

  /** @brief Creates a generalized hypercube tree (high-dimensional
   * generalization of quad-tree, octree) from data.
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
   * @param leaf_size the maximum points in a leaf
   *
   * @param old_from_new pointer to an unitialized arraylist; it
   * will map new indices to original
   *
   * @param new_from_old pointer to an unitialized arraylist; it
   * will map original indexes to new indices
   */
  template<typename TStatistic>
  GenHypercubeTree<TStatistic> *MakeGenHypercubeTree
  (ArrayList<Matrix *> &matrices, index_t leaf_size,
   ArrayList< ArrayList<GenHypercubeTree<TStatistic> *> > *nodes_in_each_level,
   ArrayList< ArrayList<index_t> > *old_from_new = NULL,
   ArrayList< ArrayList<index_t> > *new_from_old = NULL) {
    
    GenHypercubeTree<TStatistic> *node = new GenHypercubeTree<TStatistic>();
    
    if (old_from_new) {
      old_from_new->Init(matrices.size());

      for(index_t j = 0; j < matrices.size(); j++) {
	(*old_from_new)[j].Init(matrices[j]->n_cols());
	
	for (index_t i = 0; i < matrices[j]->n_cols(); i++) {
	  (*old_from_new)[j][i] = i;
	}
      }
    }
    
    // Initialize the global list of nodes.
    nodes_in_each_level->Init(1);    

    // Initialize the root node.
    node->Init(matrices.size());
    for(index_t i = 0; i < matrices.size(); i++) {
      node->Init(i, 0, matrices[i]->n_cols());
    }
    
    // Make the tightest cube bounding box you can fit around the
    // current set of points.
    tree_gen_hypercube_tree_private::ComputeBoundingHypercube(matrices, node);

    // Put the root node into the initial list of level 0.
    ((*nodes_in_each_level)[0]).Init();
    ((*nodes_in_each_level)[0]).PushBackCopy(node);

    tree_gen_hypercube_tree_private::SplitGenHypercubeTree
      (matrices, node, leaf_size, nodes_in_each_level, old_from_new, 0);

    // Index shuffling business...
    if (new_from_old) {
      new_from_old->Init(matrices.size());

      for(index_t j = 0; j < matrices.size(); j++) {
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
