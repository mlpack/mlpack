#ifndef GEN_HYPERCUBE_TREE_H
#define GEN_HYPERCUBE_TREE_H

#include "fastlib/fastlib.h"

namespace proximity {

  class GenHypercubeTree {

   public:

    typedef DHrectBound<2> Bound;
    typedef Matrix Dataset;
    
    Bound bound_;
    ArrayList<GenHypercubeTree *> *children_;
    index_t begin_;
    index_t count_;
    index_t level_;
    index_t node_index_;
    
    OT_DEF(GenHypercubeTree) {
      OT_MY_OBJECT(bound_);
      OT_PTR_NULLABLE(children_);
      OT_MY_OBJECT(begin_);
      OT_MY_OBJECT(count_);
      OT_MY_OBJECT(level_);
      OT_MY_OBJECT(node_index_);
    }
    
   public:

    void AllocateChildren(index_t dim) {
      children_ = new ArrayList<GenHypercubeTree *>();
      children_->Init(1 << dim);
      for(index_t i = 0; i < children_->size(); i++) {
	(*children_)[i] = NULL;
      }
    }
    
    void DeleteChildren() {
      if(children_ != NULL) {
	delete children_;
	children_ = NULL;
      }
    }

    void Init(index_t begin_in, index_t count_in, index_t node_index_in) {
      DEBUG_ASSERT(begin_ == BIG_BAD_NUMBER);
      children_ = NULL;
      begin_ = begin_in;
      count_ = count_in;
      node_index_ = node_index_in;
    }

    const Bound& bound() const {
      return bound_;
    }
    
    Bound& bound() {
      return bound_;
    }

    GenHypercubeTree *get_child(int index) const {
      return (*children_)[index];
    }

    void set_level(index_t level) {
      level_ = level;
    }

    void set_child(int code, index_t first, index_t count, 
		   index_t node_index_in) {
      (*children_)[code] = new GenHypercubeTree();
      ((*children_)[code])->Init(first, count, node_index_in);
    }

    /**
     * Gets the index of the begin point of this subset.
     */
    index_t begin() const {
      return begin_;
    }
    
    /**
     * Gets the index one beyond the last index in the series.
     */
    index_t end() const {
      return begin_ + count_;
    }
    
    index_t node_index() const {
      return node_index_;
    }

    /**
     * Gets the number of points in this subset.
     */
    index_t count() const {
      return count_;
    }

    index_t level() const {
      return level_;
    }

    /**
     * Gets the number of children.
     */
    index_t num_children() const {
      return children_->size();
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
  GenHypercubeTree *MakeGenHypercubeTree
  (Matrix& matrix, index_t leaf_size,
   ArrayList< ArrayList <GenHypercubeTree *> > *nodes_in_each_level,
   ArrayList<index_t> *old_from_new = NULL,
   ArrayList<index_t> *new_from_old = NULL) {
    
    GenHypercubeTree *node = new GenHypercubeTree();
    index_t *old_from_new_ptr;
    
    if (old_from_new) {
      old_from_new->Init(matrix.n_cols());
      
      for (index_t i = 0; i < matrix.n_cols(); i++) {
	(*old_from_new)[i] = i;
      }
      
      old_from_new_ptr = old_from_new->begin();
    } else {
      old_from_new_ptr = NULL;
    }
    
    // Initialize the global list of nodes.
    nodes_in_each_level->Init(1);    

    // Initialize the root node.
    node->Init(0, matrix.n_cols(), 0);
    
    // Make the tightest cube bounding box you can fit around the
    // current set of points.
    tree_gen_hypercube_tree_private::ComputeBoundingHypercube(matrix, node);

    // Put the root node into the initial list of level 0.
    ((*nodes_in_each_level)[0]).Init();
    ((*nodes_in_each_level)[0]).PushBackCopy(node);

    tree_gen_hypercube_tree_private::SplitGenHypercubeTree
      (matrix, node, leaf_size, nodes_in_each_level, old_from_new_ptr, 0);

    // Index shuffling business...
    if (new_from_old) {
      new_from_old->Init(matrix.n_cols());
      for (index_t i = 0; i < matrix.n_cols(); i++) {
	(*new_from_old)[(*old_from_new)[i]] = i;
      }
    }
    
    return node;
  }
};

#endif
