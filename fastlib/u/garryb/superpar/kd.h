/**
 * @file kd.h
 *
 * Pointerless versions of everything needed to make different kinds of
 * trees.
 *
 * Eventually we'll have to figure out how to make these dynamic -- this
 * task ain't no trivial thing.
 */

#ifndef SUPERPAR_KD_H
#define SUPERPAR_KD_H

/**
 * A binary space partitioning tree, such as KD or ball tree, for use
 * with super-par.
 *
 * This particular tree forbids you from having more children.
 *
 * @param TBound the bounding type of each child (TODO explain interface)
 * @param TDataset the data set type
 * @param TStatistic extra data in the node
 *
 * @experimental
 */
template<class TBound,
         int t_cardinality = 2>
class SpNode {
 public:
  typedef TBound Bound;
  typedef TDataset Dataset;
  typedef TStatistic Statistic;
  
  enum {
    /** The root node of a tree is always at index zero. */
    ROOT_INDEX = 0;
  };
  
 private:
  index_t begin_;
  index_t count_;
  
  Bound bound_;

  index_t children_[t_cardinality];
  
  OT_DEF(SpNode) {
    OT_MY_OBJECT(begin_);
    OT_MY_OBJECT(count_);
    OT_MY_OBJECT(bound_);
    OT_MY_ARRAY(children_, t_cardinality);
  }
  
 public:
  SpNode() {
    DEBUG_ONLY(begin_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(count_ = BIG_BAD_NUMBER);
    mem::DebugPoison(&children_, t_cardinality);
  }  

  ~SpNode() {
    DEBUG_ONLY(begin_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(count_ = BIG_BAD_NUMBER);
    mem::DebugPoison(&children_, t_cardinality);
  }
  
  void Init(index_t begin_in, index_t count_in) {
    DEBUG_ASSERT(begin_ == BIG_BAD_NUMBER);
    begin_ = begin_in;
    count_ = count_in;
  }

  const Bound& bound() const {
    return bound_;
  }

  Bound& bound() {
    return bound_;
  }

  index_t child(index_t child_number) const {
    return children_[child_number];
  }

  void set_child(index_t child_number, index_t child_index) {
    DEBUG_BOUNDS(child_number, t_cardinality);
    children_[child_number] = child_index;
  }

  void set_leaf() {
    children_[0] = -index_t(1);
  }

  bool is_leaf() const {
    return children_[0] == -index_t(1);
  }

  /**
   * Gets the index of the first point of this subset.
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
  
  /**
   * Gets the number of points in this subset.
   */
  index_t count() const {
    return count_;
  }
  
  /**
   * Returns the number of children of this node.
   */
  index_t cardinality() const {
    return t_cardinality;
  }
  
  void PrintSelf() const {
    printf("node: %d to %d: %d points total\n",
       begin_, begin_ + count_ - 1, count_);
  }
};

/* */


namespace tree_kdtree_private {

  template<typename TBound>
  void FindBoundFromMatrix(const Matrix& matrix,
      index_t first, index_t count, TBound *bounds) {
    index_t end = first + count;
    for (index_t i = first; i < end; i++) {
      Vector col;

      matrix.MakeColumnVector(i, &col);
      bounds->Update(col);
    }
  }

  template<typename TBound>
  index_t MatrixPartition(
      Matrix& matrix, index_t dim, double splitvalue,
      index_t first, index_t count,
      TBound* left_bound, TBound* right_bound,
      index_t *old_from_new) {
    index_t left = first;
    index_t right = first + count - 1;
    
    /* At any point:
     *
     *   everything < left is correct
     *   everything > right is correct
     */
    for (;;) {
      while (matrix.get(dim, left) < splitvalue && likely(left <= right)) {
        Vector left_vector;
        matrix.MakeColumnVector(left, &left_vector);
        left_bound->Update(left_vector);
        left++;
      }

      while (matrix.get(dim, right) >= splitvalue && likely(left <= right)) {
        Vector right_vector;
        matrix.MakeColumnVector(right, &right_vector);
        right_bound->Update(right_vector);
        right--;
      }

      if (unlikely(left > right)) {
        /* left == right + 1 */
        break;
      }

      Vector left_vector;
      Vector right_vector;

      matrix.MakeColumnVector(left, &left_vector);
      matrix.MakeColumnVector(right, &right_vector);

      left_vector.SwapValues(&right_vector);

      left_bound->Update(left_vector);
      right_bound->Update(right_vector);
      
      if (old_from_new) {
        index_t t = old_from_new[left];
        old_from_new[left] = old_from_new[right];
        old_from_new[right] = t;
      }

      DEBUG_ASSERT(left <= right);
      right--;
      
      // this conditional is always true, I belueve
      //if (likely(left <= right)) {
      //  right--;
      //}
    }

    DEBUG_ASSERT(left == right + 1);

    return left;
  }

  template<typename TKdTree>
  void SplitKdTreeMidpoint(Matrix& matrix,
      TKdTree *node, index_t leaf_size, index_t *old_from_new) {
    TKdTree *left = NULL;
    TKdTree *right = NULL;
    
    //FindBoundFromMatrix(matrix, node->begin(), node->count(),
    //    &node->bound());

    if (node->count() > leaf_size) {
      index_t split_dim = BIG_BAD_NUMBER;
      double max_width = -1;

      for (index_t d = 0; d < matrix.n_rows(); d++) {
        double w = node->bound().get(d).width();

        if (unlikely(w > max_width)) {
          max_width = w;
          split_dim = d;
        }
      }

      double split_val = node->bound().get(split_dim).mid();

      if (max_width == 0) {
        // Okay, we can't do any splitting, because all these points are the
        // same.  We have to give up.
      } else {
        left = new TKdTree();
        left->bound().Init(matrix.n_rows());

        right = new TKdTree();
        right->bound().Init(matrix.n_rows());

        index_t split_col = MatrixPartition(matrix, split_dim, split_val,
            node->begin(), node->count(),
            &left->bound(), &right->bound(),
            old_from_new);
        
        DEBUG_MSG(3.0,"split (%d,[%d],%d) dim %d on %f (between %f, %f)",
            node->begin(), split_col,
            node->begin() + node->count(), split_dim, split_val,
            node->bound().get(split_dim).lo,
            node->bound().get(split_dim).hi);

        left->Init(node->begin(), split_col - node->begin());
        right->Init(split_col, node->begin() + node->count() - split_col);

        // This should never happen if max_width > 0
        DEBUG_ASSERT(left->count() != 0 && right->count() != 0);

        SplitKdTreeMidpoint(matrix, left, leaf_size, old_from_new);
        SplitKdTreeMidpoint(matrix, right, leaf_size, old_from_new);
      }
    }

    node->set_children(matrix, left, right);
  }
};

#endif
