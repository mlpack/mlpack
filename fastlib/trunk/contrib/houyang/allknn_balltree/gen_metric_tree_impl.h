/** @file gen_metric_tree_impl.h
 *
 *  Implementation for the learning of ball-trees
 *
 */

#include "fastlib/fastlib_int.h"

namespace learntrees_private {

  /**
   * Make a leaf node.
   * This function assumes that we have points embedded in Euclidean space.
   */
  template<typename TBound>
  void MakeLeafMetricTreeNode(const Matrix& matrix,
			      index_t begin, index_t count, TBound *bounds) {
    if (count==1) {
      Vector tmp;
      matrix.MakeColumnVector(0, &tmp);
      bounds->center().CopyValues(tmp);
      bounds->set_radius(0.0);
    }
    else {
      bounds->center().SetZero();
      
      index_t end = begin + count;
      for (index_t i = begin; i < end; i++) {
	Vector col;      
	matrix.MakeColumnVector(i, &col);
	la::AddTo(col, &(bounds->center()));
      }
      la::Scale(1.0 / ((double) count), &(bounds->center()));
      
      double furthest_distance;
      FurthestColumnIndex(bounds->center(), matrix, begin, count, &furthest_distance);
      bounds->set_radius(furthest_distance);
    }
  }

  template<typename TBound>
  void LearnMakeLeafMetricTreeNode(const Matrix& matrix,
			      index_t begin, index_t count, TBound *bounds, index_t *old_from_new) {

    bounds->center().SetZero();

    index_t end = begin + count;
    for (index_t i = begin; i < end; i++) {
      Vector col;      
      matrix.MakeColumnVector(old_from_new[i], &col);
      la::AddTo(col, &(bounds->center()));
    }
    la::Scale(1.0 / ((double) count), &(bounds->center()));

    double furthest_distance;
    LearnFurthestColumnIndex(bounds->center(), matrix, begin, count, &furthest_distance, old_from_new);
    bounds->set_radius(furthest_distance);
  }
  

  /**
   * Split the matrix into 2 using bounds
   */
  template<typename TBound>
  index_t MatrixPartition(Matrix& matrix, index_t first, index_t count,
			  TBound &left_bound, TBound &right_bound,
			  index_t *old_from_new, index_t *new_from_old) {
    
    index_t end = first + count;
    index_t left_count = 0;

    ArrayList<bool> left_membership;
    left_membership.Init(count);
    
    for (index_t left = first; left < end; left++) {

      // Make alias of the current point.
      Vector point;
      matrix.MakeColumnVector(left, &point);

      // Compute the distances from the two pivots.
      double distance_from_left_pivot =
	LMetric<2>::Distance(point, left_bound.center());
      double distance_from_right_pivot =
	LMetric<2>::Distance(point, right_bound.center());

      // We swap if the point is further away from the left pivot.
      if(distance_from_left_pivot > distance_from_right_pivot) {	
	left_membership[left - first] = false;
      }
      else {
	left_membership[left - first] = true;
	left_count++;
      }
    }

    index_t left = first;
    index_t right = first + count - 1;
    
    /* At any point:
     *
     *   everything < left is correct
     *   everything > right is correct
     */
    for (;;) {
      while (left_membership[left - first] && likely(left <= right)) {
        left++;
      }

      while (!left_membership[right - first] && likely(left <= right)) {
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

      // Swap the left vector with the right vector.
      left_vector.SwapValues(&right_vector);
      bool tmp = left_membership[left - first];
      left_membership[left - first] = left_membership[right - first];
      left_membership[right - first] = tmp;
      
      // Rearrange new_from_old
      if (new_from_old && old_from_new) {
	index_t t = new_from_old[old_from_new[left]];
        new_from_old[old_from_new[left]] = new_from_old[old_from_new[right]];
        new_from_old[old_from_new[right]] = t;
      }
      // Rearrange old_from_new
      if (old_from_new) {
        index_t t = old_from_new[left];
        old_from_new[left] = old_from_new[right];
        old_from_new[right] = t;
      }
      
      DEBUG_ASSERT(left <= right);
      right--;
    }
    
    DEBUG_ASSERT(left == right + 1);
    //printf("Left:%d, Right:%d\n", left_count, count-left_count);

    return left_count;
  }


  /**
   * Split the matrix into 2 using seperating hyperplane P; For Learning ball trees
   */
  template<typename TBound>
  void LearnMatrixPartition(Matrix& matrix, index_t first, index_t count, Vector& p, 
	  index_t *old_from_new, index_t *new_from_old, index_t *counts, const Matrix& Adj, const Matrix& Aff) {
    index_t d = matrix.n_rows();
    index_t N = matrix.n_cols();
    index_t d_plus_one = d+1;

    index_t end = first+ count- 1 ;
    
    ArrayList<bool> left_membership; // point belong to left or right side
    left_membership.Init(N);
    ArrayList<bool> cut_membership; // point on the cut or not
    cut_membership.Init(N);

    // Determine left/right membership
    double y; // predicted value y=P^Tx
    Vector x;
    x.Init(d_plus_one);
    for (index_t left_idx = first; left_idx <= end; left_idx++) {
      for (index_t i=0; i<d; i++)
	x[i] = matrix.get(i, left_idx);
      x[d] = 1.0;
      y = la::Dot(p, x);
      if (y >= 0) // y=P^Tx>=0 => left side
	left_membership[left_idx] = true;
      else // y=P^Tx<0 => right side
	left_membership[left_idx] = false;
    }
    /*
    // Determine on-cut/not-on-cut membership
    index_t cut_idx;
    // Points that are already on-cut in their ancestor nodes, or points that are not in [first end] region remain on-cut
    //for (cut_idx = first+count_noncut; cut_idx <= end; cut_idx++)
    for (cut_idx = 0; cut_idx < N; cut_idx++)
      cut_membership[cut_idx]= true;
    // Points that are in [first end] region and  not-on-cuts in their ancestor nodes remain not-on-cut
    for (cut_idx = first; cut_idx < first+count_noncut; cut_idx++)
      cut_membership[cut_idx]= false;
    index_t idx_data, opt_pos;
    for (index_t left_idx = first; left_idx <= end; left_idx++) {
      idx_data = old_from_new[left_idx];
      for (index_t k=0; k<(index_t)Adj.get(0, idx_data); k++) {
	opt_pos = (index_t)Adj.get(1, idx_data)+ k;
	if ( !cut_membership[ new_from_old[(index_t)Aff.get(1,opt_pos)] ] ) {
	  if (left_membership[ new_from_old[(index_t)Aff.get(0,opt_pos)] ] != 
	      left_membership[ new_from_old[(index_t)Aff.get(1,opt_pos)] ])
	    // two neighboring points are on different sides, i.e. they're on-cut
	    cut_membership[left_idx] = true;
	  else
	    // if a point is on-cut in any of its ancestor nodes, it remains on-cut
	    cut_membership[left_idx] = (false || cut_membership[left_idx]);
	}
	else{
	  cut_membership[left_idx] = true;
	}
      }
    }
    */
    // Fill-in the output ''counts'' info:
    // counts[0]==left_count; counts[1]==right_count
    for (index_t i=0; i<2; i++)
      counts[i]=0;
    for (index_t left_idx = first; left_idx <= end; left_idx++) {
      if (left_membership[left_idx]) // left side
	  counts[0] = counts[0] + 1;
      else // right side
	  counts[1] = counts[1] + 1;
    }

    // Rearrange old_from_new, new_from_old index vectors according to left/right membership
    index_t l= first, r=end;
    for (;;) {
      while (left_membership[l] && likely(l <= r) && likely(l<end)) {
        l++;
      }
      while (!left_membership[r] && likely(l <= r) && likely(r>first)) {
        r--;
      }
      if (l > r || r==first || l==end) {
        break;
      }
      Vector left_vector;
      Vector right_vector;
      matrix.MakeColumnVector(l, &left_vector);
      matrix.MakeColumnVector(r, &right_vector);
      left_vector.SwapValues(&right_vector);

      bool tmp = left_membership[l];
      left_membership[l] = left_membership[r];
      left_membership[r] = tmp;
      index_t t;
      if (new_from_old && old_from_new) {
	t = new_from_old[old_from_new[l]];
        new_from_old[old_from_new[l]] = new_from_old[old_from_new[r]];
        new_from_old[old_from_new[r]] = t;
      }
      if (old_from_new) {
        t = old_from_new[l];
        old_from_new[l] = old_from_new[r];
        old_from_new[r] = t;
      }
      DEBUG_ASSERT(l <= r);
      r--; l++;
    }
    //DEBUG_ASSERT(l == r + 1);
    /*
    // Rearrange old_from_new, new_from_old index vectors according to on-cut/not-on-cut membership,
    // for both left and right sides.
    // Left side
    l= first; r=first+ counts[0]+ counts[1]- 1;
    for (;;) {
      while (!cut_membership[l] && likely(l <= r)) {
        l++;
      }
      DEBUG_ASSERT(r >= first);
      while (cut_membership[r] && likely(l <= r)) {
        r--;
      }
      if (l > r) {
        break;
      }

      Vector left_vector;
      Vector right_vector;
      matrix.MakeColumnVector(l, &left_vector);
      matrix.MakeColumnVector(r, &right_vector);
      left_vector.SwapValues(&right_vector);

      bool tmp = cut_membership[l];
      cut_membership[l] = cut_membership[r];
      cut_membership[r] = tmp;
      index_t t;
      if (new_from_old && old_from_new) {
	t = new_from_old[old_from_new[l]];
        new_from_old[old_from_new[l]] = new_from_old[old_from_new[r]];
        new_from_old[old_from_new[r]] = t;
      }
      if (old_from_new) {
        t = old_from_new[l];
        old_from_new[l] = old_from_new[r];
        old_from_new[r] = t;
      }
      DEBUG_ASSERT(l <= r);
      r--;
    }
    //DEBUG_ASSERT(l == r+1);
    // Right side
    index_t start= first+ counts[0]+ counts[1];
    l= start; r=end;
    for (;;) {
      while (!cut_membership[l] && likely(l <= r)) {
        l++;
      }
      while (cut_membership[r] && likely(l <= r)) {
        r--;
      }
      if (l > r) {
        break;
      }

      Vector left_vector;
      Vector right_vector;
      matrix.MakeColumnVector(l, &left_vector);
      matrix.MakeColumnVector(r, &right_vector);
      left_vector.SwapValues(&right_vector);
      
      bool tmp = cut_membership[l];
      cut_membership[l] = cut_membership[r];
      cut_membership[r] = tmp;
      index_t t;
      if (new_from_old && old_from_new) {
	t = new_from_old[old_from_new[l]];
        new_from_old[old_from_new[l]] = new_from_old[old_from_new[r]];
        new_from_old[old_from_new[r]] = t;
      }
      if (old_from_new) {
        t = old_from_new[l];
        old_from_new[l] = old_from_new[r];
        old_from_new[r] = t;
      }
      DEBUG_ASSERT(l <= r);
      r--;
    }
    //DEBUG_ASSERT(l == r+1);
    */
  }



	
  index_t FurthestColumnIndex(const Vector &pivot, const Matrix &matrix, 
			      index_t begin, index_t count,
			      double *furthest_distance) {
    
    index_t furthest_index = -1;
    index_t end = begin + count;
    *furthest_distance = -1.0;

    for(index_t i = begin; i < end; i++) {
      Vector point;
      matrix.MakeColumnVector(i, &point);
      double distance_between_center_and_point = 
	LMetric<2>::Distance(pivot, point);
      
      if((*furthest_distance) < distance_between_center_and_point) {
	*furthest_distance = distance_between_center_and_point;
	furthest_index = i;
      }
    }

    return furthest_index;
  }

  
  index_t LearnFurthestColumnIndex(const Vector &pivot, const Matrix &matrix, 
			      index_t begin, index_t count,
			      double *furthest_distance, index_t *old_from_new) {
    
    index_t furthest_index = -1;
    index_t end = begin + count;
    *furthest_distance = -1.0;

    for(index_t i = begin; i < end; i++) {
      Vector point;
      matrix.MakeColumnVector(old_from_new[i], &point);
      double distance_between_center_and_point = LMetric<2>::Distance(pivot, point);
      
      if((*furthest_distance) < distance_between_center_and_point) {
	*furthest_distance = distance_between_center_and_point;
	furthest_index = i;
      }
    }

    return furthest_index;
  }


  template<typename TMetricTree>
  void CombineBounds(Matrix &matrix, TMetricTree *node, TMetricTree *left,
		     TMetricTree *right) {
    node->bound().center().SetZero();

    // Weighted center finding ball tree
    /*
    // Compute the weighted sum of left-most and right most points, set as center of this node
    la::AddExpert(left->count(), left->bound().center(), &(node->bound().center()));
    la::AddExpert(right->count(), right->bound().center(), &(node->bound().center()));
    la::Scale(1.0 / ((double) node->count()), &(node->bound().center()));
    
    double left_max_dist, right_max_dist;
    FurthestColumnIndex(node->bound().center(), matrix, left->begin(), left->count(), &left_max_dist);
    FurthestColumnIndex(node->bound().center(), matrix, right->begin(), right->count(), &right_max_dist);    
    node->bound().set_radius(std::max(left_max_dist, right_max_dist));
    //printf("Left_max_dist:%f, Right_max_dist:%f\n", left_max_dist, right_max_dist);
    */

    // Classical center finding ball tree
    la::AddOverwrite(left->bound().center(), right->bound().center(), &(node->bound().center()));
    la::Scale(0.5, &(node->bound().center()));
    node->bound().set_radius( LMetric<2>::Distance(left->bound().center(), right->bound().center()) );
  }


  /**
   * Generate a metric tree
   */
  template<typename TMetricTree>
  void SplitGenMetricTree(Matrix& matrix, TMetricTree *node,
			  index_t leaf_size, index_t *old_from_new, index_t *new_from_old) {
    TMetricTree *left = NULL;
    TMetricTree *right = NULL;

    // If the node is just too small, then do not split.
    if(node->count() < leaf_size) {
      MakeLeafMetricTreeNode(matrix, node->begin(), node->count(), &(node->bound()));
    }
    // Otherwise, attempt to split.
    else {
      bool can_cut = AttemptSplitting(matrix, node, &left, &right, leaf_size,
				      old_from_new, new_from_old);
      if(can_cut) {
	//printf("%f\n",(double(left->count())-double(right->count()))/(double(left->count())+double(right->count())) );

	// recursively generate metric tree for the left child
	SplitGenMetricTree(matrix, left, leaf_size, old_from_new, new_from_old);
	// recursively generate metric tree for the right child
	SplitGenMetricTree(matrix, right, leaf_size, old_from_new, new_from_old);
	// handle bounds, node.center and node.radius are updated
	CombineBounds(matrix, node, left, right);
      }
      else {
	MakeLeafMetricTreeNode(matrix, node->begin(), node->count(), &(node->bound()));
      }
    }
    // Set children information appropriately.
    node->set_children(matrix, left, right);
  }

  
  /**
   * Attemp to split a node into left and right children
   */
  template<typename TMetricTree>
  bool AttemptSplitting(Matrix& matrix, TMetricTree *node, TMetricTree **left, 
			TMetricTree **right, index_t leaf_size,
			index_t *old_from_new, index_t *new_from_old) {
    // DEBUG: check the correctness of old_from_new and new_from_old
    if (old_from_new && new_from_old)
      for (index_t i=0; i<matrix.n_cols(); i++)
	DEBUG_ASSERT(old_from_new[new_from_old[i]] == i);

    // Pick a random row.
    index_t random_row = math::RandInt(node->begin(), node->begin() +
				       node->count());
    random_row = node->begin();
    Vector random_row_vec;
    matrix.MakeColumnVector(random_row, &random_row_vec);

    // Now figure out the furthest point from the random row picked above.
    double furthest_distance;
    index_t furthest_from_random_row =
      FurthestColumnIndex(random_row_vec, matrix, node->begin(), node->count(),
			  &furthest_distance);
    Vector furthest_from_random_row_vec;
    matrix.MakeColumnVector(furthest_from_random_row,
			    &furthest_from_random_row_vec);
    // Then figure out the furthest point from the furthest point.
    double furthest_from_furthest_distance;
    index_t furthest_from_furthest_random_row =
      FurthestColumnIndex(furthest_from_random_row_vec, matrix, node->begin(),
			  node->count(), &furthest_from_furthest_distance);
    Vector furthest_from_furthest_random_row_vec;
    matrix.MakeColumnVector(furthest_from_furthest_random_row,
			    &furthest_from_furthest_random_row_vec);

    if(furthest_from_furthest_distance < DBL_EPSILON) {
      return false;
    }
    else {
      *left = new TMetricTree();
      *right = new TMetricTree();
      // Set left-most and right-most points for left and right child
      ((*left)->bound().center()).Init(matrix.n_rows());
      ((*right)->bound().center()).Init(matrix.n_rows());
      ((*left)->bound().center()).CopyValues(furthest_from_random_row_vec);
      ((*right)->bound().center()).CopyValues(furthest_from_furthest_random_row_vec);
      // Split node into left and right children
      index_t left_count = MatrixPartition(matrix, node->begin(), node->count(),
	 (*left)->bound(), (*right)->bound(), old_from_new, new_from_old);

      (*left)->Init(node->begin(), left_count, matrix.n_rows());
      (*right)->Init(node->begin() + left_count, node->count() - left_count, matrix.n_rows());

    }
    return true;
  }


  /**
   * Learn to recursively generate a ball tree
   */
  template<typename TMetricTree>
  void LearnSplitGenMetricTree(Matrix& matrix, TMetricTree *node,
       index_t leaf_size, index_t *old_from_new, index_t *new_from_old, 
       index_t knns, const Vector& D, const Matrix& Adj, const Matrix& Aff) {
    TMetricTree *left = NULL;
    TMetricTree *right = NULL;

    if(node->count()< leaf_size) {
      //printf("here1, %d\n", node->count());
      // If the node is just too small, then do not split
      MakeLeafMetricTreeNode(matrix, node->begin(), node->count(), &(node->bound()));
    }
    //else if(node->count()>= leaf_size && node->count_noncut()< leaf_size) {
    //  // If the number of not-on-cut points is too small, then do normal ball tree split
    //  SplitGenMetricTree(matrix, node, leaf_size, old_from_new, new_from_old);
    //}
    // Otherwise, attempt to split.
    else {
      int can_cut_ind = LearnAttemptSplitting(matrix, node, &left, &right, 
		          leaf_size, old_from_new, new_from_old, knns, D, Adj, Aff);

      if(can_cut_ind==0) { // left and right children generated

	//printf("%f\n",(double(left->count())-double(right->count()))/(double(left->count())+double(right->count())) );

	// recursively generate metric tree for the left child
	LearnSplitGenMetricTree(matrix, left, leaf_size, old_from_new, new_from_old, knns, D, Adj, Aff);
	// recursively generate metric tree for the right child
	LearnSplitGenMetricTree(matrix, right, leaf_size, old_from_new, new_from_old, knns, D, Adj, Aff);
	// Combine bounds, i.e. calculate radius of the current node
	CombineBounds(matrix, node, left, right);
	node->set_learn_flag(true);
      }
      else  if (can_cut_ind==1){ // can_cut_ind==1
	// Too many samples have no in-node affinities (can_cut_ind==1), or
	// the (attempt) splitted left/right children has/have too small # of not-on-cut data.
	// Just do normal ball tree split (can_cut_ind==2)
	
	SplitGenMetricTree(matrix, node, leaf_size, old_from_new, new_from_old);
	node->set_learn_flag(false);
	//MakeLeafMetricTreeNode(matrix, node->begin(), node->count(), &(node->bound()));
      }
      else { // can_cut_ind==2
	// Too many samples have no in-node affinities (can_cut_ind==1), or
	// the (attempt) splitted left/right children has/have too small # of not-on-cut data.
	// Just do normal ball tree split (can_cut_ind==2)
	
	SplitGenMetricTree(matrix, node, leaf_size, old_from_new, new_from_old);
	node->set_learn_flag(false);
	//MakeLeafMetricTreeNode(matrix, node->begin(), node->count(), &(node->bound()));
      }
    }
    // Set children information appropriately.
    node->set_children(matrix, left, right);
  }
  

  /**
   * Helper Function to Compare EigenValues
   */
  int EigenCompare(const void *col_a, const void *col_b) {
    const double *col_a_ptr = (double*)col_a;
    const double *col_b_ptr = (double*)col_b;
    if (*col_a_ptr> *col_b_ptr)
      return 1;
    else if (*col_a_ptr< *col_b_ptr)
      return -1;
    else
      return 0;
  }
  
  /**
   * Attemp to split a node by learning a normalized min-cut hyperplane
   */
  template<typename TMetricTree>
  int LearnAttemptSplitting(Matrix& matrix, TMetricTree *node, TMetricTree **left, TMetricTree **right, 
			index_t leaf_size, index_t *old_from_new, index_t *new_from_old, 
			index_t knns, const Vector& D, const Matrix& Adj, const Matrix& Aff) {
    index_t i, j, k, n;
    index_t d = matrix.n_rows();
    index_t N = matrix.n_cols();
    index_t d_plus_one = d+1;
    double dbl_tmp = 0.0;
    // Number of data that will be used to do matrix multiplication: node->count_noncut_
    // The rest: node->count_cut_ will NOT be used for matrix multiplication
    //index_t num_data_noncut = node->count_noncut(); 
    index_t num_data_node = node->count(); // number of all data in current node
    index_t first = node->begin();
    index_t end = first + node->count() - 1;

    index_t idx_data;
    index_t opt_pos; // operating position(col) in Aff
    index_t aff_pos; // the position of the affinite sample to opt_pos
    
    /*for(i=0;i<node->count();i++)
      //printf("first=%d, num_data_noncut=%d, num_data_cut=%d\n", first,num_data_noncut, node->count_cut());
      printf("%d_", old_from_new[i]);
    printf("\n");
    */

    // DEBUG: check the correctness of old_from_new and new_from_old
    for (i=0; i<N; i++)
      DEBUG_ASSERT(old_from_new[new_from_old[i]] == i);

    // 0.Check how many samples in the node do not have in-node affinities.
    ArrayList<bool> non_in_node_aff_membership;
    non_in_node_aff_membership.Init(N);
    for (n=0; n<N; n++)
      non_in_node_aff_membership[n] = false;
    index_t num_non_in_node_aff = 0; // number of samples in the current node that do not have in-node affinities
    for (n=0; n<num_data_node; n++) { // N'
      idx_data = old_from_new[first+n];
      index_t non_in_node_aff_ct= 0;
      index_t num_aff= (index_t)Adj.get(0, idx_data);
      for (k=0; k<num_aff; k++) { //k
	opt_pos = (index_t)Adj.get(1, idx_data) + k;
	aff_pos =(index_t)Aff.get(1, opt_pos);
	// affinite sample to the current sample in a node is NOT within this node
	if (new_from_old[aff_pos]<first || new_from_old[aff_pos]>end)
	  non_in_node_aff_ct++;
      }
      if (non_in_node_aff_ct== num_aff){
	non_in_node_aff_membership[first+n] = true;
	num_non_in_node_aff++;
      }
    }
    // If too many (>=40%) samples have no in-node affinities, just use normal ball tree split
    //printf("Portion:%f\n",(double)num_non_in_node_aff/ (double)num_data_node);
    if ((double)num_non_in_node_aff/ (double)num_data_node>= 0.4) {
      //printf("here3_1, %d: Portion large\n\n", node->count());
      return 1;
    }
    // Rearrange old_from_new, new_from_old index vectors according to non_in_node_aff_membership
    index_t l= first, r=end;
    for (;;) {
      while (!non_in_node_aff_membership[l] && likely(l <= r) && likely(l<end)) {
        l++;
      }
      while (non_in_node_aff_membership[r] && likely(l <= r) && likely(r>first)) {
        r--;
      }
      if (l > r || r==first || l==end) {
        break;
      }
      Vector left_vector;
      Vector right_vector;
      matrix.MakeColumnVector(l, &left_vector);
      matrix.MakeColumnVector(r, &right_vector);
      left_vector.SwapValues(&right_vector);

      bool tmp = non_in_node_aff_membership[l];
      non_in_node_aff_membership[l] = non_in_node_aff_membership[r];
      non_in_node_aff_membership[r] = tmp;
      index_t t;
      if (new_from_old && old_from_new) {
	t = new_from_old[old_from_new[l]];
        new_from_old[old_from_new[l]] = new_from_old[old_from_new[r]];
        new_from_old[old_from_new[r]] = t;
      }
      if (old_from_new) {
        t = old_from_new[l];
        old_from_new[l] = old_from_new[r];
        old_from_new[r] = t;
      }
      DEBUG_ASSERT(l <= r);
      r--; l++;
    }
    // number of samples in the current node that have in-node affinities
    // these number of samples will be used for the following normalized min-cut
    index_t num_in_node_aff= num_data_node- num_non_in_node_aff;
    
    // 1.Calculate d+1-by-d+1 matrices XDX^T and XLX^T, where L==D-A, d==matrix.n_rows().
    //   Do the multiplications by hand, since D and A are both very sparse and highly structured.
    Vector XD_XA_row; // temp vector to store a row (1~d^th) of XD or XA
    XD_XA_row.Init(num_in_node_aff);
    XD_XA_row.SetZero();
    Vector X_T_col;
    X_T_col.Init(num_in_node_aff);
    Vector XD_XA_last_row; // temp vector to store the d+1_th row of XD or XA
    XD_XA_last_row.Init(num_in_node_aff);
    XD_XA_last_row.SetZero();
    
    // Construct XDX^T(should be symmetric), time complexity: O(kdN)+O(ddN)
    Matrix XDXT; // d+1-by-d+1
    XDXT.Init(d_plus_one, d_plus_one);
    for (i=0; i<d; i++) { // d
      for (n=0; n<num_in_node_aff; n++) { // N'
	idx_data = old_from_new[first+n];
	double D_diag_tmp= D[idx_data];
	for (k=0; k<(index_t)Adj.get(0, idx_data); k++) { //k
	  opt_pos = (index_t)Adj.get(1, idx_data) + k;
	  aff_pos =(index_t)Aff.get(1, opt_pos);
	  // if affinite sample to the current sample in a node is NOT within this node, decrease the degree of this current sample
	  if (new_from_old[aff_pos]<first || new_from_old[aff_pos]>end)
	    D_diag_tmp = D_diag_tmp- Aff.get(2,opt_pos);
	}
	DEBUG_ASSERT(D_diag_tmp>=0);
	XD_XA_last_row[n] = D_diag_tmp;
	XD_XA_row[n] = matrix.get(i, first+n) * D_diag_tmp;
      }
      for (j=0; j<d; j++) { // d
	for (n=0; n<num_in_node_aff; n++) { // N'
	  X_T_col[n] = matrix.get( j, first+n ); // jth row, nth col of submatrix
	}	
	XDXT.set(i, j, la::Dot(XD_XA_row, X_T_col)); // Set the [i j]th item of XDXT
      }
      dbl_tmp=0.0;
      for (n=0; n<num_in_node_aff; n++)
	dbl_tmp += XD_XA_row[n];
      XDXT.set(i, d, dbl_tmp); // Set the d+1-th column(except the last item) of XDXT
    }
    for (i=0; i<d; i++) {
      for (n=0; n<num_in_node_aff; n++)
	X_T_col[n] = matrix.get( i, first+n ); // ith row, nth col of submatrix
      XDXT.set(d, i, la::Dot(XD_XA_last_row, X_T_col)); // Set the d+1-th row(except the last item) of XDXT
    }
    dbl_tmp= 0.0;
    for (n=0; n<num_in_node_aff; n++)
      dbl_tmp += XD_XA_last_row[n];
    XDXT.set(d, d, dbl_tmp); // Set the [d+1 d+1]th item
    
    // Construct XAX^T(should be symmetric), time complexity: O(kdN)+O(ddN)
    Matrix XAXT; // d+1-by-d+1
    XAXT.Init(d_plus_one, d_plus_one);
    for (i=0; i<d; i++) { // d
      XD_XA_row.SetZero();
      for (n=0; n<num_in_node_aff; n++) { // N'
	idx_data = old_from_new[first+n];
	for (k=0; k<(index_t)Adj.get(0, idx_data); k++) { // k
	  opt_pos = (index_t)Adj.get(1, idx_data)+ k;
	  aff_pos =(index_t)Aff.get(1, opt_pos);
	  // if affinite sample to the current sample in a node is within this node, count it, otherwise dicard it
	  if (new_from_old[aff_pos]>=first && new_from_old[aff_pos]<=end)
	    XD_XA_row[n] = XD_XA_row[n] + matrix.get(i, new_from_old[aff_pos]) * Aff.get(2, opt_pos);
	}
      }
      for (j=0; j<d; j++) { // d
	for (n=0; n<num_in_node_aff; n++) { // N'
	  X_T_col[n] = matrix.get( j, first+n ); // jth row, nth col of submatrix
	}	
	XAXT.set(i, j, la::Dot(XD_XA_row, X_T_col)); // Set the [i j]th item of XAXT
      }
      dbl_tmp=0.0;
      for (n=0; n<num_in_node_aff; n++)
	dbl_tmp += XD_XA_row[n];
      XAXT.set(i, d, dbl_tmp); // Set the d+1-th column(except the last item) of XAXT
    }
    XD_XA_last_row.SetZero();
    for (n=0; n<num_in_node_aff; n++){
      idx_data = old_from_new[first+n];
      for (k=0; k<(index_t)Adj.get(0, idx_data); k++) {
	opt_pos = (index_t)Adj.get(1, idx_data) + k;
	aff_pos =(index_t)Aff.get(1, opt_pos);
	  if (new_from_old[aff_pos]>=first && new_from_old[aff_pos]<=end)
	    XD_XA_last_row[n] = XD_XA_last_row[n] + Aff.get(2, opt_pos);
      }
    }
    for (i=0; i<d; i++) {
      for (n=0; n<num_in_node_aff; n++)
	X_T_col[n] = matrix.get( i, first+n ); // ith row, nth col of submatrix
      XAXT.set(d, i, la::Dot(XD_XA_last_row, X_T_col)); // Set the d+1-th row(except the last item) of XATX
    }
    dbl_tmp= 0.0;
    for (n=0; n<num_in_node_aff; n++)
      dbl_tmp += XD_XA_last_row[n];
    XAXT.set(d, d, dbl_tmp); // Set the [d+1 d+1]th item

    // Construct XLX^T==XDX^T-XAX^T, it should be symmetric.
    Matrix XLXT;
    la::SubInit(XAXT, XDXT, &XLXT);
    /*
    printf("XAX0_0:%f,XAX0_1:%f,XAX0_2:%f,XAX0_3:%f\n",XAXT.get(0,0),XAXT.get(0,1),XAXT.get(0,2),XAXT.get(0,3));
    printf("XAX1_0:%f,XAX1_1:%f,XAX1_2:%f,XAX1_3:%f\n",XAXT.get(1,0),XAXT.get(1,1),XAXT.get(1,2),XAXT.get(1,3));
    printf("XAX2_0:%f,XAX2_1:%f,XAX2_2:%f,XAX2_3:%f\n",XAXT.get(2,0),XAXT.get(2,1),XAXT.get(2,2),XAXT.get(2,3));
    printf("XAX3_0:%f,XAX3_1:%f,XAX3_2:%f,XAX3_3:%f\n",XAXT.get(3,0),XAXT.get(3,1),XAXT.get(3,2),XAXT.get(3,3));
    
    printf("XLX0_0:%f,XLX0_1:%f,XLX0_2:%f,XLX0_3:%f\n",XLXT.get(0,0),XLXT.get(0,1),XLXT.get(0,2),XLXT.get(0,3));
    printf("XLX1_0:%f,XLX1_1:%f,XLX1_2:%f,XLX1_3:%f\n",XLXT.get(1,0),XLXT.get(1,1),XLXT.get(1,2),XLXT.get(1,3));
    printf("XLX2_0:%f,XLX2_1:%f,XLX2_2:%f,XLX2_3:%f\n",XLXT.get(2,0),XLXT.get(2,1),XLXT.get(2,2),XLXT.get(2,3));
    printf("XLX3_0:%f,XLX3_1:%f,XLX3_2:%f,XLX3_3:%f\n",XLXT.get(3,0),XLXT.get(3,1),XLXT.get(3,2),XLXT.get(3,3));
    */

    data::Save("data_3_1000_graph_XAXT.csv", XAXT);
    data::Save("data_3_1000_graph_XDXT.csv", XDXT);
    data::Save("data_3_1000_graph_XLXT.csv", XLXT);
    
    // 2.Linear Normalized Cut: solving generalized eigenvalue problem: (XLX^T)P=lambda(XDX^T)P
    // Take the gen-eigenvector corresponding to the second smallest gen-eigenvalue
    //double alpha_real[d_plus_one];
    //double alpha_imag[d_plus_one];
    //double beta[d_plus_one];
    double gen_eigenvalues[d_plus_one];
    //double gen_eigenvalues_img[d_plus_one];
    
    Matrix V_raw; // generalized eigenvectors
    //V_raw.Init(d_plus_one, d_plus_one);
    //la::GenEigenNonSymmetric(&XLXT, &XDXT, alpha_real, alpha_imag, beta, V_raw.ptr());
    V_raw.Copy(XLXT);
    Matrix XDXT_cp;
    XDXT_cp.Copy(XDXT);
    
    // Linear Normalized Cut
    la::GenEigenSymmetric(1, &V_raw, &XDXT_cp, gen_eigenvalues);

    // Linear Ratio Cut
    //la::EigenExpert(&XLXT, gen_eigenvalues, gen_eigenvalues_img, V_raw.ptr());

    // generalized eigenvalues: (alpha_real(j) + alpha_imag(j)*i)/beta(j), j=1,...,N
    //for (j=0; j<d_plus_one; j++){
      //DEBUG_ASSERT(beta[j] != 0);
      //gen_eigenvalues[j] = alpha_real[j]/beta[j];
    //}

    //printf("r:%f_r:%f_r:%f_r:%f\n", alpha_real[0], alpha_real[1], alpha_real[2], alpha_real[3]);
    //printf("i:%f_i:%f_i:%f_i:%f\n", alpha_imag[0], alpha_imag[1], alpha_imag[2], alpha_imag[3]);
    //printf("%f_%f_%f_%f\n", beta[0], beta[1], beta[2], beta[3]);
    //for (i=0; i<d_plus_one; i++)
    //  printf("%f\n", gen_eigenvalues[i]);   

    if (gen_eigenvalues[1]<1e-4&&gen_eigenvalues[2]<1e-4&&gen_eigenvalues[3]<1e-4&&gen_eigenvalues[4]<1e-4){
      //printf("here 3_3: %d. Eigenvalues_Zero:\n\n", node->count());
      data::Save("data_3_1000_graph_XAXT_zero.csv", XAXT);
      data::Save("data_3_1000_graph_XDXT_zero.csv", XDXT);
      data::Save("data_3_1000_graph_XLXT_zero.csv", XLXT);
      return 2;
    }


    if (fabs(gen_eigenvalues[0])>1e-4){
      //printf("here 3_4: %d. Eigenvalues_NonZero:\n\n", node->count());
      data::Save("data_3_1000_graph_XAXT_nonzero.csv", XAXT);
      data::Save("data_3_1000_graph_XDXT_nonzero.csv", XDXT);
      data::Save("data_3_1000_graph_XLXT_nonzero.csv", XLXT);
    }

    //data::Save("data_3_1000_graph_eigen_vec.csv", V_raw); 
    
    // Sort eigenvalues
    Matrix eigens_for_sort;
    eigens_for_sort.Init(1+d_plus_one, d_plus_one); // stack eigenvalues and eigenvectors for sort
    for (i=0; i<d_plus_one; i++) {
      eigens_for_sort.set(0, i, gen_eigenvalues[i]);
      for (j=1; j<=d_plus_one; j++) {
	eigens_for_sort.set(j, i, V_raw.get(j-1, i));
      }
    }

    qsort(eigens_for_sort.ptr(), d_plus_one, (1+d_plus_one)*sizeof(double), EigenCompare);
    Vector p;
    p.Init(d_plus_one);
    // p is the 2nd smallest eigenvector of the sorted eigenvalues
    //for (k=0; k<d_plus_one; k++)
    //  if (eigens_for_sort.get(0,k)>1e-4 )
    //	break;
    //printf("k=%d\n",k);
    //if (k>=(index_t)d_plus_one/2)
    //  return false;
    for (i=0; i<d_plus_one; i++){
      p[i]= eigens_for_sort.get(i+1, 1);
      //printf("%f_", p[i]);
    }
    //printf("\n");

    // 3.Do node partition.
    ArrayList<index_t> counts;
    counts.Init(2);
    // counts[0]==left_count; counts[1]==right_count;
    LearnMatrixPartition<TMetricTree>(matrix, first, node->count(), 
				      p, old_from_new, new_from_old, counts.begin(), Adj, Aff);
    //printf("c0=%d,c1=%d\n",counts[0], counts[1]);
    
    //if (counts[0]>=leaf_size && counts[1]>=leaf_size) {
    if (counts[0]>=1 && counts[1]>=1) {
      *left = new TMetricTree();
      *right = new TMetricTree();
      //(*left)->LearnInit(first, counts[0], counts[1]);
      (*left)->Init(first, counts[0], d);
      //(*right)->LearnInit(first+ counts[0]+ counts[1], counts[2], counts[3]);
      (*right)->Init(first+counts[0], counts[1], d);

      // 4.Calculate centroids of left and right children
      index_t num_data_left = counts[0];
      index_t num_data_right = counts[1];
      Vector left_centroid;
      left_centroid.Init(d);
      left_centroid.SetZero();
      for (n=first; n<first+num_data_left; n++) {
	Vector tmp;
	matrix.MakeColumnVector(n, &tmp);
	la::AddTo(tmp, &left_centroid);
      }
      la::Scale(1.0/((double) num_data_left), &left_centroid);
      ((*left)->bound().center()).Init(d);
      ((*left)->bound().center()).CopyValues(left_centroid);
      
      Vector right_centroid;
      right_centroid.Init(d);
      right_centroid.SetZero();
      for (n=first+num_data_left; n<first+num_data_left+num_data_right; n++) {
	Vector tmp;
	matrix.MakeColumnVector(n, &tmp);
	la::AddTo(tmp, &right_centroid);
      }
      la::Scale(1.0/((double) num_data_right), &right_centroid);
      ((*right)->bound().center()).Init(d);
      ((*right)->bound().center()).CopyValues(right_centroid);

      //      Vector p_cp;
      //      p_cp.Alias(node->p());
      //      p_cp.CopyValues(p);
      (node->p_).CopyValues(p);

      /*      Vector pp;
      pp.Alias(node->p());
      printf("node-p-0=%f\n", pp[0]);
      printf("node-p-0=%f\n", (node->p_)[0]);
      */

      //printf("here2, %d: GOOD\n\n", node->count());
      return 0;
    }
    else{
      //printf("here3_2, %d: Too small splits\n\n", node->count());
      // the (attempt) splitted left/right children has/have too small # of data
      // Instead of learning node split, just make it a leaf node
      return 2;
    }
  }
};




  /**
   * Generate a metric tree, using method in Ting Liu's PhD Thesis
   */
  /*
  template<typename TMetricTree>
  void NormalSplitGenMetricTree(Matrix& matrix, TMetricTree *node,
			  index_t leaf_size, index_t *old_from_new, index_t *new_from_old) {
    TMetricTree *left = NULL;
    TMetricTree *right = NULL;

    // If the node is just too small, then do not split.
    if(node->count() < leaf_size) {
      //printf("here1, %d\n", node->count());
      MakeLeafMetricTreeNode(matrix, node->begin(), node->count(), &(node->bound()));
    }
    // Otherwise, attempt to split.
    else {
      bool can_cut = AttemptSplitting(matrix, node, &left, &right, leaf_size,
				      old_from_new, new_from_old);
      if(can_cut) {
	//printf("here2, %d\n", node->count());
	// recursively generate metric tree for the left child
	NormalSplitGenMetricTree(matrix, left, leaf_size, old_from_new, new_from_old);
	// recursively generate metric tree for the right child
	NormalSplitGenMetricTree(matrix, right, leaf_size, old_from_new, new_from_old);
	// handle bounds
	CombineBounds(matrix, node, left, right);
      }
      else {
	//printf("here3, %d\n", node->count());
	MakeLeafMetricTreeNode(matrix, node->begin(), node->count(), &(node->bound()));
      }
    }
    // Set children information appropriately.
    node->set_children(matrix, left, right);
  }
  */
