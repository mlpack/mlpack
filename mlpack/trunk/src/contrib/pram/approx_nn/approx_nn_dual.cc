#include <fastlib/fastlib.h>
#include <vector>
#include "approx_nn_dual.h"

/**
 * This function computes the probability of
 * a particular quantile given the set and sample sizes
 * Computes P(d_(1) <= d_(1+rank_approx))
 */
double ApproxNN::ComputeProbability_(size_t set_size,
				     size_t sample_size,
				     size_t rank_approx) {
  double sum;
  Vector temp_a, temp_b;

  temp_a.Init(rank_approx+1);
  temp_a.SetAll(1.0);
  temp_b.Init(rank_approx+1);

  // calculating the temp_b
  temp_b[rank_approx] = 1;
  size_t i, j = 1;
  for (i = rank_approx-1; i > -1; i--, j++) {
    double frac = (double)(set_size-(sample_size-1)-j)
      / (double)(set_size - j);
    temp_b[i] = temp_b[i+1]*frac;
  }
  DEBUG_ASSERT(j == rank_approx + 1);

  // computing the sum and the product with n/N
  sum = la::Dot(temp_a, temp_b);
  double prob = (double) sample_size / (double) set_size;
  prob *= sum;

  // asserting that the probability is < 1.0
  // This may not be the case when 'n' is close to 'N' and 
  // 'rank_approx' is large enough to make N-1-rank_approx+i < n-1
  return prob;
}

/**
 * This function computes the probability of
 * a particular quantile given the set and sample sizes
 * Computes P(d_(k) <= d_(rank_approx))
 */
double ApproxNN::ComputeProbability_(size_t set_size,
				     size_t sample_size,
				     size_t k,
				     size_t rank_approx) {
							
  double gamma, sum, prod1, prod2;
  gamma = (double) (sample_size - k + 1) 
    / (double) (set_size - k + 1);
  sum = 0.0;
		
  for (size_t i = 0; i < (rank_approx - k +1); i++) {
    prod1 = 1.0; prod2 = 1.0;
    for (size_t j = 0; j < (k - 1); j++)
      prod1 *= (double) ((rank_approx - i - 1 -j)*(sample_size -j))
	/ (double) ((k - 1 -j)*(set_size -j));
			
    for (size_t j = 0; j < (sample_size - k); j++)
      prod2 *= (double) (set_size - rank_approx + i -j)
	/ (double) (set_size - k -j);
			
    sum += prod1*prod2;
  }
		
  return (gamma * sum);
}

/**
 * This function computes the minimum sample sizes
 * required to obtain the approximate rank with
 * a given probability (alpha).
 * 
 * It assumes that the ArrayList<size_t> *samples
 * has been initialized to length N.
 */
void ApproxNN::ComputeSampleSizes_(size_t rank_approx,
				   double alpha,
				   ArrayList<size_t> *samples) {

  size_t set_size = samples->size(),
    n = samples->size();

  double prob;
  DEBUG_ASSERT(alpha <= 1.0);
  double beta = 0;

  // going through all values of sample sizes 
  // to find the minimum samples required to satisfy the 
  // desired bound
  do {
    n--;
    prob = ComputeProbability_(set_size,
			       n, rank_approx);
  } while (prob >= alpha);
  
  (*samples)[--set_size] = ++n;
  beta = (double) n / (double) (set_size+1);

  // set the sample sizes for every node size for fast-access
  // during the algorithm
  do {
    n = (size_t) (beta * (double)(set_size)) +1;
    (*samples)[--set_size] = n;
  } while (set_size > rank_approx);
  
  while (set_size > 0) {
    (*samples)[--set_size] = 1;
  }
}

/**
 * This function computes the minimum sample sizes
 * required to obtain the approximate rank with
 * a given probability (alpha) and for the kNN version
 * of the problem.
 * 
 * It assumes that the ArrayList<size_t> *samples
 * has been initialized to length N.
 */
void ApproxNN::ComputeSampleSizes_(size_t rank_approx,
				   double alpha,
				   size_t k,
				   ArrayList<size_t> *samples) {

  size_t set_size = samples->size();
  size_t ub = set_size, lb = k;
  size_t n = lb;

  if (rank_approx > k) {
    bool done = false;
    double prob;
    do {
      prob = ComputeProbability_(set_size, n, 
				 k, rank_approx);

      // doing a binary search on the sample size
      // for the desired accuracy bound.
      if (prob > alpha) {
	if (prob - alpha < 0.001 || ub == lb + 1) {
	  done = true;
	  break;
	}
	else 
	  ub = n;
      } else {
	if (prob < alpha) {
	  if (n == lb) {
	    n++;
	    continue;
	  } else {
	    lb = n;
	  }
	} else {
	  done = true;
	  break;
	}
      }
      n = (ub + lb) / 2;
    } while (!done);
    (*samples)[--set_size] = n + 1;
    
    double beta = (double) (n+1) / (double) (set_size +1);
			
    // set the sample sizes for every node size for fast-access
    // during the algorithm
    do {
      n = (size_t) (beta * (double)(set_size)) +1;
      (*samples)[--set_size] = n;
    } while (set_size > rank_approx);
			
    while (set_size > 0) {
      set_size--;
      (*samples)[set_size] = min(k, set_size + 1);
    }
  } else {
    while (set_size > 0) {
      set_size--;
      (*samples)[set_size] = set_size +1;
    }
  }
}


/**
 * Performs exhaustive computation between two leaves.  
 */
void ApproxNN::ComputeBaseCase_(TreeType* query_node,
				TreeType* reference_node) {
   
  DEBUG_ASSERT(query_node != NULL);
  DEBUG_ASSERT(reference_node != NULL);

  DEBUG_WARN_IF(!query_node->is_leaf());
  DEBUG_WARN_IF(!reference_node->is_leaf());

  // Used to find the query node's new upper bound
  double query_max_neighbor_distance = -1.0;
  std::vector<std::pair<double, size_t> > neighbors(knns_);

  // cycle through each query in the leaf node individually
  for (size_t query_index = query_node->begin(); 
       query_index < query_node->end(); query_index++) {
       
    // Get the query point from the matrix
    Vector query_point;
    queries_.MakeColumnVector(query_index, &query_point);
      
    // setting up the bounds for the k potential 
    // nearest neighbors.
    size_t ind = query_index*knns_;
    for(size_t i=0; i<knns_; i++) {
      neighbors[i]=std::make_pair(neighbor_distances_[ind+i],
				  neighbor_indices_[ind+i]);
    }
    // We'll do the same for the references
    // cycle through the reference nodes
    for (size_t reference_index = reference_node->begin(); 
	 reference_index < reference_node->end();
	 reference_index++) {

      // Confirm that points do not identify themselves as neighbors
      // in the monochromatic case
      if (likely(reference_node != query_node ||
		 reference_index != query_index)) {
	Vector reference_point;
	references_.MakeColumnVector(reference_index, 
				     &reference_point);
	// We'll use lapack to find the distance
	// between the two vectors
	double distance =
	  la::DistanceSqEuclidean(query_point, reference_point);
	// If the reference point is closer than
	// the current candidate, 
	// we'll update the candidate
	if (distance < neighbor_distances_[ind+knns_-1]) {
	  neighbors.push_back(std::make_pair(distance,
					     reference_index));
	}
      }
    } // for reference_index
 
    // sort the list to select the current top k potential neighbors
    std::sort(neighbors.begin(), neighbors.end());
    for(size_t i=0; i<knns_; i++) {
      neighbor_distances_[ind+i] = neighbors[i].first;
      neighbor_indices_[ind+i]  = neighbors[i].second;
    }
    neighbors.resize(knns_);
    // We need to find the upper bound distance for this query node
    if (neighbor_distances_[ind+knns_-1]
	> query_max_neighbor_distance) {
      query_max_neighbor_distance
	= neighbor_distances_[ind+knns_-1]; 
    }
  } // for query_index 
  
  // Update the upper bound for the query_node
  query_node->stat().set_max_distance_so_far(query_max_neighbor_distance);
         
} // ComputeBaseCase_
  
  
/**
 * The recursive function
 */
void ApproxNN::ComputeNeighborsRecursion_(
		   TreeType* query_node,
		   TreeType* reference_node, 
		   double lower_bound_distance) {

  DEBUG_ASSERT(query_node != NULL);
  DEBUG_ASSERT(reference_node != NULL);
 
 // Make sure the bounding information is correct
  DEBUG_ASSERT(lower_bound_distance == MinNodeDistSq_(query_node, 
						      reference_node));
    
  // if the node is farther than current lowerbound
  if (lower_bound_distance
      > query_node->stat().max_distance_so_far()) {
    // Pruned by distance
    number_of_prunes_++;
  } else if (query_node->is_leaf() && reference_node->is_leaf()) {
    // Base Case
    ComputeBaseCase_(query_node, reference_node);
  } else if (query_node->is_leaf()) {
     
    double left_distance
      = MinNodeDistSq_(query_node, reference_node->left());
    double right_distance
      = MinNodeDistSq_(query_node, reference_node->right());
      
    if (left_distance < right_distance) {
      ComputeNeighborsRecursion_(query_node, reference_node->left(), 
				 left_distance);
      ComputeNeighborsRecursion_(query_node, reference_node->right(), 
				 right_distance);
    } else {
      ComputeNeighborsRecursion_(query_node, reference_node->right(), 
				 right_distance);
      ComputeNeighborsRecursion_(query_node, reference_node->left(), 
				 left_distance);
    }
  } else if (reference_node->is_leaf()) {
     
    double left_distance
      = MinNodeDistSq_(query_node->left(), reference_node);
    double right_distance
      = MinNodeDistSq_(query_node->right(), reference_node);
      
    ComputeNeighborsRecursion_(query_node->left(), reference_node, 
			       left_distance);
    ComputeNeighborsRecursion_(query_node->right(), reference_node, 
			       right_distance);
      
    // We need to update the upper bound based on the new upper bounds of 
    // the children
    query_node->stat().set_max_distance_so_far(
	 max(query_node->left()->stat().max_distance_so_far(),
	     query_node->right()->stat().max_distance_so_far()));
  
  } else {
    // Recurse on both as above
      
    double left_distance
      = MinNodeDistSq_(query_node->left(), reference_node->left());
    double right_distance
      = MinNodeDistSq_(query_node->left(), reference_node->right());
      
    if (left_distance < right_distance) {
      ComputeNeighborsRecursion_(query_node->left(),
				 reference_node->left(), 
				 left_distance);
      ComputeNeighborsRecursion_(query_node->left(),
				 reference_node->right(), 
				 right_distance);
    } else {
      ComputeNeighborsRecursion_(query_node->left(),
				 reference_node->right(), 
				 right_distance);
      ComputeNeighborsRecursion_(query_node->left(),
				 reference_node->left(), 
				 left_distance);
    }

    left_distance
      = MinNodeDistSq_(query_node->right(), reference_node->left());
    right_distance
      = MinNodeDistSq_(query_node->right(), reference_node->right());
      
    if (left_distance < right_distance) {
      ComputeNeighborsRecursion_(query_node->right(),
				 reference_node->left(), 
				 left_distance);
      ComputeNeighborsRecursion_(query_node->right(),
				 reference_node->right(), 
				 right_distance);
    } else {
      ComputeNeighborsRecursion_(query_node->right(),
				 reference_node->right(), 
				 right_distance);
      ComputeNeighborsRecursion_(query_node->right(),
				 reference_node->left(), 
				 left_distance);
    }
      
    // Update the upper bound as above
    query_node->stat().set_max_distance_so_far(
        max(query_node->left()->stat().max_distance_so_far(),
	    query_node->right()->stat().max_distance_so_far()));
      
  }
} // ComputeNeighborsRecursion_



/**
 * Performs exhaustive approximate computation
 * between two nodes.
 */
void ApproxNN::ComputeApproxBaseCase_(TreeType* query_node,
				      TreeType* reference_node) {
   
  // Check that the pointers are not NULL
  DEBUG_ASSERT(query_node != NULL);
  DEBUG_ASSERT(reference_node != NULL);

  // Obtain the number of samples to be obtained
  size_t set_size
    = reference_node->end() - reference_node->begin();
  size_t sample_size = sample_sizes_[set_size - 1];
  DEBUG_ASSERT_MSG(sample_size <= set_size,
		   "n = %zud, N = %zud",
		   sample_size, set_size);

  size_t query_samples_needed
    = min_samples_per_q_ - query_node->stat().samples();

  sample_size = min(sample_size, query_samples_needed);
  DEBUG_WARN_IF(sample_size > sample_limit_);

  // Used to find the query node's new upper bound
  double query_max_neighbor_distance = -1.0;
  std::vector<std::pair<double, size_t> > neighbors(knns_);
  for (size_t query_index = query_node->begin(); 
       query_index < query_node->end(); query_index++) {
       
    // Get the query point from the matrix
    Vector query_point;
    queries_.MakeColumnVector(query_index, &query_point);
      
    size_t ind = query_index*knns_;
    for(size_t i=0; i<knns_; i++) {
      neighbors[i]=std::make_pair(neighbor_distances_[ind+i],
				  neighbor_indices_[ind+i]);
    }
    // We'll do the same for the references
    // but on the sample size number of points

    // Here we need to permute the reference set randomly
    for (size_t i = 0; i < sample_size; i++) {				
      // pick a random reference point
      size_t reference_index = reference_node->begin()
	+ math::RandInt(set_size);
      DEBUG_ASSERT(reference_index < reference_node->end());

      // Confirm that points do not identify themselves as neighbors
      // in the monochromatic case
      if (likely(reference_node != query_node ||
		 reference_index != query_index)) {
	Vector reference_point;
	references_.MakeColumnVector(reference_index, &reference_point);
	// We'll use lapack to find the distance between the two vectors
	double distance =
	  la::DistanceSqEuclidean(query_point, reference_point);
	// If the reference point is closer than the current candidate, 
	// we'll update the candidate
	if (distance < neighbor_distances_[ind+knns_-1]) {
	  neighbors.push_back(std::make_pair(distance, reference_index));
	}
      } else {
	--i;
      }
    } // for reference_index

    std::sort(neighbors.begin(), neighbors.end());
    for(size_t i=0; i<knns_; i++) {
      neighbor_distances_[ind+i] = neighbors[i].first;
      neighbor_indices_[ind+i]  = neighbors[i].second;
    }
    neighbors.resize(knns_);
    // We need to find the upper bound distance for this query node
    if (neighbor_distances_[ind+knns_-1] > query_max_neighbor_distance) {
      query_max_neighbor_distance = neighbor_distances_[ind+knns_-1]; 
    }
  } // for query_index 
    // Update the upper bound for the query_node
  query_node->stat().set_max_distance_so_far(query_max_neighbor_distance);

  // update the number of points considered and points sampled
  query_node->stat().add_total_points(set_size);
  query_node->stat().add_samples(sample_size);
} // ComputeApproxBaseCase_


/**
 * The recursive function for the approximate computation
 */
void ApproxNN::ComputeApproxRecursion_(TreeType* query_node,
				       TreeType* reference_node, 
				       double lower_bound_distance) {
  // A DEBUG statement with no predefined message
  DEBUG_ASSERT(query_node != NULL);
  DEBUG_ASSERT(reference_node != NULL);
  // Make sure the bounding information is correct
  DEBUG_ASSERT(lower_bound_distance == MinNodeDistSq_(query_node, 
						      reference_node));
  // Check if enough sampling is already done for
  // this query node.
  if (is_done(query_node)) {
    query_node->stat().add_total_points(reference_node->end()
					- reference_node->begin());
  } else if (lower_bound_distance
	     > query_node->stat().max_distance_so_far()) {
    // Pruned by distance
    number_of_prunes_++;

    // since we pruned this node, we can say that we summarized
    // this node exactly
    size_t reference_size
      = reference_node->end() - reference_node->begin();
    query_node->stat().add_total_points(reference_size);
    // query_node->stat().add_samples(reference_size);

    // what if we still plan to sample equally from here and 
    // not selective sampling
    query_node->stat().add_samples(sample_sizes_[reference_size -1]);

  } else if (query_node->is_leaf() && reference_node->is_leaf()) {
    // Base Case
    // first check if we can do exact. If so then we do so
    // and add the number of samples encountered.
    ComputeBaseCase_(query_node, reference_node);
    size_t reference_size
      = reference_node->end() - reference_node->begin();
    query_node->stat().add_total_points(reference_size);
    query_node->stat().add_samples(reference_size);

  } else if (reference_node->is_leaf()) {
    // Only reference is a leaf 
    double left_distance
      = MinNodeDistSq_(query_node->left(), reference_node);
    double right_distance
      = MinNodeDistSq_(query_node->right(), reference_node);

    // Passing the information down to the children if it 
    // encountered some pruning earlier
    DEBUG_ASSERT_MSG(query_node->left()->stat().total_points()
		     == query_node->right()->stat().total_points(),
		     "While getting: Left:%zud, Right:%zud",
		     query_node->left()->stat().total_points(),
		     query_node->right()->stat().total_points());

    // if the parent has encountered extra points, pass
    // that information down to the children.
    size_t extra_points_encountered
      = query_node->stat().total_points()
      - query_node->left()->stat().total_points();
    DEBUG_ASSERT(extra_points_encountered > -1);

    if (extra_points_encountered > 0) {
      query_node->left()->stat().add_total_points(extra_points_encountered);
      query_node->right()->stat().add_total_points(extra_points_encountered);
      size_t extra_points_sampled
	= query_node->stat().samples()
	- min(query_node->left()->stat().samples(),
	      query_node->right()->stat().samples());
      DEBUG_ASSERT(extra_points_sampled > -1);
      query_node->left()->stat().add_samples(extra_points_sampled);
      query_node->right()->stat().add_samples(extra_points_sampled);
    }

    // recurse down the query tree      
    ComputeApproxRecursion_(query_node->left(), reference_node, 
			    left_distance);
    ComputeApproxRecursion_(query_node->right(), reference_node, 
			    right_distance);
      
    // We need to update the upper bound based on the new upper bounds of 
    // the children
    query_node->stat().set_max_distance_so_far(
        max(query_node->left()->stat().max_distance_so_far(),
	    query_node->right()->stat().max_distance_so_far()));

    // updating the number of points considered
    // and number of samples taken

    // both the children of the query node have encountered
    // the same number of reference points. So making sure of
    // that.
    DEBUG_ASSERT_MSG(query_node->left()->stat().total_points()
		     == query_node->right()->stat().total_points(),
		     "While setting: Left:%zud, Right:%zud",
		     query_node->left()->stat().total_points(),
		     query_node->right()->stat().total_points());

    DEBUG_ASSERT(query_node->stat().total_points()
		 < query_node->left()->stat().total_points());
    query_node->stat().set_total_points(
        query_node->left()->stat().total_points());

    // the number of samples made for each of the query points
    // is actually the minimum of both the children. And we 
    // are setting it instead of adding because we don't want
    // to have repetitions (since the information goes bottom up)
    DEBUG_ASSERT(query_node->stat().samples() < 
		 min(query_node->left()->stat().samples(),
		     query_node->right()->stat().samples()));
    query_node->stat().set_samples(min(query_node->left()->stat().samples(),
				       query_node->right()->stat().samples()));

  } else if (is_base(reference_node)) {
    // if the reference set is small enough to be
    // approximated by sampling.
    ComputeApproxBaseCase_(query_node, reference_node);

  } else if (is_almost_satisfied(query_node)) {
    // query node has almost enough samples,
    // just pick some samples from the reference
    // set.
    ComputeApproxBaseCase_(query_node, reference_node);

  } else if (query_node->is_leaf()) {
    // Only query is a leaf
      
    // We'll order the computation by distance 
    double left_distance
      = MinNodeDistSq_(query_node, reference_node->left());
    double right_distance
      = MinNodeDistSq_(query_node, reference_node->right());

    if (left_distance < right_distance) {
      ComputeApproxRecursion_(query_node, reference_node->left(), 
			      left_distance);
      ComputeApproxRecursion_(query_node, reference_node->right(), 
			      right_distance);
    } else {
      ComputeApproxRecursion_(query_node, reference_node->right(), 
			      right_distance);
      ComputeApproxRecursion_(query_node, reference_node->left(), 
			      left_distance);
    }
  } else {

    // Recurse on both as above
    double left_distance
      = MinNodeDistSq_(query_node->left(), reference_node->left());
    double right_distance
      = MinNodeDistSq_(query_node->left(), reference_node->right());

    // Passing the information down to the children if it 
    // encountered some pruning earlier
    DEBUG_ASSERT_MSG(query_node->left()->stat().total_points()
		     == query_node->right()->stat().total_points(),
		     "While getting: Left:%zud, Right:%zud",
		     query_node->left()->stat().total_points(),
		     query_node->right()->stat().total_points());

    // if the parent has encountered extra points, pass
    // that information down to the children.
    size_t extra_points_encountered
      = query_node->stat().total_points()
      - query_node->left()->stat().total_points();
    DEBUG_ASSERT(extra_points_encountered > -1);

    if (extra_points_encountered > 0) {
      query_node->left()->stat().add_total_points(extra_points_encountered);
      query_node->right()->stat().add_total_points(extra_points_encountered);
      size_t extra_points_sampled
	= query_node->stat().samples()
	- min(query_node->left()->stat().samples(),
	      query_node->right()->stat().samples());
      DEBUG_ASSERT(extra_points_sampled > -1);
      query_node->left()->stat().add_samples(extra_points_sampled);
      query_node->right()->stat().add_samples(extra_points_sampled);
    }
      
    if (left_distance < right_distance) {
      ComputeApproxRecursion_(query_node->left(),
			      reference_node->left(), 
			      left_distance);
      ComputeApproxRecursion_(query_node->left(),
			      reference_node->right(), 
			      right_distance);
    } else {
      ComputeApproxRecursion_(query_node->left(),
			      reference_node->right(), 
			      right_distance);
      ComputeApproxRecursion_(query_node->left(),
			      reference_node->left(), 
			      left_distance);
    }

    left_distance
      = MinNodeDistSq_(query_node->right(), reference_node->left());
    right_distance
      = MinNodeDistSq_(query_node->right(), reference_node->right());
      
    if (left_distance < right_distance) {
      ComputeApproxRecursion_(query_node->right(),
			      reference_node->left(), 
			      left_distance);
      ComputeApproxRecursion_(query_node->right(),
			      reference_node->right(), 
			      right_distance);
    } else {
      ComputeApproxRecursion_(query_node->right(),
			      reference_node->right(), 
			      right_distance);
      ComputeApproxRecursion_(query_node->right(),
			      reference_node->left(), 
			      left_distance);
    }
      
    // Update the upper bound as above
    query_node->stat().set_max_distance_so_far(
        max(query_node->left()->stat().max_distance_so_far(),
	    query_node->right()->stat().max_distance_so_far()));

    // both the children of the query node have encountered
    // the same number of reference points. So making sure of
    // that.
    DEBUG_ASSERT_MSG(query_node->left()->stat().total_points()
		     == query_node->right()->stat().total_points(),
		     "While setting: Left:%zud, Right:%zud",
		     query_node->left()->stat().total_points(),
		     query_node->right()->stat().total_points());

    DEBUG_ASSERT(query_node->stat().total_points() < 
		 query_node->left()->stat().total_points());
    query_node->stat().set_total_points(
        query_node->left()->stat().total_points());

    // the number of samples made for each of the query points
    // is actually the minimum of both the children. And we 
    // are setting it instead of adding because we don't want
    // to have repetitions (since the information goes bottom up)
    DEBUG_ASSERT(query_node->stat().samples() < 
		 min(query_node->left()->stat().samples(),
		     query_node->right()->stat().samples()));
    query_node->stat().set_samples(
        min(query_node->left()->stat().samples(),
	    query_node->right()->stat().samples()));
  }
} // ComputeApproxRecursion_

  
/**
 * Setup the class and build the trees.
 * Note: we are initializing with const references to prevent 
 * local copies of the data.
 * This Init is just for performing exact nearest neighbor. 
 */
void ApproxNN::Init(const Matrix& queries_in,
		    const Matrix& references_in,
		    struct datanode* module_in) {
    
  // set the module
  module_ = module_in;
    
  // track the number of prunes
  number_of_prunes_ = 0;
    
  // Get the leaf size from the module
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  // Make sure the leaf size is valid
  DEBUG_ASSERT(leaf_size_ > 0);
    
  // Copy the matrices to the class members since they will be rearranged.  
  queries_.Copy(queries_in);
  references_.Copy(references_in);
    
  // The data sets need to have the same number of points
  DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());
    
  // keep a track of the dataset
  fx_param_int(module_, "dim", queries_.n_rows());
  fx_param_int(module_, "qsize", queries_.n_cols());
  fx_param_int(module_, "rsize", references_.n_cols());

  // K-nearest neighbors initialization
  knns_ = fx_param_int(module_, "knns", 1);
  
  // Initialize the list of nearest neighbor candidates
  neighbor_indices_.Init(queries_.n_cols() * knns_);
    
  // Initialize the vector of upper bounds for each point.  
  neighbor_distances_.Init(queries_.n_cols() * knns_);
  neighbor_distances_.SetAll(DBL_MAX);

  // We'll time tree building
  fx_timer_start(module_, "tree_building");

  // This call makes each tree from a matrix, leaf size, and two arrays 
  // that record the permutation of the data points
  // Instead of NULL, it is possible to specify an array new_from_old_
  query_tree_ 
    = tree::MakeKdTreeMidpoint<TreeType>(queries_,
					 leaf_size_, 
					 &old_from_new_queries_,
					 NULL);
  reference_tree_
    = tree::MakeKdTreeMidpoint<TreeType>(references_, 
					 leaf_size_,
					 &old_from_new_references_,
					 NULL);
    
  // Stop the timer we started above
  fx_timer_stop(module_, "tree_building");

  // initializing the sample_sizes_
  sample_sizes_.Init();
} // Init



// Initializing for the monochromatic case
void ApproxNN::Init(const Matrix& references_in,
		    struct datanode* module_in) {
    
  // set the module
  module_ = module_in;
    
  // track the number of prunes
  number_of_prunes_ = 0;
    
  // Get the leaf size from the module
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  // Make sure the leaf size is valid
  DEBUG_ASSERT(leaf_size_ > 0);
    
  // Copy the matrices to the class members since they will be rearranged.  
  references_.Copy(references_in);
  queries_.Alias(references_);
    
  // keep a track of the dataset
  fx_param_int(module_, "dim", references_.n_rows());
  fx_param_int(module_, "qsize", references_.n_cols());
  fx_param_int(module_, "rsize", references_.n_cols());

  // K-nearest neighbors initialization
  knns_ = fx_param_int(module_, "knns", 1);
  
  // Initialize the list of nearest neighbor candidates
  neighbor_indices_.Init(references_.n_cols() * knns_);
    
  // Initialize the vector of upper bounds for each point.  
  neighbor_distances_.Init(references_.n_cols() * knns_);
  neighbor_distances_.SetAll(DBL_MAX);

  // We'll time tree building
  fx_timer_start(module_, "tree_building");

  // This call makes each tree from a matrix, leaf size, and two arrays 
  // that record the permutation of the data points
  // Instead of NULL, it is possible to specify an array new_from_old_
  query_tree_ = NULL;
  reference_tree_
    = tree::MakeKdTreeMidpoint<TreeType>(references_, 
					 leaf_size_,
					 &old_from_new_references_,
					 NULL);
    
  fx_timer_stop(module_, "tree_building");

  // initializing stuff not needed here
  sample_sizes_.Init();
  old_from_new_queries_.Init();
} // Init

/**
 * Initializes the AllNN structure for naive computation.  
 * This means that we simply ignore the tree building.
 */
void ApproxNN::InitNaive(const Matrix& queries_in, 
			 const Matrix& references_in,
			 size_t knns){
  
  queries_.Copy(queries_in);
  references_.Copy(references_in);
  knns_=knns;
    
  DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());
    
  neighbor_indices_.Init(queries_.n_cols()*knns_);
  neighbor_distances_.Init(queries_.n_cols()*knns_);
  neighbor_distances_.SetAll(DBL_MAX);
    
  // The only difference is that we set leaf_size_ to be large enough 
  // that each tree has only one node
  leaf_size_ = max(queries_.n_cols(), references_.n_cols());
    
  query_tree_
    = tree::MakeKdTreeMidpoint<TreeType>(queries_, 
					 leaf_size_,
					 &old_from_new_queries_,
					 NULL);
  reference_tree_
    = tree::MakeKdTreeMidpoint<TreeType>(references_,
					 leaf_size_,
					 &old_from_new_references_,
					 NULL);

  // initialiazing the sample_sizes_
  sample_sizes_.Init();
} // InitNaive

  
// Initializing for the naive computation for a
// monochromatic dataset
void ApproxNN::InitNaive(const Matrix& references_in,
			 size_t knns){
    
  references_.Copy(references_in);
  queries_.Alias(references_);
  knns_=knns;

  neighbor_indices_.Init(references_.n_cols()*knns_);
  neighbor_distances_.Init(references_.n_cols()*knns_);
  neighbor_distances_.SetAll(DBL_MAX);
    
  // The only difference is that we set leaf_size_ to be large enough 
  // that each tree has only one node
  leaf_size_ = references_.n_cols();
    
  query_tree_ = NULL;
  reference_tree_
    = tree::MakeKdTreeMidpoint<TreeType>(references_,
					 leaf_size_,
					 &old_from_new_references_,
					 NULL);

  // initialiazing stuff not needed here
  sample_sizes_.Init();
  old_from_new_queries_.Init();
} // InitNaive


/**
 * Initialization for the Rank-Approximate nearest neighbor
 * computation for which we store the number of samples 
 * to be made for each size of a dataset.
 */
void ApproxNN::InitApprox(const Matrix& queries_in,
			  const Matrix& references_in,
			  struct datanode* module_in) {
    
  // set the module
  module_ = module_in;

  // Check if the probability is <=1
  double alpha = fx_param_double(module_, "alpha", 1.0);
  DEBUG_ASSERT(alpha <= 1.0);
    
  // track the number of prunes
  number_of_prunes_ = 0;
    
  // Get the leaf size from the module
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  // Make sure the leaf size is valid
  DEBUG_ASSERT(leaf_size_ > 0);

  // Getting the sample_limit
  sample_limit_ = fx_param_int(module_, "sample_limit", 20);
    
  // Copy the matrices to the class members since they will be rearranged.  
  queries_.Copy(queries_in);
  references_.Copy(references_in);
    
  // The data sets need to have the same number of points
  DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());
    
  // keep a track of the dataset
  fx_param_int(module_, "dim", queries_.n_rows());
  fx_param_int(module_, "qsize", queries_.n_cols());
  fx_param_int(module_, "rsize", references_.n_cols());

  // K-nearest neighbors initialization
  knns_ = fx_param_int(module_, "knns", 1);
  
  // Initialize the list of nearest neighbor candidates
  neighbor_indices_.Init(queries_.n_cols() * knns_);
    
  // Initialize the vector of upper bounds for each point.  
  neighbor_distances_.Init(queries_.n_cols() * knns_);
  neighbor_distances_.SetAll(DBL_MAX);

  // We'll time tree building
  fx_timer_start(module_, "tree_building_approx");

  // Making the trees
  query_tree_
    = tree::MakeKdTreeMidpoint<TreeType>(queries_,
					 leaf_size_, 
					 &old_from_new_queries_,
					 NULL);
  reference_tree_
    = tree::MakeKdTreeMidpoint<TreeType>(references_, 
					 leaf_size_,
					 &old_from_new_references_,
					 NULL);
    
  // Stop the timer we started above
  fx_timer_stop(module_, "tree_building_approx");

  // We will time the initialization of the sample size
  // table
  fx_timer_start(module_, "computing_sample_sizes");

  // initialize the sample_sizes array
  sample_sizes_.Init(references_.n_cols());

  // compute the sample sizes
  epsilon_ = fx_param_double(module_, "epsilon", 0.0);
  rank_approx_ = (size_t) (epsilon_
			    * (double) references_.n_cols()
			    / 100.0);
  NOTIFY("Rank Approximation: %2.3f%% or %zud"
	 " with Probability:%1.2f",
	 epsilon_, rank_approx_, alpha);

  if (knns_ == 1)
    ComputeSampleSizes_(rank_approx_, alpha, &sample_sizes_);
  else
    ComputeSampleSizes_(rank_approx_, alpha, knns_, &sample_sizes_);

  NOTIFY("n: %zud", sample_sizes_[references_.n_cols() -1]);
  fx_timer_stop(module_, "computing_sample_sizes");    

  // initializing the minimum samples required per
  // query to hold the probability bound
  min_samples_per_q_ = sample_sizes_[references_.n_cols() -1];

} // InitApprox

  
// InitApprox for the monochromatic case
void ApproxNN::InitApprox(const Matrix& references_in,
			  struct datanode* module_in) {
    
  // set the module
  module_ = module_in;

  // Check if the probability is <=1
  double alpha = fx_param_double(module_, "alpha", 1.0);
  DEBUG_ASSERT(alpha <= 1.0);
    
  // track the number of prunes
  number_of_prunes_ = 0;
    
  // Get the leaf size from the module
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  // Make sure the leaf size is valid
  DEBUG_ASSERT(leaf_size_ > 0);

  // Getting the sample_limit
  sample_limit_ = fx_param_int(module_, "sample_limit", 20);
    
  // Copy the matrices to the class members 
  // since they will be rearranged.  
  references_.Copy(references_in);
  queries_.Alias(references_);
    
  // keep a track of the dataset
  fx_param_int(module_, "dim", references_.n_rows());
  fx_param_int(module_, "qsize", references_.n_cols());
  fx_param_int(module_, "rsize", references_.n_cols());

  // K-nearest neighbors initialization
  knns_ = fx_param_int(module_, "knns", 1);
  
  // Initialize the list of nearest neighbor candidates
  neighbor_indices_.Init(references_.n_cols() * knns_);
    
  // Initialize the vector of upper bounds for each point.  
  neighbor_distances_.Init(references_.n_cols() * knns_);
  neighbor_distances_.SetAll(DBL_MAX);

  // We'll time tree building
  fx_timer_start(module_, "tree_building_approx");

  // Making the trees
  query_tree_ = NULL;
  reference_tree_
    = tree::MakeKdTreeMidpoint<TreeType>(references_, 
					 leaf_size_,
					 &old_from_new_references_,
					 NULL);
    
  // Stop the timer we started above
  fx_timer_stop(module_, "tree_building_approx");

  // We will time the initialization of the sample size
  // table
  fx_timer_start(module_, "computing_sample_sizes");

  // initialize the sample_sizes array
  sample_sizes_.Init(references_.n_cols());

  // compute the sample sizes
  epsilon_ = fx_param_double(module_, "epsilon", 0.0);
  rank_approx_ = (size_t) (epsilon_
			    * (double) references_.n_cols()
			    / 100.0);
  NOTIFY("Rank Approximation: %2.3f%% or %zud"
	 " with Probability:%1.2f",
	 epsilon_, rank_approx_, alpha);
  if (knns_ == 1)
    ComputeSampleSizes_(rank_approx_, alpha, &sample_sizes_);
  else
    ComputeSampleSizes_(rank_approx_, alpha, knns_, &sample_sizes_);

  fx_timer_stop(module_, "computing_sample_sizes");

  // initializing the minimum samples required per
  // query to hold the probability bound
  min_samples_per_q_ = sample_sizes_[references_.n_cols() -1];

  // initializing stuff not used here
  old_from_new_queries_.Init();

} // InitApprox
  

/**
 * Computes the exact nearest neighbors and stores them in *results
 */
void ApproxNN::ComputeNeighbors(ArrayList<size_t>* resulting_neighbors,
				ArrayList<double>* distances) {

  // Start on the root of each tree
  if (query_tree_!=NULL) {
    ComputeNeighborsRecursion_(query_tree_, reference_tree_, 
			       MinNodeDistSq_(query_tree_,
					      reference_tree_));
  } else {
    ComputeNeighborsRecursion_(reference_tree_, reference_tree_, 
			       MinNodeDistSq_(reference_tree_,
					      reference_tree_));
  }
 
  // We need to initialize the results list before filling it
  resulting_neighbors->Init(neighbor_indices_.size());
  distances->Init(neighbor_distances_.length());
    
  // We need to map the indices back from how they have 
  // been permuted
  if (query_tree_ != NULL) {
    for (size_t i = 0; i < neighbor_indices_.size(); i++) {
      (*resulting_neighbors)[old_from_new_queries_[i/knns_]*knns_+ i%knns_]
	= old_from_new_references_[neighbor_indices_[i]];
      (*distances)[old_from_new_queries_[i/knns_]*knns_+ i%knns_]
	= neighbor_distances_[i];
    }
  } else {
    for (size_t i = 0; i < neighbor_indices_.size(); i++) {
      (*resulting_neighbors)[old_from_new_references_[i/knns_]
			     *knns_+ i%knns_]
	= old_from_new_references_[neighbor_indices_[i]];
      (*distances)[old_from_new_references_[i/knns_]*knns_+ i%knns_]
	= neighbor_distances_[i];
    }
  }
} // ComputeNeighbors
  
  
  
/**
 * Does the entire computation naively computing the exact
 * nearest neighbors
 */
void ApproxNN::ComputeNaive(ArrayList<size_t>* resulting_neighbors,
			    ArrayList<double>*  distances) {

  if (query_tree_!=NULL) {
    ComputeBaseCase_(query_tree_, reference_tree_);
  } else {
    ComputeBaseCase_(reference_tree_, reference_tree_);
  }

  // The same code as above
  resulting_neighbors->Init(neighbor_indices_.size());
  distances->Init(neighbor_distances_.length());

  // We need to map the indices back from how they have 
  // been permuted
  if (query_tree_ != NULL) {
    for (size_t i = 0; i < neighbor_indices_.size(); i++) {
      (*resulting_neighbors)[old_from_new_queries_[i/knns_]*knns_+ i%knns_]
	= old_from_new_references_[neighbor_indices_[i]];
      (*distances)[old_from_new_queries_[i/knns_]*knns_+ i%knns_]
	= neighbor_distances_[i];
    }
  } else {
    for (size_t i = 0; i < neighbor_indices_.size(); i++) {
      (*resulting_neighbors)[old_from_new_references_[i/knns_]
			     *knns_+ i%knns_]
	= old_from_new_references_[neighbor_indices_[i]];
      (*distances)[old_from_new_references_[i/knns_]*knns_+ i%knns_]
	= neighbor_distances_[i];
    }
  }
} // ComputeNaive


/**
 * Does the entire computation to find the rank-approximate
 * nearest neighbors
 */
void ApproxNN::ComputeApprox(ArrayList<size_t>* resulting_neighbors,
			     ArrayList<double>*  distances) {

  // Start on the root of each tree
  if (query_tree_!=NULL) {
    ComputeApproxRecursion_(query_tree_, reference_tree_, 
			    MinNodeDistSq_(query_tree_,
					   reference_tree_));
    DEBUG_ASSERT_MSG(query_tree_->stat().total_points()
		     == references_.n_cols(),"N':%zud, N:%zud",
		     query_tree_->stat().total_points(),
		     references_.n_cols());
    DEBUG_ASSERT_MSG(query_tree_->stat().samples()
		     >= min_samples_per_q_,"n':%zud, n:%zud",
		     query_tree_->stat().samples(),
		     min_samples_per_q_);
    NOTIFY("n:%zud, N:%zud", query_tree_->stat().samples(),
	   query_tree_->stat().total_points());
  } else {
    ComputeApproxRecursion_(reference_tree_, reference_tree_, 
			    MinNodeDistSq_(reference_tree_,
					   reference_tree_));
  }
 
  // We need to initialize the results list before filling it
  resulting_neighbors->Init(neighbor_indices_.size());
  distances->Init(neighbor_distances_.length());
    
  // We need to map the indices back from how they have 
  // been permuted
  if (query_tree_ != NULL) {
    for (size_t i = 0; i < neighbor_indices_.size(); i++) {
      (*resulting_neighbors)[old_from_new_queries_[i/knns_]*knns_+ i%knns_]
	= old_from_new_references_[neighbor_indices_[i]];
      (*distances)[old_from_new_queries_[i/knns_]*knns_+ i%knns_]
	= neighbor_distances_[i];
    }
  } else {
    for (size_t i = 0; i < neighbor_indices_.size(); i++) {
      (*resulting_neighbors)[old_from_new_references_[i/knns_]
			     *knns_+ i%knns_]
	= old_from_new_references_[neighbor_indices_[i]];
      (*distances)[old_from_new_references_[i/knns_]*knns_+ i%knns_]
	= neighbor_distances_[i];
    }
  }
} // ComputeApprox


/**
 * Does the entire computation naively on the sample 
 * (no reference tree).
 */
void ApproxNN::ComputeApproxNoTree(ArrayList<size_t>* resulting_neighbors,
				   ArrayList<double>*  distances) {
  if (query_tree_!=NULL) {
    ComputeApproxBaseCase_(query_tree_, reference_tree_);
  } else {
    ComputeApproxBaseCase_(reference_tree_, reference_tree_);
  }

  // The same code as above
  resulting_neighbors->Init(neighbor_indices_.size());
  distances->Init(neighbor_distances_.length());

  // We need to map the indices back from how they have 
  // been permuted
  if (query_tree_ != NULL) {
    for (size_t i = 0; i < neighbor_indices_.size(); i++) {
      (*resulting_neighbors)[old_from_new_queries_[i/knns_]*knns_+ i%knns_]
	= old_from_new_references_[neighbor_indices_[i]];
      (*distances)[old_from_new_queries_[i/knns_]*knns_+ i%knns_]
	= neighbor_distances_[i];
    }
  } else {
    for (size_t i = 0; i < neighbor_indices_.size(); i++) {
      (*resulting_neighbors)[old_from_new_references_[i/knns_]
			     *knns_+ i%knns_] = old_from_new_references_[neighbor_indices_[i]];
      (*distances)[old_from_new_references_[i/knns_]*knns_+ i%knns_]
	= neighbor_distances_[i];
    }
  }
} // ComputeApproxNoTree
