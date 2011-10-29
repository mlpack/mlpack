/**
 * @file approx_max_ip.cc
 * @author Parikshit Ram
 *
 * This file implements the functions declared 
 * for the class ApproxMaxIP.
 */

#include "approx_max_ip.h"

double ApproxMaxIP::MaxNodeIP_(TreeType* reference_node) {

  // counting the split decisions 
  split_decisions_++;

  // compute maximum possible inner product 
  // between a point and a ball in terms of 
  // the ball's center and radius
  arma::vec q = queries_.col(query_);
  arma::vec centroid = reference_node->bound().center();

  // +1: Can be cached in the reference tree
  double c_norm = arma::norm(centroid, 2);

  assert(arma::norm(q, 2) == query_norms_(query_));

  double rad = std::sqrt(reference_node->bound().radius());

  double max_cos_qr = 1.0;

  if (mlpack::CLI::HasParam("approx_maxip/angle_prune")) { 
    // tighter bound of \max_{r \in B_p^R} <q,r> 
    //    = |q| \max_{r \in B_p^R} |r| cos <qr 
    //    \leq |q| \max_{r \in B_p^R} |r| \max_{r \in B_p^R} cos <qr 
    //    \leq |q| (|p|+R) if <qp \leq \max_r <pr
    //    \leq |q| (|p|+R) cos( <qp - \max_r <pr ) otherwise

    if (rad <= c_norm) {
      // +1
      double cos_qp = arma::dot(q, centroid) 
	/ (query_norms_(query_) * c_norm);
      double sin_qp = std::sqrt(1 - cos_qp * cos_qp);

      double max_sin_pr = rad / c_norm;
      double min_cos_pr = std::sqrt(1 - max_sin_pr * max_sin_pr);

      if (min_cos_pr > cos_qp) { // <qp \geq \max_r <pr
	// cos( <qp - <pr ) = cos <qp * cos <pr + sin <qp * sin <pr
	double cos_qp_max_pr = (cos_qp * min_cos_pr) 
	  + (sin_qp * max_sin_pr);

	// FIXIT: this should be made general to 
	// return negative values as well
	max_cos_qr = std::max(cos_qp_max_pr, 0.0);
      }
    } else { 
      ball_has_origin_++;
    }

  }

  // Otherwise :
  // simple bound of \max_{r \in B_p^R} <q,r> 
  //    = |q| \max_{r \in B_p^R} |r| cos <qr 
  //    \leq |q| \max_{r \in B_p^R} |r| \leq |q| (|p|+R)

  return (query_norms_(query_) * (c_norm + rad) * max_cos_qr);
}

double ApproxMaxIP::MaxNodeIP_(CTreeType* query_node,
			       TreeType* reference_node) {

  // counting the split decisions 
  split_decisions_++;

  // min_{q', q} cos <qq' = cos_w
  arma::vec q = query_node->bound().center();
  double cos_w = query_node->bound().radius();
  double sin_w = query_node->bound().radius_conjugate();

  // +1: Can cache it in the query tree
  double q_norm = arma::norm(q, 2);

  arma::vec centroid = reference_node->bound().center();

  // +1: can be cached in the reference tree
  double c_norm = arma::norm(centroid, 2);
  double rad = std::sqrt(reference_node->bound().radius());

  double max_cos_qp = 1.0;

  if (mlpack::CLI::HasParam("approx_maxip/angle_prune")) { 

    if (rad <= c_norm) {
      // cos <pq = cos_phi

      // +1
      double cos_phi = arma::dot(q, centroid) / (c_norm * q_norm);
      double sin_phi = std::sqrt(1 - cos_phi * cos_phi);

      // max_r sin <pr = sin_theta
      double sin_theta = rad / c_norm;
      double cos_theta = std::sqrt(1 - sin_theta * sin_theta);

      if ((cos_phi < cos_theta) && (cos_phi < cos_w)) { 
	// phi > theta and phi > w
	// computing cos(phi - theta)
	double cos_phi_theta 
	  = cos_phi * cos_theta + sin_phi * sin_theta;

	if (cos_phi_theta < cos_w) {
	  // phi - theta > w
	  // computing cos (phi - theta - w)
	  double cos_phi_theta_w = cos_phi_theta * cos_w;
	  cos_phi_theta_w
	    += (std::sqrt(1 - cos_phi_theta * cos_phi_theta)
		* sin_w);
	  max_cos_qp = std::max(cos_phi_theta_w, 0.0);
	}
      }
    } else {
      ball_has_origin_++;
    }
  }

  return ((c_norm + rad) * max_cos_qp);
}


/**
 * This function computes the probability of
 * a particular quantile given the set and sample sizes
 * Computes P(d_(1) <= d_(1+rank_approx))
 */
double ApproxMaxIP::ComputeProbability_(size_t set_size,
					size_t sample_size,
					size_t rank_approx) {
  double sum;
  arma::vec temp_a, temp_b;

  temp_a = arma::ones<arma::vec>(rank_approx + 1);
  temp_b = arma::ones<arma::vec>(rank_approx + 1);

  // calculating the temp_b
  size_t j = 1;
  size_t i = rank_approx - 1;

  do {
    double frac = (double)(set_size-(sample_size-1)-j)
      / (double)(set_size - j);
    temp_b(i) = temp_b(i+1) * frac;
    j++; i--;
  } while (i > 0);
  assert(j == rank_approx);

  // computing the sum and the product with n/N
  sum = arma::dot(temp_a, temp_b);
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
double ApproxMaxIP::ComputeProbability_(size_t set_size,
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
void ApproxMaxIP::ComputeSampleSizes_(size_t rank_approx,
				double alpha,
				arma::Col<size_t> *samples) {

  size_t set_size = samples->n_elem,
    n = samples->n_elem;

  double prob;
  assert(alpha <= 1.0);
  double beta = 0;

  // going through all values of sample sizes 
  // to find the minimum samples required to satisfy the 
  // desired bound
  do {
    n--;
    prob = ComputeProbability_(set_size,
                               n, rank_approx);
  } while (prob >= alpha);
  
  (*samples)(--set_size) = ++n;
  beta = (double) n / (double) (set_size+1);

  // set the sample sizes for every node size for fast-access
  // during the algorithm
  do {
    n = (size_t) (beta * (double)(set_size)) +1;
    (*samples)(--set_size) = n;
  } while (set_size > rank_approx);
  
  while (set_size > 0) {
    (*samples)(--set_size) = 1;
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
void ApproxMaxIP::ComputeSampleSizes_(size_t rank_approx,
				      double alpha,
				      size_t k,
				      arma::Col<size_t> *samples) {

  size_t set_size = samples->n_elem;
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
    (*samples)(--set_size) = n + 1;
    
    double beta = (double) (n+1) / (double) (set_size +1);
                        
    // set the sample sizes for every node size for fast-access
    // during the algorithm
    do {
      n = (size_t) (beta * (double)(set_size)) +1;
      (*samples)(--set_size) = n;
//     } while (set_size > rank_approx);
    } while (set_size > 0);
    

    // Maybe we do not do this, this is throwing things off
    // Let's try removing this
//     while (set_size > 0) {
//       set_size--;
//       (*samples)(set_size) = std::min(k, set_size + 1);
//     }
  } else {
    while (set_size > 0) {
      set_size--;
      (*samples)(set_size) = set_size +1;
    }
  }
}


/**
 * Performs exhaustive approximate computation
 * between two nodes.
 */
void ApproxMaxIP::ComputeApproxBaseCase_(TreeType* reference_node) {
   
  // Check that the pointers are not NULL
  assert(reference_node != NULL);
  if (!CLI::HasParam("approx_maxip/no_tree"))
    assert(reference_node->is_leaf() 
 	   || is_base(reference_node)
	   || is_almost_satisfied());

  // Obtain the number of samples to be obtained
  size_t set_size
    = reference_node->end() - reference_node->begin();
  size_t sample_size = sample_sizes_[set_size - 1];
  assert(sample_size <= set_size);

//   size_t query_samples_needed
//     = min_samples_per_q_ - query_node->stat().samples();

  sample_size = std::min(sample_size, query_samples_needed_);

  if (!CLI::HasParam("approx_maxip/no_tree")) {
//     printf("Leaf size: %zu, Sample size: %zu\n", 
// 	   set_size, sample_size); fflush(NULL);
    assert(sample_size <= sample_limit_);
  }


  // Get the query point from the matrix
  arma::vec q = queries_.unsafe_col(query_); 

  // We'll do the same for the references
  // but on the sample size number of points

  // Here we need to permute the reference set randomly
  for (size_t i = 0; i < sample_size; i++) {                          
    // pick a random reference point
    size_t reference_index = reference_node->begin()
      + math::RandInt(set_size);
    assert(reference_index < reference_node->end());

    arma::vec rpoint = references_.unsafe_col(reference_index);

    // We'll use arma to find the inner product of the two vectors
    // +1
    double ip = arma::dot(q, rpoint);
    // If the reference point is greater than the current candidate, 
    // we'll update the candidate
    size_t insert_position = SortValue(ip);
    if (insert_position != (size_t() -1)) {
      InsertNeighbor(insert_position, reference_index, ip);
    }
  } // for reference_index

  // for now the query lower bounds are accessed from 
  // the variable 'max_ips_(knns_ - 1, query_)'
  query_samples_needed_ -= sample_size;
  distance_computations_ += sample_size;

} // ComputeApproxBaseCase_


void ApproxMaxIP::ComputeBaseCase_(TreeType* reference_node) {
   
  assert(reference_node != NULL);
  assert(reference_node->is_leaf());
  assert(query_ >= 0);
  assert(query_ < queries_.n_cols);

    
  // Get the query point from the matrix
  arma::vec q = queries_.unsafe_col(query_); 

  // We'll do the same for the references
  for (size_t reference_index = reference_node->begin(); 
       reference_index < reference_node->end(); reference_index++) {

    arma::vec rpoint = references_.unsafe_col(reference_index);

    // We'll use arma to find the inner product of the two vectors
    // +1
    double ip = arma::dot(q, rpoint);
    // If the reference point is greater than the current candidate, 
    // we'll update the candidate
    size_t insert_position = SortValue(ip);
    if (insert_position != (size_t() -1)) {
      InsertNeighbor(insert_position, reference_index, ip);
    }
  } // for reference_index

  // for now the query lower bounds are accessed from 
  // the variable 'max_ips_(knns_ - 1, query_)'
  distance_computations_ 
    += reference_node->end() - reference_node->begin();
         
} // ComputeBaseCase_



size_t ApproxMaxIP::SortValue(double value) {

  // The first element in the list is the best neighbor.  We only want to
  // insert if the new distance is less than the last element in the list.
  if (value <  max_ips_(knns_ -1, query_))
    return (size_t() - 1); // Do not insert.

  // Search from the beginning.  This may not be the best way.
  for (size_t i = 0; i < knns_; i++) {
    if (value > max_ips_(i, query_))
      return i;
  }

  // Control should never reach here.
  return (size_t() - 1);
}

void ApproxMaxIP::InsertNeighbor(size_t pos, size_t point_ind, double value) {

  // We only memmove() if there is actually a need to shift something.
  if (pos < (knns_ - 1)) {
    size_t len = (knns_ - 1) - pos;

    memmove(max_ips_.colptr(query_) + (pos + 1),
	    max_ips_.colptr(query_) + pos,
	    sizeof(double) * len);
    memmove(max_ip_indices_.colptr(query_) + (pos + 1),
	    max_ip_indices_.colptr(query_) + pos,
	    sizeof(size_t) * len);
  }

  // Now put the new information in the right index.
  max_ips_(pos, query_) = value;
  max_ip_indices_(pos, query_) = point_ind;
}

void ApproxMaxIP::ComputeApproxBaseCase_(CTreeType* query_node,
					 TreeType* reference_node) {
   
  // Check that the pointers are not NULL
  assert(query_node != NULL);
  assert(reference_node != NULL);


  // Used to find the query node's new lower bound
  double query_worst_p_cos_pq = DBL_MAX;
  bool new_bound = false;
  size_t samples_needed_per_query
    = min_samples_per_q_ - query_node->stat().samples();

  // Iterating over the queries individually
  for (query_ = query_node->begin();
       query_ < query_node->end(); query_++) {

    // checking if this node has potential
    double query_to_node_max_ip = MaxNodeIP_(reference_node);

    if (query_to_node_max_ip > max_ips_(knns_ -1, query_)) {
      // this node has potential

      // Obtain the number of samples to be required
      query_samples_needed_ = samples_needed_per_query;
      ComputeApproxBaseCase_(reference_node);

    }

    double p_cos_pq = max_ips_(knns_ -1, query_)
      / query_norms_(query_);

    if (query_worst_p_cos_pq > p_cos_pq) {
      query_worst_p_cos_pq = p_cos_pq;
      new_bound = true;
    }
  } // for query_
  
  // Update the lower bound for the query_node
  if (new_bound) 
    query_node->stat().set_bound(query_worst_p_cos_pq);


  // update the number of points considered and points sampled
  size_t set_size
    = reference_node->end() - reference_node->begin();
  size_t sample_size = std::min(sample_sizes_[set_size - 1], 
				samples_needed_per_query);
  query_node->stat().add_total_points(set_size);
  query_node->stat().add_samples(sample_size);
} // ComputeApproxBaseCase_


void ApproxMaxIP::ComputeBaseCase_(CTreeType* query_node, 
				   TreeType* reference_node) {

  // Check that the pointers are not NULL
  assert(reference_node != NULL);
  assert(reference_node->is_leaf());
  assert(query_node != NULL);
  assert(query_node->is_leaf());

  // Used to find the query node's new lower bound
  double query_worst_p_cos_pq = DBL_MAX;
  bool new_bound = false;
    
  // Iterating over the queries individually
  for (query_ = query_node->begin();
       query_ < query_node->end(); query_++) {

    // checking if this node has potential
    double query_to_node_max_ip = MaxNodeIP_(reference_node);

    if (query_to_node_max_ip > max_ips_(knns_ -1, query_))
      // this node has potential
      ComputeBaseCase_(reference_node);

    double p_cos_pq = max_ips_(knns_ -1, query_)
      / query_norms_(query_);

    if (query_worst_p_cos_pq > p_cos_pq) {
      query_worst_p_cos_pq = p_cos_pq;
      new_bound = true;
    }
  } // for query_
  
  // Update the lower bound for the query_node
  if (new_bound) 
    query_node->stat().set_bound(query_worst_p_cos_pq);

} // ComputeBaseCase_


void ApproxMaxIP::CheckPrune(CTreeType* query_node, TreeType* ref_node) {

  size_t missed_nns = 0;
  double max_p_cos_pq = 0.0;
  double min_p_cos_pq = DBL_MAX;

  // Iterating over the queries individually
  for (query_ = query_node->begin();
       query_ < query_node->end(); query_++) {

    // Get the query point from the matrix
    arma::vec q = queries_.unsafe_col(query_);

    double p_cos_qp = max_ips_(knns_ -1, query_) / query_norms_(query_);
    if (min_p_cos_pq > p_cos_qp)
      min_p_cos_pq = p_cos_qp;

    // We'll do the same for the references
    for (size_t reference_index = ref_node->begin(); 
	 reference_index < ref_node->end(); reference_index++) {

      arma::vec r = references_.unsafe_col(reference_index);

      double ip = arma::dot(q, r);
      if (ip > max_ips_(knns_-1, query_))
	missed_nns++;

      double p_cos_pq = ip / query_norms_(query_);

      if (p_cos_pq > max_p_cos_pq)
	max_p_cos_pq = p_cos_pq;
      
    } // for reference_index
  } // for query_
  
  if (missed_nns > 0 || query_node->stat().bound() != min_p_cos_pq) 
    printf("Prune %zu - Missed candidates: %zu\n"
	   "QLBound: %lg, ActualQLBound: %lg\n"
	   "QRBound: %lg, ActualQRBound: %lg\n",
	   number_of_prunes_, missed_nns,
	   query_node->stat().bound(), min_p_cos_pq, 
	   MaxNodeIP_(query_node, ref_node), max_p_cos_pq);

}


void ApproxMaxIP::CheckPrune(TreeType* ref_node) {

  size_t missed_nns = 0;
  double max_ip = 0.0;

  // Get the query point from the matrix
  arma::vec q = queries_.unsafe_col(query_);

  double min_ip = max_ips_(knns_ -1, query_); 

  // We'll do the same for the references
  for (size_t reference_index = ref_node->begin(); 
       reference_index < ref_node->end(); reference_index++) {

    arma::vec r = references_.unsafe_col(reference_index);

    double ip_r = arma::dot(q, r);
    if (ip_r > min_ip)
      missed_nns++;

    if (ip_r > max_ip)
      max_ip = ip_r;
    
  } // for reference_index

  if (missed_nns > 0) 
    printf("Prune %zu - Missed candidates: %zu\n"
	   "QLBound: %lg, QRBound: %lg, ActualQRBound: %lg\n",
	   number_of_prunes_, missed_nns,
	   min_ip, MaxNodeIP_(ref_node), max_ip);

}

void ApproxMaxIP::ComputeApproxRecursion_(TreeType* reference_node, 
					  double upper_bound_ip) {

  assert(reference_node != NULL);
  //  assert(upper_bound_ip == MaxNodeIP_(reference_node));


  // check if the query has enough number of samples
  if (!is_done()) {
    if (upper_bound_ip < max_ips_(knns_ -1, query_)) { 
      // Pruned by distance
      number_of_prunes_++;

      if (CLI::HasParam("approx_maxip/check_prune"))
	CheckPrune(reference_node);

      query_samples_needed_ 
	-= sample_sizes_[reference_node->end()
			 - reference_node->begin() - 1];

    } else if (reference_node->is_leaf()) {
	// base case for the single tree case
      ComputeBaseCase_(reference_node);
      query_samples_needed_
	-= (reference_node->end() - reference_node->begin());

      // trying to see if this was the issue (DIDN'T WORK)
      // ComputeApproxBaseCase_(reference_node);

    } else if (is_base(reference_node)) {
      // base case for the approximate case
      ComputeApproxBaseCase_(reference_node);

    } else if (is_almost_satisfied()) {
      // base case for the approximate case
      ComputeApproxBaseCase_(reference_node);
    }  else {
      // Recurse on both as above
      double left_ip = MaxNodeIP_(reference_node->left());
      double right_ip = MaxNodeIP_(reference_node->right());
      
      if (left_ip > right_ip) {
	ComputeApproxRecursion_(reference_node->left(), 
				left_ip);
	ComputeApproxRecursion_(reference_node->right(),
				right_ip);
      } else {
	ComputeApproxRecursion_(reference_node->right(),
				right_ip);
	ComputeApproxRecursion_(reference_node->left(), 
				left_ip);
      }
    }    
  } else {
//     assert(query_samples_needed_ <= 0);
  }
} // ComputeApproxRecursion_



void ApproxMaxIP::ComputeApproxRecursion_(CTreeType* query_node,
					  TreeType* reference_node, 
					  double upper_bound_p_cos_pq) {

  assert(query_node != NULL);
  assert(reference_node != NULL);
  //assert(upper_bound_p_cos_pq == MaxNodeIP_(query_node, reference_node));

  if (is_done(query_node)) {
    query_node->stat().add_total_points(reference_node->end()
					- reference_node->begin());

  } else if (upper_bound_p_cos_pq < query_node->stat().bound()) { 
    // Pruned
    number_of_prunes_++;

    if (CLI::HasParam("approx_maxip/check_prune"))
      CheckPrune(query_node, reference_node);

    // add appropriate effective number of samples
    size_t ref_size
      = reference_node->end() - reference_node->begin();
    query_node->stat().add_total_points(ref_size);
    query_node->stat().add_samples(sample_sizes_[ref_size -1]);

  } else if (reference_node->is_leaf() 
	     || is_base(reference_node)
	     || is_almost_satisfied(query_node)) {

    if (query_node->is_leaf()) {

      if (reference_node->is_leaf()) {
	// Base case	
	ComputeBaseCase_(query_node, reference_node);
	// add appropriate effective number of samples
	size_t ref_size
	  = reference_node->end() - reference_node->begin();
	query_node->stat().add_total_points(ref_size);
	query_node->stat().add_samples(ref_size);

      } else {
	// Approx base case
	ComputeApproxBaseCase_(query_node, reference_node);

      }
    } else {
      // go down the query tree

      double left_p_cos_pq
	= MaxNodeIP_(query_node->left(), reference_node);
      double right_p_cos_pq
	= MaxNodeIP_(query_node->right(), reference_node);

      // passing sampling information to the children
      assert(query_node->left()->stat().total_points()
	     == query_node->right()->stat().total_points());


      // if the parent has encountered extra points, pass
      // that information down to the children.
      size_t extra_points_encountered
	= query_node->stat().total_points()
	- query_node->left()->stat().total_points();
      assert(extra_points_encountered >= 0);

      if (extra_points_encountered > 0) {
	query_node->left()->stat().add_total_points(extra_points_encountered);
	query_node->right()->stat().add_total_points(extra_points_encountered);
	size_t extra_points_sampled
	  = query_node->stat().samples()
	  - std::min(query_node->left()->stat().samples(),
		     query_node->right()->stat().samples());
	assert(extra_points_sampled >= 0);
	query_node->left()->stat().add_samples(extra_points_sampled);
	query_node->right()->stat().add_samples(extra_points_sampled);
      }

     
      // recurse down the query_tree 
      ComputeApproxRecursion_(query_node->left(), reference_node, 
			      left_p_cos_pq);
      ComputeApproxRecursion_(query_node->right(), reference_node, 
			      right_p_cos_pq);
      

      // save sampling information from the children
      assert(query_node->left()->stat().total_points()
	     == query_node->right()->stat().total_points());

      assert(query_node->left()->stat().total_points()
	     >  query_node->stat().total_points());
      query_node->stat().set_total_points(
	  query_node->left()->stat().total_points());

//       printf("%zu: L:%zu, R:%zu\n", query_node->stat().samples(),
// 	     query_node->left()->stat().samples(),
// 	     query_node->right()->stat().samples()); fflush(NULL);


      assert(query_node->stat().samples() <= 
	     std::min(query_node->left()->stat().samples(),
		      query_node->right()->stat().samples()));
      query_node->stat().set_samples(std::min(query_node->left()->stat().samples(),
	  query_node->right()->stat().samples()));


      // We need to update the upper bound based on the new upper bounds of 
      // the children
      query_node->stat().set_bound(std::min(query_node->left()->stat().bound(),
          query_node->right()->stat().bound()));

    } // query tree descend

  } else if (query_node->is_leaf()) {
    // Only query is a leaf
      
    // We'll order the computation by distance 
    double left_p_cos_pq = MaxNodeIP_(query_node,
				      reference_node->left());
    double right_p_cos_pq = MaxNodeIP_(query_node,
				       reference_node->right());
      
    if (left_p_cos_pq > right_p_cos_pq) {
      ComputeApproxRecursion_(query_node, reference_node->left(), 
				 left_p_cos_pq);
      ComputeApproxRecursion_(query_node, reference_node->right(), 
				 right_p_cos_pq);
    } else {
      ComputeApproxRecursion_(query_node, reference_node->right(), 
				 right_p_cos_pq);
      ComputeApproxRecursion_(query_node, reference_node->left(), 
				 left_p_cos_pq);
    }
  } else {
    // Recurse on both as above
    double left_p_cos_pq = MaxNodeIP_(query_node->left(), 
				      reference_node->left());
    double right_p_cos_pq = MaxNodeIP_(query_node->left(), 
				       reference_node->right());
      
    // passing sampling information to the children
    assert(query_node->left()->stat().total_points()
	   == query_node->right()->stat().total_points());


    // if the parent has encountered extra points, pass
    // that information down to the children.
    size_t extra_points_encountered
      = query_node->stat().total_points()
      - query_node->left()->stat().total_points();
    assert(extra_points_encountered >= 0);

    if (extra_points_encountered > 0) {
      query_node->left()->stat().add_total_points(extra_points_encountered);
      query_node->right()->stat().add_total_points(extra_points_encountered);
      size_t extra_points_sampled
	= query_node->stat().samples()
	- std::min(query_node->left()->stat().samples(),
		   query_node->right()->stat().samples());
      assert(extra_points_sampled >= 0);
      query_node->left()->stat().add_samples(extra_points_sampled);
      query_node->right()->stat().add_samples(extra_points_sampled);
    }


    if (left_p_cos_pq > right_p_cos_pq) {
      ComputeApproxRecursion_(query_node->left(),
			      reference_node->left(), 
			      left_p_cos_pq);
      ComputeApproxRecursion_(query_node->left(),
			      reference_node->right(), 
			      right_p_cos_pq);
    } else {
      ComputeApproxRecursion_(query_node->left(),
			      reference_node->right(), 
			      right_p_cos_pq);
      ComputeApproxRecursion_(query_node->left(),
			      reference_node->left(), 
			      left_p_cos_pq);
    }

    left_p_cos_pq = MaxNodeIP_(query_node->right(),
			       reference_node->left());
    right_p_cos_pq = MaxNodeIP_(query_node->right(), 
				reference_node->right());
      
    if (left_p_cos_pq > right_p_cos_pq) {
      ComputeApproxRecursion_(query_node->right(),
			      reference_node->left(), 
			      left_p_cos_pq);
      ComputeApproxRecursion_(query_node->right(),
			      reference_node->right(), 
			      right_p_cos_pq);
    } else {
      ComputeApproxRecursion_(query_node->right(),
			      reference_node->right(), 
			      right_p_cos_pq);
      ComputeApproxRecursion_(query_node->right(),
			      reference_node->left(), 
			      left_p_cos_pq);
    }
      

    // save sampling information from the children
    assert(query_node->left()->stat().total_points()
	   == query_node->right()->stat().total_points());

    assert(query_node->left()->stat().total_points()
	   >  query_node->stat().total_points());
    query_node->stat().set_total_points(
	query_node->left()->stat().total_points());

    assert(query_node->stat().samples() <= 
	   std::min(query_node->left()->stat().samples(),
		    query_node->right()->stat().samples()));
    query_node->stat().set_samples(
        std::min(query_node->left()->stat().samples(),
	query_node->right()->stat().samples()));


    // Update the upper bound as above
    query_node->stat().set_bound(std::min(query_node->left()->stat().bound(),
					  query_node->right()->stat().bound()));
  }
} // ComputeApproxRecursion_
  

void ApproxMaxIP::InitApprox(const arma::mat& queries_in,
			     const arma::mat& references_in) {
    
    
  // track the number of prunes and computations
  number_of_prunes_ = 0;
  ball_has_origin_ = 0;
  distance_computations_ = 0;
  split_decisions_ = 0;
    
  // Get the leaf size from the module
  leaf_size_ = mlpack::CLI::GetParam<int>("approx_maxip/leaf_size");
  sample_limit_ = mlpack::CLI::GetParam<int>("approx_maxip/sample_limit");


  // Make sure the leaf size is valid
  assert(leaf_size_ > 0);
  assert(sample_limit_ > 0);
    
  // Copy the matrices to the class members since they will be rearranged.  
  queries_ = queries_in;
  references_ = references_in;
    
  // The data sets need to have the same number of points
  assert(queries_.n_rows == references_.n_rows);
    
  // K-nearest neighbors initialization
  knns_ = mlpack::CLI::GetParam<int>("approx_maxip/knns");

  // Initialize the list of nearest neighbor candidates
  max_ip_indices_ 
    = -1 * arma::ones<arma::Mat<size_t> >(knns_, queries_.n_cols);
    
  // Initialize the vector of upper bounds for each point.
  // We do not consider negative values for inner products.
  max_ips_ = 0.0 * arma::ones<arma::mat>(knns_, queries_.n_cols);

  // We'll time tree building
  mlpack::CLI::StartTimer("approx_maxip/tree_building");

  reference_tree_
    = proximity::MakeGenMetricTree<TreeType>(references_, 
					     leaf_size_,
					     &old_from_new_references_,
					     NULL);
    
  if (mlpack::CLI::HasParam("approx_maxip/dual_tree")) {
    query_tree_
      = proximity::MakeGenConeTree<CTreeType>(queries_,
					      leaf_size_,
					      &old_from_new_queries_,
					      NULL);
  }

  // Stop the timer we started above
  mlpack::CLI::StopTimer("approx_maxip/tree_building");

  // saving the query norms beforehand to use 
  // in the tree-based searches -- need to do it 
  // after the shuffle to correspond to correct indices
  query_norms_ = 0.0 * arma::ones<arma::vec>(queries_.n_cols);
  for (size_t i = 0; i < queries_.n_cols; i++)
    query_norms_(i) = arma::norm(queries_.col(i), 2);

  // pre-computing the sample sizes
  mlpack::CLI::StartTimer("approx_maxip/computing_sample_sizes");

  // Check if the probability is <=1
  double alpha = mlpack::CLI::GetParam<double>("approx_maxip/alpha");
  assert(alpha <= 1.0);

  epsilon_ = mlpack::CLI::GetParam<double>("approx_maxip/epsilon");
  rank_approx_ = (size_t) (epsilon_ * (double)references_.n_cols / 100.0);

  mlpack::Log::Info << "Rank Approximation: " << epsilon_ 
		    << "% or " << rank_approx_
		    << " with prob. " << alpha << std::endl;


  sample_sizes_.set_size(references_.n_cols);

  if (rank_approx_ > 1) {
    if (knns_ == 1)
      ComputeSampleSizes_(rank_approx_, alpha, &sample_sizes_);
    else
      ComputeSampleSizes_(rank_approx_, alpha, knns_, &sample_sizes_);
  } else {
    for (size_t i = 0; i < sample_sizes_.n_elem; i++)
      sample_sizes_[i] = i+1;
  }

  mlpack::Log::Info << "n: " << sample_sizes_[references_.n_cols -1]
		    << std::endl;

  mlpack::CLI::StopTimer("approx_maxip/computing_sample_sizes");

  min_samples_per_q_ = sample_sizes_[references_.n_cols -1];      

} // InitApprox


void ApproxMaxIP::WarmInitApprox(size_t knns, double epsilon) {
    
    
  // track the number of prunes and computations
  number_of_prunes_ = 0;
  ball_has_origin_ = 0;
  distance_computations_ = 0;
  split_decisions_ = 0;
    
  // K-nearest neighbors initialization
  knns_ = knns;

  // Initialize the list of nearest neighbor candidates
  max_ip_indices_.reset();
  max_ip_indices_ 
    = -1 * arma::ones<arma::Mat<size_t> >(knns_, queries_.n_cols);
    
  // Initialize the vector of upper bounds for each point.
  // We do not consider negative values for inner products (FOR NOW)
  max_ips_.reset();
  max_ips_ =  0.0 * arma::ones<arma::mat>(knns_, queries_.n_cols);

  // need to reset the querystats in the Query Tree
  if (mlpack::CLI::HasParam("approx_maxip/dual_tree"))
    if (query_tree_ != NULL)
      reset_tree_(query_tree_);


  // update the sample sizes
  // Check if the probability is <=1
  double alpha = mlpack::CLI::GetParam<double>("approx_maxip/alpha");
  assert(alpha <= 1.0);

  epsilon_ = epsilon;
  rank_approx_ = (size_t) (epsilon_ * (double)references_.n_cols / 100.0);

  mlpack::Log::Info << "Rank Approximation: " << epsilon_ 
		    << "% or " << rank_approx_
		    << " with prob. " << alpha << std::endl;


  // free(sample_sizes_);
  sample_sizes_.zeros();

  if (rank_approx_ > 1) {
    if (knns_ == 1)
      ComputeSampleSizes_(rank_approx_, alpha, &sample_sizes_);
    else
      ComputeSampleSizes_(rank_approx_, alpha, knns_, &sample_sizes_);
  } else {
    for (size_t i = 0; i < sample_sizes_.n_elem; i++)
      sample_sizes_[i] = i+1;
  }

  mlpack::Log::Info << "n: " << sample_sizes_[references_.n_cols -1]
		    << std::endl;

  min_samples_per_q_ = sample_sizes_[references_.n_cols -1];      

} // WarmInitApprox

void ApproxMaxIP::reset_tree_(CTreeType* tree) {
  assert(tree != NULL);
  tree->stat().set_bound(0.0);
  tree->stat().set_total_points(0);
  tree->stat().set_samples(0);

  if (!tree->is_leaf()) {
    reset_tree_(tree->left());
    reset_tree_(tree->right());
  }
}

double ApproxMaxIP::ComputeApprox(arma::Mat<size_t>* resulting_neighbors,
				  arma::mat* ips) {


  if (mlpack::CLI::HasParam("approx_maxip/dual_tree")) {
    // do dual-tree search
    mlpack::Log::Info << "DUAL-TREE Search: " << std::endl;


    CLI::StartTimer("approx_maxip/fast_dual");
    ComputeApproxRecursion_(query_tree_, reference_tree_,
			    MaxNodeIP_(query_tree_, reference_tree_));
    CLI::StopTimer("approx_maxip/fast_dual");
    
    resulting_neighbors->set_size(max_ip_indices_.n_rows,
				  max_ip_indices_.n_cols);
    ips->set_size(max_ips_.n_rows, max_ips_.n_cols);


    for (size_t i = 0; i < max_ip_indices_.n_cols; i++) {
      for (size_t k = 0; k < max_ip_indices_.n_rows; k++) {

	assert(max_ip_indices_(k, i) != (size_t) -1 
	       || max_ips_(k, i) == 0.0);

	if (max_ip_indices_(k, i) != (size_t) -1)
	  (*resulting_neighbors)(k, old_from_new_queries_[i]) =
	    old_from_new_references_[max_ip_indices_(k, i)];
	else 
	  (*resulting_neighbors)(k, old_from_new_queries_[i]) = -1;

        (*ips)(k, old_from_new_queries_[i]) = max_ips_(k, i);
      }
    }

  } else {
    // do single-tree search
    mlpack::Log::Info << "SINGLE-TREE Search: " << std::endl;

    CLI::StartTimer("approx_maxip/fast_single");
    for (query_ = 0; query_ < queries_.n_cols; ++query_) {
      query_samples_needed_ = min_samples_per_q_;

      if (CLI::HasParam("approx_maxip/no_tree")) {
	// ComputeApproxBaseCase_(reference_tree_);
	ComputeApproxBaseCase_(reference_tree_->left());
	ComputeApproxBaseCase_(reference_tree_->right());
      } else 
	ComputeApproxRecursion_(reference_tree_, 
				MaxNodeIP_(reference_tree_));

//       assert(!(query_samples_needed_ > 0));
    }
    CLI::StopTimer("approx_maxip/fast_single");


    resulting_neighbors->set_size(max_ip_indices_.n_rows,
				  max_ip_indices_.n_cols);
    ips->set_size(max_ips_.n_rows, max_ips_.n_cols);

    // We need to map the indices back from how they have been permuted
    for (size_t i = 0; i < max_ip_indices_.n_cols; i++) {
      for (size_t k = 0; k < max_ip_indices_.n_rows; k++) {

	assert(max_ip_indices_(k, i) != (size_t) -1 
	       || max_ips_(k, i) == 0.0);
	if (max_ip_indices_(k, i) != (size_t) -1)
	  (*resulting_neighbors)(k, i) =
	    old_from_new_references_[max_ip_indices_(k, i)];
	else 
	  (*resulting_neighbors)(k, i) = -1;

	(*ips)(k, i) = max_ips_(k, i);
      }
    }
  }

  mlpack::Log::Info << "Tree-based Search - Number of prunes: " 
		   << number_of_prunes_ << std::endl;
  mlpack::Log::Info << "\t \t Avg. # of DC: " 
		   << (double) distance_computations_ 
    / (double) queries_.n_cols << std::endl;
  mlpack::Log::Info << "\t \t Avg. # of SD: " 
		   << (double) split_decisions_ 
    / (double) queries_.n_cols << std::endl;

  return (double) (distance_computations_ + split_decisions_)
    / (double) queries_.n_cols;
} // ComputeApprox
