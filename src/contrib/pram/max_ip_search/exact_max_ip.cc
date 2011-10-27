/**
 * @file exact_max_ip.cc
 * @author Parikshit Ram
 *
 * This file implements the functions declared 
 * for the class MaxIP.
 */

#include "exact_max_ip.h"

double MaxIP::MaxNodeIP_(TreeType* reference_node) {

  // counting the split decisions 
  split_decisions_++;

  // compute maximum possible inner product 
  // between a point and a ball in terms of 
  // the ball's center and radius
  arma::vec q = queries_.col(query_);
  arma::vec centroid = reference_node->bound().center();

  // +1: Can be cached in the reference tree
  double c_norm = reference_node->stat().dist_to_origin();

  assert(arma::norm(q, 2) == query_norms_(query_));

  double rad = std::sqrt(reference_node->bound().radius());

  if (mlpack::CLI::HasParam("maxip/angle_prune")) { 
    // tighter bound of \max_{r \in B_p^R} <q,r> 
    //    = |q| \max_{r \in B_p^R} |r| cos <qr 
    //    \leq |q| \max_{r \in B_p^R} |r| \max_{r \in B_p^R} cos <qr 
    //    \leq |q| (|p|+R) if <qp \leq \max_r <pr
    //    \leq |q| (|p|+R) cos( <qp - \max_r <pr ) otherwise

    double max_cos_qr = 1.0;
    if (rad <= c_norm) {
      // +1
      double cos_qp = arma::dot(q, centroid) 
	/ (query_norms_(query_) * c_norm);
      double sin_qp = std::sqrt(1 - cos_qp * cos_qp);

      double max_sin_pr = reference_node->stat().sine_origin();
      double min_cos_pr = reference_node->stat().cosine_origin();;

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

    return (query_norms_(query_) * (c_norm + rad) * max_cos_qr);
  } // angle-prune

  if (mlpack::CLI::HasParam("maxip/alt_angle_prune")) { 
    // tighter bound of \max_{r \in B_p^R} <q,r> 
    //    = |q| \max_{r \in B_p^R} |r| cos <qr 
    //    \leq  |q| (|p| cos <qp + R )  (closed-form solution 
    // the maximization above (I think it is correct))
    //    = ( <q, p> + |p| R )

    // +1
    return (arma::dot(q, centroid) + (query_norms_(query_) * rad));
  } // alt-angle-prune
  
  // Otherwise :
  // simple bound of \max_{r \in B_p^R} <q,r> 
  //    = |q| \max_{r \in B_p^R} |r| cos <qr 
  //    \leq |q| \max_{r \in B_p^R} |r| \leq |q| (|p|+R)
  return (query_norms_(query_) * (c_norm + rad));
}

double MaxIP::MaxNodeIP_(CTreeType* query_node,
			 TreeType* reference_node) {

  // counting the split decisions 
  split_decisions_++;

  // min_{q', q} cos <qq' = cos_w
  arma::vec q = query_node->bound().center();
  double cos_w = query_node->bound().radius();
  double sin_w = query_node->bound().radius_conjugate();

  // +1: Can cache it in the query tree
  //  double q_norm = arma::norm(q, 2);
  double q_norm = query_node->stat().center_norm();

  arma::vec centroid = reference_node->bound().center();

  // +1: can be cached in the reference tree
//   double c_norm = arma::norm(centroid, 2);
  double c_norm = reference_node->stat().dist_to_origin();
  double rad = std::sqrt(reference_node->bound().radius());

  double max_cos_qp = 1.0;

  if (mlpack::CLI::HasParam("maxip/angle_prune")) { 

    if (rad <= c_norm) { 
      // cos <pq = cos_phi

      // +1
      double cos_phi = arma::dot(q, centroid) / (c_norm * q_norm);
      double sin_phi = std::sqrt(1 - cos_phi * cos_phi);

      // max_r sin <pr = sin_theta
//       double sin_theta = rad / c_norm;
//       double cos_theta = std::sqrt(1 - sin_theta * sin_theta);
      double sin_theta = reference_node->stat().sine_origin();
      double cos_theta = reference_node->stat().cosine_origin();

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
    return ((c_norm + rad) * max_cos_qp);
  } // angle-prune

  if (mlpack::CLI::HasParam("maxip/alt_angle_prune")) { 
    // using the closed-form-maximization,
    // |p| cos (phi - w) + R
      // +1
    double c_norm_cos_phi = arma::dot(q, centroid) / q_norm;
    double c_norm_sin_phi = std::sqrt(c_norm * c_norm 
				      - c_norm_cos_phi * c_norm_cos_phi);

    return (c_norm_cos_phi * cos_w + c_norm_sin_phi * sin_w + rad);
  } // alt-angle-prune

  return (c_norm + rad);

}


void MaxIP::ComputeBaseCase_(TreeType* reference_node) {
   
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

size_t MaxIP::SortValue(double value) {

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

void MaxIP::InsertNeighbor(size_t pos, size_t point_ind, double value) {

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


void MaxIP::ComputeNeighborsRecursion_(TreeType* reference_node, 
				       double upper_bound_ip) {

  assert(reference_node != NULL);
  //assert(upper_bound_ip == MaxNodeIP_(reference_node));

  if (upper_bound_ip < max_ips_(knns_ -1, query_)) { 
    // Pruned by distance
    number_of_prunes_++;
  } else if (reference_node->is_leaf()) {
    // base case for the single tree case
    ComputeBaseCase_(reference_node);

  } else {
    // Recurse on both as above
    double left_ip = MaxNodeIP_(reference_node->left());
    double right_ip = MaxNodeIP_(reference_node->right());

    if (left_ip > right_ip) {
      ComputeNeighborsRecursion_(reference_node->left(), 
				 left_ip);
      ComputeNeighborsRecursion_(reference_node->right(),
				 right_ip);
    } else {
      ComputeNeighborsRecursion_(reference_node->right(),
				 right_ip);
      ComputeNeighborsRecursion_(reference_node->left(), 
				 left_ip);
    }
  }      
} // ComputeNeighborsRecursion_


void MaxIP::ComputeBaseCase_(CTreeType* query_node, 
			     TreeType* reference_node) {

  // Check that the pointers are not NULL
  assert(reference_node != NULL);
  assert(reference_node->is_leaf());
  assert(query_node != NULL);

  // query node may not be a leaf
  //  assert(query_node->is_leaf());

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
  

void MaxIP::CheckPrune(CTreeType* query_node, TreeType* ref_node) {

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

void MaxIP::ComputeNeighborsRecursion_(CTreeType* query_node,
				       TreeType* reference_node, 
				       double upper_bound_p_cos_pq) {

  assert(query_node != NULL);
  assert(reference_node != NULL);
  //assert(upper_bound_p_cos_pq == MaxNodeIP_(query_node, reference_node));

  if (upper_bound_p_cos_pq < query_node->stat().bound()) { 
    // Pruned
    number_of_prunes_++;

    if (CLI::HasParam("maxip/check_prune"))
      CheckPrune(query_node, reference_node);
  }
  // node->is_leaf() works as one would expect
  else if (query_node->is_leaf() && reference_node->is_leaf()) {
    // Base Case
    ComputeBaseCase_(query_node, reference_node);
  } else if (query_node->is_leaf()) {
    // Only query is a leaf
      
    // We'll order the computation by distance 
    double left_p_cos_pq = MaxNodeIP_(query_node,
				      reference_node->left());
    double right_p_cos_pq = MaxNodeIP_(query_node,
				       reference_node->right());
      
    if (left_p_cos_pq > right_p_cos_pq) {
      ComputeNeighborsRecursion_(query_node, reference_node->left(), 
				 left_p_cos_pq);
      ComputeNeighborsRecursion_(query_node, reference_node->right(), 
				 right_p_cos_pq);
    } else {
      ComputeNeighborsRecursion_(query_node, reference_node->right(), 
				 right_p_cos_pq);
      ComputeNeighborsRecursion_(query_node, reference_node->left(), 
				 left_p_cos_pq);
    }
  } else if (reference_node->is_leaf()) {
    // Only reference is a leaf 
    double left_p_cos_pq
      = MaxNodeIP_(query_node->left(), reference_node);
    double right_p_cos_pq
      = MaxNodeIP_(query_node->right(), reference_node);
      
    ComputeNeighborsRecursion_(query_node->left(), reference_node, 
			       left_p_cos_pq);
    ComputeNeighborsRecursion_(query_node->right(), reference_node, 
			       right_p_cos_pq);
      
    // We need to update the upper bound based on the new upper bounds of 
    // the children
    query_node->stat().set_bound(std::min(query_node->left()->stat().bound(),
					  query_node->right()->stat().bound()));
  } else {

    if (CLI::HasParam("maxip/alt_dual_traversal")) {
      // try something new 
      // if query_node->radius() > reference_node->stat().cosine_origin()
      // traverse down the reference tree
      // else
      // traverse down the query tree

      if (query_node->bound().radius() 
	  > reference_node->stat().cosine_origin()) {

	// go down the reference tree
	double left_p_cos_pq = MaxNodeIP_(query_node, 
					  reference_node->left());
	double right_p_cos_pq = MaxNodeIP_(query_node, 
					   reference_node->right());
      
	if (left_p_cos_pq > right_p_cos_pq) {
	  ComputeNeighborsRecursion_(query_node,
				     reference_node->left(), 
				     left_p_cos_pq);
	  ComputeNeighborsRecursion_(query_node,
				     reference_node->right(), 
				     right_p_cos_pq);
	} else {
	  ComputeNeighborsRecursion_(query_node,
				     reference_node->right(), 
				     right_p_cos_pq);
	  ComputeNeighborsRecursion_(query_node,
				     reference_node->left(), 
				     left_p_cos_pq);
	}
      } else {
      
	// go down the query tree
	double left_p_cos_pq = MaxNodeIP_(query_node->left(),
					  reference_node);
	double right_p_cos_pq = MaxNodeIP_(query_node->right(), 
					   reference_node);
      
	ComputeNeighborsRecursion_(query_node->left(),
				   reference_node, 
				   left_p_cos_pq);
	ComputeNeighborsRecursion_(query_node->right(),
				   reference_node, 
				   right_p_cos_pq);

	// Update the upper bound as above
	query_node->stat().set_bound(std::min(query_node->left()->stat().bound(),
					      query_node->right()->stat().bound()));
      }
    }  else {
      // Recurse on both as above
      double left_p_cos_pq = MaxNodeIP_(query_node->left(), 
					reference_node->left());
      double right_p_cos_pq = MaxNodeIP_(query_node->left(), 
					 reference_node->right());
      
      if (left_p_cos_pq > right_p_cos_pq) {
	ComputeNeighborsRecursion_(query_node->left(),
				   reference_node->left(), 
				   left_p_cos_pq);
	ComputeNeighborsRecursion_(query_node->left(),
				   reference_node->right(), 
				   right_p_cos_pq);
      } else {
	ComputeNeighborsRecursion_(query_node->left(),
				   reference_node->right(), 
				   right_p_cos_pq);
	ComputeNeighborsRecursion_(query_node->left(),
				   reference_node->left(), 
				   left_p_cos_pq);
      }

      left_p_cos_pq = MaxNodeIP_(query_node->right(),
				 reference_node->left());
      right_p_cos_pq = MaxNodeIP_(query_node->right(), 
				  reference_node->right());
      
      if (left_p_cos_pq > right_p_cos_pq) {
	ComputeNeighborsRecursion_(query_node->right(),
				   reference_node->left(), 
				   left_p_cos_pq);
	ComputeNeighborsRecursion_(query_node->right(),
				   reference_node->right(), 
				   right_p_cos_pq);
      } else {
	ComputeNeighborsRecursion_(query_node->right(),
				   reference_node->right(), 
				   right_p_cos_pq);
	ComputeNeighborsRecursion_(query_node->right(),
				   reference_node->left(), 
				   left_p_cos_pq);
      }
      
      // Update the upper bound as above
      query_node->stat().set_bound(std::min(query_node->left()->stat().bound(),
					    query_node->right()->stat().bound()));
    } // alt-traversal
  } // All cases of dual-tree traversal
} // ComputeNeighborsRecursion_
  

void MaxIP::Init(const arma::mat& queries_in,
		 const arma::mat& references_in) {
    
    
  // track the number of prunes and computations
  number_of_prunes_ = 0;
  ball_has_origin_ = 0;
  distance_computations_ = 0;
  split_decisions_ = 0;
    
  // Get the leaf size from the module
  leaf_size_ = mlpack::CLI::GetParam<int>("maxip/leaf_size");
  // Make sure the leaf size is valid
  assert(leaf_size_ > 0);
    
  // Copy the matrices to the class members since they will be rearranged.  
  queries_ = queries_in;
  references_ = references_in;
    
  // The data sets need to have the same number of points
  assert(queries_.n_rows == references_.n_rows);
    
  // K-nearest neighbors initialization
  knns_ = mlpack::CLI::GetParam<int>("maxip/knns");

  // Initialize the list of nearest neighbor candidates
  max_ip_indices_ 
    = -1 * arma::ones<arma::Mat<size_t> >(knns_, queries_.n_cols);
    
  // Initialize the vector of upper bounds for each point.
  // We do not consider negative values for inner products.
  max_ips_ = 0.0 * arma::ones<arma::mat>(knns_, queries_.n_cols);

  // We'll time tree building
  mlpack::CLI::StartTimer("maxip/tree_building");

  reference_tree_
    = proximity::MakeGenMetricTree<TreeType>(references_, 
					     leaf_size_,
					     &old_from_new_references_,
					     NULL);
  set_angles_in_balls_(reference_tree_);
   
  if (mlpack::CLI::HasParam("maxip/dual_tree")) {
    query_tree_
      = proximity::MakeGenCosineTree<CTreeType>(queries_,
						leaf_size_,
						&old_from_new_queries_,
						NULL);
    set_norms_in_cones_(query_tree_);
  }

  // saving the query norms beforehand to use 
  // in the tree-based searches -- need to do it 
  // after the shuffle to correspond to correct indices
  query_norms_ = 0.0 * arma::ones<arma::vec>(queries_.n_cols);
  for (size_t i = 0; i < queries_.n_cols; i++)
    query_norms_(i) = arma::norm(queries_.col(i), 2);
      
  // Stop the timer we started above
  mlpack::CLI::StopTimer("maxip/tree_building");

} // Init


void MaxIP::InitNaive(const arma::mat& queries_in, 
		      const arma::mat& references_in) {
    
  queries_ = queries_in;
  references_ = references_in;
    
  // track the number of prunes and computations
  number_of_prunes_ = 0;
  ball_has_origin_ = 0;
  distance_computations_ = 0;
  split_decisions_ = 0;
    
  // The data sets need to have the same number of dimensions
  assert(queries_.n_rows == references_.n_rows);
    
  // K-nearest neighbors initialization
  knns_ = mlpack::CLI::GetParam<int>("maxip/knns");
  
  // Initialize the list of nearest neighbor candidates
  max_ip_indices_
    = -1 * arma::ones<arma::Mat<size_t> >(knns_, queries_.n_cols);
    
  // Initialize the vector of upper bounds for each point.
  // We do not consider negative values for inner products (FOR NOW).
  max_ips_ = 0.0 * arma::ones<arma::mat>(knns_, queries_.n_cols);

  // The only difference is that we set leaf_size_ to be large enough 
  // that each tree has only one node
  leaf_size_ = std::max(queries_.n_cols, references_.n_cols) + 1;

  // We'll time tree building
  mlpack::CLI::StartTimer("maxip/tree_building");
    
  reference_tree_
    = proximity::MakeGenMetricTree<TreeType>(references_, 
					     leaf_size_,
					     &old_from_new_references_,
					     NULL);

  // Stop the timer we started above
  mlpack::CLI::StopTimer("maxip/tree_building");
    
} // InitNaive
  
void MaxIP::WarmInit(size_t knns) {
    
    
  // track the number of prunes and computations
  number_of_prunes_ = 0;
  ball_has_origin_ = 0;
  distance_computations_ = 0;
  split_decisions_ = 0;
    
  // K-nearest neighbors initialization
  knns_ = knns;

  // Initialize the list of nearest neighbor candidates
  max_ip_indices_ 
    = -1 * arma::ones<arma::Mat<size_t> >(knns_, queries_.n_cols);
    
  // Initialize the vector of upper bounds for each point.
  // We do not consider negative values for inner products (FOR NOW)
  max_ips_ =  0.0 * arma::ones<arma::mat>(knns_, queries_.n_cols);

  // need to reset the querystats in the Query Tree
  if (mlpack::CLI::HasParam("maxip/dual_tree"))
    if (query_tree_ != NULL)
      reset_tree_(query_tree_);

} // WarmInit

void MaxIP::reset_tree_(CTreeType* tree) {
  assert(tree != NULL);
  tree->stat().set_bound(0.0);

  if (!tree->is_leaf()) {
    reset_tree_(tree->left());
    reset_tree_(tree->right());
  }

  return;
} // reset_tree_

void MaxIP::set_angles_in_balls_(TreeType* tree) {

  assert(tree != NULL);

  // set up node stats
  double c_norm = arma::norm(tree->bound().center(), 2);
  double rad = std::sqrt(tree->bound().radius());
  
  tree->stat().set_dist_to_origin(c_norm);
  if (rad <= c_norm)
    tree->stat().set_angles(rad / c_norm, (size_t) 1);
  else
    tree->stat().set_angles(-1.0, (size_t) 0);

  // traverse down the children
  if (!tree->is_leaf()) {
    set_angles_in_balls_(tree->left());
    set_angles_in_balls_(tree->right());
  }

  return;
} // set_angles_in_balls_


void MaxIP::set_norms_in_cones_(CTreeType* tree) {

  assert(tree != NULL);

  // set up node stats
  tree->stat().set_center_norm(arma::norm(tree->bound().center(), 2));

  // traverse down the children
  if (!tree->is_leaf()) {
    set_norms_in_cones_(tree->left());
    set_norms_in_cones_(tree->right());
  }

  return;
} // set_norms_in_cones_


double MaxIP::ComputeNeighbors(arma::Mat<size_t>* resulting_neighbors,
			       arma::mat* ips) {



  if (mlpack::CLI::HasParam("maxip/dual_tree")) {
    // do dual-tree search
    mlpack::Log::Info << "DUAL-TREE Search: " << std::endl;


    CLI::StartTimer("maxip/fast_dual");
    ComputeNeighborsRecursion_(query_tree_, reference_tree_,
			       MaxNodeIP_(query_tree_, reference_tree_));
    CLI::StopTimer("maxip/fast_dual");

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

    CLI::StartTimer("maxip/fast_single");
    for (query_ = 0; query_ < queries_.n_cols; ++query_) {
      ComputeNeighborsRecursion_(reference_tree_, 
				 MaxNodeIP_(reference_tree_));
    }
    CLI::StopTimer("maxip/fast_single");


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
		    << number_of_prunes_ << ", Ball has origin: "
		    << ball_has_origin_ << std::endl;
  mlpack::Log::Info << "\t \t Avg. # of DC: " 
		   << (double) distance_computations_ 
    / (double) queries_.n_cols << std::endl;
  mlpack::Log::Info << "\t \t Avg. # of SD: " 
		   << (double) split_decisions_ 
    / (double) queries_.n_cols << std::endl;

  return (double) (distance_computations_ + split_decisions_)
    / (double) queries_.n_cols;
} // ComputeNeighbors
  
double MaxIP::ComputeNaive(arma::Mat<size_t>* resulting_neighbors,
			   arma::mat* ips) {

  CLI::StartTimer("maxip/naive_search");
  for (query_ = 0; query_ < queries_.n_cols; ++query_) {
    ComputeBaseCase_(reference_tree_);
  }
  CLI::StopTimer("maxip/naive_search");

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

  mlpack::Log::Info << "Brute-force Search - Number of prunes: " 
		   << number_of_prunes_ << std::endl;
  mlpack::Log::Info << "\t \t Avg. # of DC: " 
		   << (double) distance_computations_ 
    / (double) queries_.n_cols << std::endl;
  mlpack::Log::Info << "\t \t Avg. # of SD: " 
		   << (double) split_decisions_ 
    / (double) queries_.n_cols << std::endl;

  return (double) (distance_computations_ + split_decisions_)
    / (double) queries_.n_cols;
}
