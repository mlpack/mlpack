/**
 * @file allkfn.h
 *
 * Defines AllkFN class to perform all-k-furthest-neighbors on two specified
 * data sets.
 */

#include "allkfn.h"

using namespace mlpack::allkfn;

// We call an advanced constructor of arma::mat which allows us to alias a
// matrix.
AllkFN::AllkFN(arma::mat& queries_in, arma::mat& references_in,
               struct datanode* module_in, int options) :
    module_(module_in),
    references_(references_in.memptr(), references_in.n_rows,
        references_in.n_cols, !(options & ALIAS_MATRIX)),
    queries_(queries_in.memptr(), queries_in.n_rows, queries_in.n_cols,
        !(options & ALIAS_MATRIX)),
    naive_(options & NAIVE),
    dual_mode_(!(options & MODE_SINGLE)),
    number_of_prunes_(0) {

  // C++0x will allow us to call out to other constructors so we can avoid this
  // copypasta problem.

  // Get the leaf size from the module; naive mode uses a cheap trick and makes
  // the entire tree be one node, which forces naive computation even when using
  // the dual-mode recursion.
  if(naive_)
    leaf_size_ = max(queries_.n_cols, references_.n_cols);
  else
    leaf_size_ = fx_param_int(module_, "leaf_size", 20);

  // Make sure the leaf size is valid
  DEBUG_ASSERT(leaf_size_ > 0);

  // The data sets need to have the same number of points
  DEBUG_SAME_SIZE(queries_.n_rows, references_.n_rows);

  // Get number of furthest neighbors
  kfns_ = fx_param_int(module_, "kfns", 5);

  // Initialize the list of nearest neighbor candidates
  neighbor_indices_.set_size(queries_.n_cols * kfns_);

  // Initialize the vector of upper bounds for each point.
  neighbor_distances_.zeros(queries_.n_cols * kfns_);

  // We'll time tree building
  fx_timer_start(module_, "tree_building");

  // This call makes each tree from a matrix, leaf size, and two arrays that
  // record the permutation of the data points.
  if (dual_mode_)
    query_tree_ = tree::MakeKdTreeMidpoint<TreeType>(queries_, leaf_size_,
        old_from_new_queries_);
  else
    query_tree_ = NULL;

  reference_tree_ = tree::MakeKdTreeMidpoint<TreeType>(references_, leaf_size_,
      old_from_new_references_);

  // Stop the timer we started above
  fx_timer_stop(module_, "tree_building");
} 

// We call an advanced constructor of arma::mat which allows us to alias a
// matrix.
AllkFN::AllkFN(arma::mat& references_in, struct datanode* module_in,
               int options) :
    module_(module_in),
    references_(references_in.memptr(), references_in.n_rows,
        references_in.n_cols, !(options & ALIAS_MATRIX)),
    queries_(references_.memptr(), references_.n_rows, references_.n_cols,
        false),
    naive_(options & NAIVE),
    dual_mode_(!(options & MODE_SINGLE)),
    number_of_prunes_(0) {
  
  // Get the leaf size from the module
  if(naive_)
    leaf_size_ = max(queries_.n_cols, references_.n_cols);
  else
    leaf_size_ = fx_param_int(module_, "leaf_size", 20);

  // Make sure the leaf size is valid
  DEBUG_ASSERT(leaf_size_ > 0);

  // The data sets need to have the same number of points
  DEBUG_SAME_SIZE(queries_.n_rows, references_.n_rows);

  // Get number of furthest neighbors
  kfns_ = fx_param_int(module_, "kfns", 5);

  // Initialize the list of nearest neighbor candidates
  neighbor_indices_.set_size(queries_.n_cols * kfns_);

  // Initialize the vector of upper bounds for each point.
  neighbor_distances_.zeros(queries_.n_cols * kfns_);

  // We'll time tree building
  fx_timer_start(module_, "tree_building");

  // This call makes each tree from a matrix, leaf size, and two arrays that
  // record the permutation of the data points.
  query_tree_ = NULL;
  reference_tree_ = tree::MakeKdTreeMidpoint<TreeType>(references_,
      leaf_size_, old_from_new_references_);

  // Stop the timer we started above
  fx_timer_stop(module_, "tree_building");
}


// We call an advanced constructor of arma::mat which allows us to alias a
// matrix.
AllkFN::AllkFN(arma::mat& queries_in, arma::mat& references_in,
               index_t leaf_size, index_t kfns, int options) :
    module_(NULL),
    references_(references_in.memptr(), references_in.n_rows,
        references_in.n_cols, !(options & ALIAS_MATRIX)),
    queries_(queries_in.memptr(), queries_in.n_rows, queries_in.n_cols,
        !(options & ALIAS_MATRIX)),
    naive_(options & NAIVE),
    dual_mode_(!(options & MODE_SINGLE)),
    leaf_size_(naive_ ? max(references_.n_cols, queries_.n_cols) : leaf_size),
    kfns_(kfns),
    number_of_prunes_(0) {
  
  // Make sure the leaf size is valid
  DEBUG_ASSERT(leaf_size_ > 0);

  // The data sets need to have the same number of points
  DEBUG_SAME_SIZE(queries_.n_rows, references_.n_rows);

  // Initialize the list of nearest neighbor candidates
  neighbor_indices_.set_size(queries_.n_cols * kfns_);

  // Initialize the vector of upper bounds for each point.
  neighbor_distances_.zeros(queries_.n_cols * kfns_);

  // We'll time tree building
  fx_timer_start(module_, "tree_building");

  // This call makes each tree from a matrix, leaf size, and two arrays that
  // record the permutation of the data points.
  if (dual_mode_)
    query_tree_ = tree::MakeKdTreeMidpoint<TreeType>(queries_, leaf_size_,
        old_from_new_queries_);
  else
    query_tree_ = NULL;

  reference_tree_ = tree::MakeKdTreeMidpoint<TreeType>(references_, leaf_size_,
      old_from_new_references_);

  // Stop the timer we started above
  fx_timer_stop(module_, "tree_building");
}

// We call an advanced constructor of arma::mat which allows us to alias a
// matrix.
AllkFN::AllkFN(arma::mat& references_in, index_t leaf_size,
               index_t kfns, int options) :
    module_(NULL),
    references_(references_in.memptr(), references_in.n_rows,
        references_in.n_cols, !(options & ALIAS_MATRIX)),
    queries_(references_.memptr(), references_.n_rows, references_.n_cols,
        false),
    naive_(options & NAIVE),
    dual_mode_(!(options & MODE_SINGLE)),
    leaf_size_(naive_ ? references_.n_cols : leaf_size),
    kfns_(kfns),
    number_of_prunes_(0) {
  
  // Make sure the leaf size is valid
  DEBUG_ASSERT(leaf_size_ > 0);

  // The data sets need to have the same number of points
  DEBUG_SAME_SIZE(queries_.n_rows, references_.n_rows);

  // Initialize the list of nearest neighbor candidates
  neighbor_indices_.set_size(queries_.n_cols * kfns_);

  // Initialize the vector of upper bounds for each point.
  neighbor_distances_.zeros(queries_.n_cols * kfns_);

  // We'll time tree building
  fx_timer_start(module_, "tree_building");

  // This call makes each tree from a matrix, leaf size, and two arrays that
  // record the permutation of the data points.
  query_tree_ = NULL;
  reference_tree_ = tree::MakeKdTreeMidpoint<TreeType>(references_,
      leaf_size_, old_from_new_references_);

  // Stop the timer we started above
  fx_timer_stop(module_, "tree_building");
}

/**
 * The tree is the only member we are responsible for deleting.  The others will
 * take care of themselves.  
 */
AllkFN::~AllkFN() {
  if (reference_tree_ != query_tree_)
    delete reference_tree_;
  if (query_tree_ != NULL)
    delete query_tree_;
} 

/**
 * Computes the maximum squared distance between the bounding boxes of two nodes
 */
double AllkFN::MaxNodeDistSq_(TreeType* query_node, TreeType* reference_node) {
  // node->bound() gives us the DHrectBound class for the node
  // It has a function MinDistanceSq which takes another DHrectBound
  return query_node->bound().MaxDistanceSq(reference_node->bound());
} 

/**
 * Performs exhaustive computation between two leaves.  
 */
void AllkFN::ComputeBaseCase_(TreeType* query_node, TreeType* reference_node) {
  // Check that the pointers are not NULL
  DEBUG_ASSERT(query_node != NULL);
  DEBUG_ASSERT(reference_node != NULL);

  // Check that we really should be in the base case
  DEBUG_WARN_IF(!query_node->is_leaf());
  DEBUG_WARN_IF(!reference_node->is_leaf());

  // Used to find the query node's new upper bound
  double query_min_neighbor_distance = DBL_MAX;
  std::vector<std::pair<double, index_t> > neighbors(kfns_);

  // node->begin() is the index of the first point in the node, 
  // node->end is one past the last index
  for (index_t query_index = query_node->begin(); 
       query_index < query_node->end(); query_index++) {

    // Get the query point from the matrix
    arma::vec query_point = queries_.unsafe_col(query_index);

    index_t ind = query_index * kfns_;
    for(index_t i = 0; i < kfns_; i++) {
      neighbors[i] = std::make_pair(neighbor_distances_[ind + i],
          neighbor_indices_[ind + i]);
    }

    // We'll do the same for the references
    for (index_t reference_index = reference_node->begin(); 
        reference_index < reference_node->end(); reference_index++) {

      // Confirm that points do not identify themselves as neighbors
      // in the monochromatic case
      if (reference_node != query_node || reference_index != query_index) {
        arma::vec reference_point = references_.unsafe_col(reference_index);

        double distance = la::DistanceSqEuclidean(query_point, reference_point);

        // If the reference point is closer than the current candidate, 
        // we'll update the candidate
        if (distance > neighbor_distances_[ind + kfns_ - 1]) {
          neighbors.push_back(std::make_pair(distance, reference_index));
        }
      }
    } // for reference_index

    std::sort(neighbors.begin(), neighbors.end(),
        std::greater<std::pair<double, index_t> >());

    for(index_t i = 0; i < kfns_; i++) {
      neighbor_distances_[ind + i] = neighbors[i].first;
      neighbor_indices_[ind + i]  = neighbors[i].second;
    }
    neighbors.resize(kfns_);

    // We need to find the lower bound distance for this query node
    if (neighbor_distances_[ind + kfns_ - 1] < query_min_neighbor_distance)
      query_min_neighbor_distance = neighbor_distances_[ind + kfns_ - 1]; 
  }
  // for query_index 
  // Update the lower bound for the query_node
  query_node->stat().set_min_distance_so_far(query_min_neighbor_distance);
} // ComputeBaseCase_

/**
 * The recursive function for dual tree computation
 */
void AllkFN::ComputeDualNeighborsRecursion_(TreeType* query_node, 
                                            TreeType* reference_node, 
                                            double higher_bound_distance) {

  if (higher_bound_distance < query_node->stat().min_distance_so_far()) {
    number_of_prunes_++; // Pruned by distance; the nodes cannot be any further
    return;              // than the already established upper bound.
  }

  if (query_node->is_leaf() && reference_node->is_leaf()) {
    ComputeBaseCase_(query_node, reference_node); // Base case: both are leaves.
    return;
  }

  if (query_node->is_leaf()) {
    // We must keep descending down the reference node to get a leaf.

    // We'll order the computation by distance; descend in the direction of more
    // distance first.
    double left_distance = MaxNodeDistSq_(query_node, reference_node->left());
    double right_distance = MaxNodeDistSq_(query_node, reference_node->right());

    if (left_distance > right_distance) {
      ComputeDualNeighborsRecursion_(query_node, reference_node->left(), 
          left_distance);
      ComputeDualNeighborsRecursion_(query_node, reference_node->right(), 
          right_distance);
    } else {
      ComputeDualNeighborsRecursion_(query_node, reference_node->right(), 
          right_distance);
      ComputeDualNeighborsRecursion_(query_node, reference_node->left(), 
          left_distance);
    }
    return;
  }

  else if (reference_node->is_leaf()) {
    // We must descend down the query node to get to a leaf.
    double left_distance = MaxNodeDistSq_(query_node->left(), reference_node);
    double right_distance = MaxNodeDistSq_(query_node->right(), reference_node);

    ComputeDualNeighborsRecursion_(query_node->left(), reference_node,
        left_distance);
    ComputeDualNeighborsRecursion_(query_node->right(), reference_node,
        right_distance);

    // We need to update the upper bound based on the new upper bounds of 
    // the children
    query_node->stat().set_min_distance_so_far(
        min(query_node->left()->stat().min_distance_so_far(),
            query_node->right()->stat().min_distance_so_far()));
    return;
  }

  // Neither side is a leaf; so we recurse on all combinations of both.  The
  // calculations are ordered by distance.
  double left_distance = MaxNodeDistSq_(query_node->left(), 
      reference_node->left());
  double right_distance = MaxNodeDistSq_(query_node->left(), 
      reference_node->right());

  // Recurse on query_node->left() first.
  if (left_distance > right_distance) {
    ComputeDualNeighborsRecursion_(query_node->left(), reference_node->left(), 
        left_distance);
    ComputeDualNeighborsRecursion_(query_node->left(), reference_node->right(), 
        right_distance);
  } else {
    ComputeDualNeighborsRecursion_(query_node->left(), reference_node->right(), 
        right_distance);
    ComputeDualNeighborsRecursion_(query_node->left(), reference_node->left(), 
        left_distance);
  }


  left_distance = MaxNodeDistSq_(query_node->right(), reference_node->left());
  right_distance = MaxNodeDistSq_(query_node->right(), reference_node->right());

  // Now recurse on query_node->right().
  if (left_distance > right_distance) {
    ComputeDualNeighborsRecursion_(query_node->right(), reference_node->left(), 
        left_distance);
    ComputeDualNeighborsRecursion_(query_node->right(), reference_node->right(), 
        right_distance);
  } else {
    ComputeDualNeighborsRecursion_(query_node->right(), reference_node->right(), 
        right_distance);
    ComputeDualNeighborsRecursion_(query_node->right(), reference_node->left(), 
        left_distance);
  }

  // Update the upper bound as above
  query_node->stat().set_min_distance_so_far(
      min(query_node->left()->stat().min_distance_so_far(),
          query_node->right()->stat().min_distance_so_far()));
} // ComputeDualNeighborsRecursion_

void AllkFN::ComputeSingleNeighborsRecursion_(index_t point_id,
                                              arma::vec& point,
                                              TreeType* reference_node,
                                              double* max_dist_so_far) {
  if (reference_node->is_leaf()) {
    // Base case: reference node is a leaf.
    std::vector<std::pair<double, index_t> > neighbors(kfns_);
    index_t ind = point_id * kfns_;
    for (index_t i = 0; i < kfns_; i++) {
      neighbors[i] = std::make_pair(neighbor_distances_[ind + i],
          neighbor_indices_[ind + i]);
    }

    // We'll do the same for the references
    for (index_t reference_index = reference_node->begin();
        reference_index < reference_node->end(); reference_index++) {
      // Confirm that points do not identify themselves as neighbors in the
      // monochromatic case
      if (!(references_.memptr() == queries_.memptr() &&
            reference_index == point_id)) {
        arma::vec reference_point = references_.unsafe_col(reference_index);

        double distance = la::DistanceSqEuclidean(point, reference_point);

        // If the reference point is further than the current candidate, we'll
        // update the candidate
        if (distance > neighbor_distances_[ind + kfns_ - 1]) {
          neighbors.push_back(std::make_pair(distance, reference_index));
        }
      }
    } // for reference_index

    std::sort(neighbors.begin(), neighbors.end(),
        std::greater<std::pair<double, index_t> >());
    for (index_t i = 0; i < kfns_; i++) {
      neighbor_distances_[ind + i] = neighbors[i].first;
      neighbor_indices_[ind + i] = neighbors[i].second;
    }
    *max_dist_so_far = neighbor_distances_[ind + kfns_ - 1];
  } else {
    // We'll order the computation by distance
    double left_distance = reference_node->left()->bound().MaxDistanceSq(point);
    double right_distance =
        reference_node->right()->bound().MaxDistanceSq(point);

    if (left_distance > right_distance) {
      if (*max_dist_so_far > left_distance)
        number_of_prunes_++;
      else
        ComputeSingleNeighborsRecursion_(point_id, point,
            reference_node->left(), max_dist_so_far);

      if (*max_dist_so_far > right_distance)
        number_of_prunes_++;
      else
        ComputeSingleNeighborsRecursion_(point_id, point,
            reference_node->right(), max_dist_so_far);
      
    } else {
      if (*max_dist_so_far > right_distance)
        number_of_prunes_++;
      else
        ComputeSingleNeighborsRecursion_(point_id, point,
            reference_node->right(), max_dist_so_far);

      if (*max_dist_so_far > left_distance)
        number_of_prunes_++;
      else
        ComputeSingleNeighborsRecursion_(point_id, point,
            reference_node->left(), max_dist_so_far);
    }
  }
}

/**
 * Computes the nearest neighbors and stores them in *results
 */
void AllkFN::ComputeNeighbors(arma::Col<index_t>& resulting_neighbors,
                              arma::vec& distances) {
  fx_timer_start(module_, "computing_neighbors");
  if (naive_) {
    // Run the base computation on all nodes
    if (query_tree_)
      ComputeBaseCase_(query_tree_, reference_tree_);
    else
      ComputeBaseCase_(reference_tree_, reference_tree_);
  } else {
    if (dual_mode_) {
      // Start on the root of each tree
      if (query_tree_)
        ComputeDualNeighborsRecursion_(query_tree_, reference_tree_,
            MaxNodeDistSq_(query_tree_, reference_tree_));
      else
        ComputeDualNeighborsRecursion_(reference_tree_, reference_tree_,
            MaxNodeDistSq_(reference_tree_, reference_tree_));
    } else { // Single tree mode
      index_t chunk = queries_.n_cols / 10;

      for (index_t i = 0; i < 10; i++) {
        for (index_t j = 0; j < chunk; j++) {
          arma::vec point = queries_.unsafe_col(i * chunk + j);
          double max_dist_so_far = 0;
          ComputeSingleNeighborsRecursion_(i * chunk + j, point,
              reference_tree_, &max_dist_so_far);
        }
      }
      
      for (index_t i = 0; i < queries_.n_cols % 10; i++) {
        index_t ind = (queries_.n_cols / 10) * 10 + i;
        arma::vec point = queries_.unsafe_col(ind);
        double max_dist_so_far = 0;
        ComputeSingleNeighborsRecursion_(i, point, reference_tree_,
            &max_dist_so_far);
      }
    }
  }

  fx_timer_stop(module_, "computing_neighbors");

  // We need to initialize the results list before filling it
  resulting_neighbors.set_size(neighbor_indices_.n_elem);
  distances.set_size(neighbor_distances_.n_elem);

  // We need to map the indices back from how they have been permuted
  if (query_tree_ != NULL) {
    for (index_t i = 0; i < neighbor_indices_.n_elem; i++) {
      resulting_neighbors[
        old_from_new_queries_[(i / kfns_)] * kfns_ + i % kfns_] = 
        old_from_new_references_[neighbor_indices_[i]];
      distances[
        old_from_new_queries_[(i / kfns_)] * kfns_ + i % kfns_] = 
        neighbor_distances_[i];
    }
  } else {
    for (index_t i = 0; i < neighbor_indices_.n_elem; i++) {
      resulting_neighbors[
        old_from_new_references_[(i / kfns_)] * kfns_ + i % kfns_] = 
        old_from_new_references_[neighbor_indices_[i]];
      distances[
        old_from_new_references_[(i / kfns_)] * kfns_+ i % kfns_] = 
        neighbor_distances_[i];
    }
  }
} // ComputeNeighbors
