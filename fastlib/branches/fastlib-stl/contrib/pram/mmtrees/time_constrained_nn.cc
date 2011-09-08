#include "time_constrained_nn.h"

void TCNN::ComputeBaseCase_(TreeType* query_node,
			    TreeType* reference_node) {
   
  // Check that the pointers are not NULL
  DEBUG_ASSERT(query_node != NULL);
  DEBUG_ASSERT(reference_node != NULL);
  // Check that we really should be in the base case
  DEBUG_WARN_IF(!query_node->is_leaf());
  DEBUG_WARN_IF(!reference_node->is_leaf());

  // just checking for the single tree version
  DEBUG_ASSERT(query_node->end()
	       - query_node->begin() == 1);
    
  // Used to find the query node's new upper bound
  double query_max_neighbor_distance = -1.0;
  std::vector<std::pair<double, size_t> > neighbors(knns_);

  // Get the query point from the matrix
  Vector query_point;

  // FIX HERE: queries_ -> test_queries_
  queries_.MakeColumnVector(query_, &query_point);
      
  // FIX HERE: query_ -> test_query_
  size_t ind = query_*knns_;
  for(size_t i=0; i<knns_; i++) {
    neighbors[i]=std::make_pair(neighbor_distances_[ind+i],
				neighbor_indices_[ind+i]);
  }

  // We'll do the same for the references
  for (size_t reference_index = reference_node->begin(); 
       reference_index < reference_node->end(); reference_index++) {

    Vector reference_point;
    references_.MakeColumnVector(reference_index, &reference_point);
    // We'll use lapack to find the distance between the two vectors
    double distance =
      la::DistanceSqEuclidean(query_point, reference_point);
    // If the reference point is closer than the current candidate, 
    // we'll update the candidate

    // making sure that you are not choosing the same point
    if (distance > 0.0) {
      if (distance < neighbor_distances_[ind+knns_-1]) {
	neighbors.push_back(std::make_pair(distance, reference_index));
      }
    }
    //      }
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

  // Update the upper bound for the query_node
  query_node->stat().set_max_distance_so_far(query_max_neighbor_distance);

  // updating the number of distance computations done
  nn_dc_[query_] += reference_node->end() - reference_node->begin();

  // update the rank error obtained on this leaf
  number_of_leaves_++;
  size_t ref_ind
    = old_from_new_references_[neighbor_indices_[query_]];
  size_t rank_error;

  if (!rank_file_too_big_)
    rank_error = (size_t) rank_matrix_.get(ref_ind, query_) - 1;
  else
    rank_error = rank_vec_[ref_ind] - 1;

  (*error_list_)[query_]->push_back(rank_error);
  (*nn_dc_list_)[query_]->push_back(nn_dc_[query_]);

  return;
} // ComputeBaseCase_


void TCNN::ComputeNeighborsRecursion_(TreeType* query_node,
				      TreeType* reference_node, 
				      double lower_bound_distance) {

  DEBUG_ASSERT(query_node != NULL);
  DEBUG_ASSERT(reference_node != NULL);

  DEBUG_ASSERT(lower_bound_distance
	       == MinNodeDistSq_(query_node, reference_node));

  // just checking for the single tree version
  DEBUG_ASSERT(query_node->end()
	       - query_node->begin() == 1);
  DEBUG_ASSERT(query_node->is_leaf());

  if (lower_bound_distance > query_node->stat().max_distance_so_far()) {
    // Pruned by distance
    number_of_prunes_++;
  }
  // node->is_leaf() works as one would expect
  else if (query_node->is_leaf() && reference_node->is_leaf()) {
    // Base Case
    ComputeBaseCase_(query_node, reference_node);
  } else if (query_node->is_leaf()) {
    // Only query is a leaf
      
    // incrementing the number of margin computations
    nn_mc_[query_]++;

    // We'll order the computation by distance 
    double left_distance = MinNodeDistSq_(query_node,
					  reference_node->left());
    double right_distance = MinNodeDistSq_(query_node,
					   reference_node->right());
      
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
  }

  return;
} // ComputeNeighborsRecursion_

void TCNN::Init(const Matrix& references_in,
		struct datanode* module_in) {
    
  // set the module
  module_ = module_in;
    
  // Get the leaf size from the module
  leaf_size_ = fx_param_int(module_, "leaf_size", 30);
  // Make sure the leaf size is valid
  DEBUG_ASSERT(leaf_size_ > 0);
    
  // Copy the matrices to the class members since they will be rearranged.  
  references_.Copy(references_in);
    
  // keep a track of the dataset
  fx_param_int(module_, "dim", references_.n_rows());
  fx_param_int(module_, "rsize", references_.n_cols());

  // We'll time tree building
  fx_timer_start(module_, "tree_building");

  // This call makes each tree from a matrix, leaf size, and two arrays 
  // that record the permutation of the data points
  // Instead of NULL, it is possible to specify an array new_from_old_

  reference_tree_
    = tree::MakeKdTreeMidpoint<TreeType>(references_, 
					 leaf_size_,
					 &old_from_new_references_,
					 NULL);
    
  // Stop the timer we started above
  fx_timer_stop(module_, "tree_building");

  return;
} // Init

void TCNN::InitQueries(const Matrix& queries,
		       const Matrix& rank_matrix) {

  // track the number of prunes
  number_of_prunes_ = 0;

  // loading in the data
  rank_file_too_big_ = false;
  queries_.Copy(queries);
  rank_matrix_.Copy(rank_matrix);
  fx_param_int(module_, "qsize", queries_.n_cols());

  //stupid init
  rank_vec_.Init(0);

  // The data sets need to have the same number of dimensions
  DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());

  // K-nearest neighbors initialization
  knns_ = fx_param_int(module_, "knns", 1);
    
  // Initialize the list of nearest neighbor candidates
  neighbor_indices_.Init(queries_.n_cols() * knns_);

  // Initialize the vector of upper bounds for each point.  
  neighbor_distances_.Init(queries_.n_cols() * knns_);

  nn_dc_.Init(queries_.n_cols());
  nn_mc_.Init(queries_.n_cols());

  error_list_ = new std::vector<std::vector<size_t>* >();
  nn_dc_list_ = new std::vector<std::vector<size_t>* >();


  // Here we need to change the query tree into N single-point
  // query trees
  for (size_t i = 0; i < queries_.n_cols(); i++) {
    Matrix query;
    queries_.MakeColumnSlice(i, 1, &query);
    TreeType *single_point_tree
      = tree::MakeKdTreeMidpoint<TreeType>(query,
					   leaf_size_, 
					   &old_from_new_queries_,
					   NULL);
    query_trees_.push_back(single_point_tree);
    old_from_new_queries_.Renew();
  }

  return;
} // InitQueries

void TCNN::InitQueries(const Matrix& queries,
		       const std::string rank_matrix_file) {

  // track the number of prunes
  number_of_prunes_ = 0;

  // loading in the data
  rank_file_too_big_ = true;
  queries_.Copy(queries);
  fx_param_int(module_, "qsize", queries_.n_cols());

  rank_fp_ = fopen(rank_matrix_file.c_str(), "r");

  // stupid init
  rank_matrix_.Init(0,0);
  rank_vec_.Init(0);

  // The data sets need to have the same number of dimensions
  DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());

  // K-nearest neighbors initialization
  knns_ = fx_param_int(module_, "knns", 1);
    
  // Initialize the list of nearest neighbor candidates
  neighbor_indices_.Init(queries_.n_cols() * knns_);

  // Initialize the vector of upper bounds for each point.  
  neighbor_distances_.Init(queries_.n_cols() * knns_);

  nn_dc_.Init(queries_.n_cols());
  nn_mc_.Init(queries_.n_cols());

  error_list_ = new std::vector<std::vector<size_t>* >();
  nn_dc_list_ = new std::vector<std::vector<size_t>* >();


  // Here we need to change the query tree into N single-point
  // query trees
  for (size_t i = 0; i < queries_.n_cols(); i++) {
    Matrix query;
    queries_.MakeColumnSlice(i, 1, &query);
    TreeType *single_point_tree
      = tree::MakeKdTreeMidpoint<TreeType>(query,
					   leaf_size_, 
					   &old_from_new_queries_,
					   NULL);
    query_trees_.push_back(single_point_tree);
    old_from_new_queries_.Renew();
  }

  return;
} // InitQueries


void TCNN::ComputeNeighborsSequential(ArrayList<double> *means,
				      ArrayList<double> *stds,
				      ArrayList<size_t> *maxs,
				      ArrayList<size_t> *mins) {

  // initialize the dc, mc for all the queries for every round
  for (size_t i = 0; i < queries_.n_cols(); i++) {
    nn_dc_[i] = 0;
    nn_mc_[i] = 0;
  }
  // initialize neighbor distances
  neighbor_distances_.SetAll(DBL_MAX);

  DEBUG_ASSERT((size_t)query_trees_.size() == queries_.n_cols());

  // doing a single tree search for each of the queries one by one
  query_ = 0;
  size_t max_num_of_leaves = 0;
  double perc_done = 10;
  double done_sky = 1;


  for (std::vector<TreeType*>::iterator query_tree = query_trees_.begin();
       query_tree < query_trees_.end(); ++query_tree, ++query_) {

    // initialize the number of leaves visited
    number_of_leaves_ = 0;
    std::vector<size_t> *q_error_list
      = new std::vector<size_t>();
    error_list_->push_back(q_error_list);
    std::vector<size_t> *q_nn_dc_list
      = new std::vector<size_t>();
    nn_dc_list_->push_back(q_nn_dc_list);


    // loading up the line of the rank file when
    // the rank file is too big to load in memory
    if (rank_file_too_big_ && rank_fp_ != NULL) {
      rank_vec_.Destruct();
      rank_vec_.Init(references_.n_cols());
      char *line = NULL;
      size_t len = 0;
      getline(&line, &len, rank_fp_);

      char *pch = strtok(line, ",\n");
      size_t rank_index = 0;
      while (pch != NULL) {
	rank_vec_[rank_index++] = atoi(pch);
	pch = strtok(NULL, ",\n");
      }

      free(line);
      free(pch);
      DEBUG_ASSERT(rank_index == rank_vec_.length());

    }


    // traversing the reference tree for the computing the 
    // nearest neighbor
    ComputeNeighborsRecursion_(*query_tree, reference_tree_, 
			       MinNodeDistSq_(*query_tree,
					      reference_tree_));

    if (number_of_leaves_ > max_num_of_leaves)
      max_num_of_leaves = number_of_leaves_;

    double pdone = query_ * 100 / queries_.n_cols();

    if (pdone >= done_sky * perc_done) {
      if (done_sky > 1) {
	printf("\b\b\b=%zud%%", (size_t) pdone); fflush(NULL); 
      } else {
	printf("=%zud%%", (size_t) pdone); fflush(NULL);
      }
      done_sky++;
    }
  }

  double pdone = query_ * 100 / queries_.n_cols();

  if (pdone >= done_sky * perc_done) {
    if (done_sky > 1) {
      printf("\b\b\b=%zud%%", (size_t) pdone); fflush(NULL); 
    } else {
      printf("=%zud%%", (size_t) pdone); fflush(NULL);
    }
    done_sky++;
  }
  printf("\n");fflush(NULL);

  DEBUG_ASSERT(query_ == queries_.n_cols());

  if (rank_file_too_big_ && rank_fp_ != NULL)
    fclose(rank_fp_);


  double avg_dc = 0.0, avg_mc = 0.0;
  for (size_t i = 0; i < queries_.n_cols(); i++) {
    avg_dc += nn_dc_[i];
    avg_mc += nn_mc_[i];
  }

  avg_dc /= (double) queries_.n_cols();
  avg_mc /= (double) queries_.n_cols();
      
  NOTIFY("NN: Avg. DC: %lg, Avg. MC: %lg", avg_dc, avg_mc);
  NOTIFY("Max. Leaves: %zud", max_num_of_leaves);

  // computing the time constrained search errors
  // the mean, std, max, min

  // initializing the stat vectors
  means->Init(max_num_of_leaves);
  stds->Init(max_num_of_leaves);
  maxs->Init(max_num_of_leaves);
  mins->Init(max_num_of_leaves);

  for (size_t i = 0; i < max_num_of_leaves; i++) {
    (*means)[i] = 0;
    (*stds)[i] = 0;
    (*maxs)[i] = -1;
    (*mins)[i] = references_.n_cols();
  }

  // looping over every query
  for (size_t i = 0; i < query_; i++) {
    std::vector<size_t> *q_error_list
      = (*error_list_)[i];

    // looping over the leaves visited
    for (size_t j = 0; j < (size_t) q_error_list->size(); j++) {
      (*means)[j] += (*q_error_list)[j];
      (*stds)[j] += ((*q_error_list)[j] * (*q_error_list)[j]);

      if ((*q_error_list)[j] > (*maxs)[j])
	(*maxs)[j] = (*q_error_list)[j];

      if ((*q_error_list)[j] < (*mins)[j])
	(*mins)[j] = (*q_error_list)[j];
    } // leaves loop
  } // queries loop

  // final fixes on the mean and std vectors
  for(size_t i = 0; i < max_num_of_leaves; i++) {
    (*stds)[i] = sqrt((query_ * (*stds)[i]) - ((*means)[i] * (*means)[i]))
      / query_;
    (*means)[i] /= query_;
  }

  return;
} // ComputeNeighborsSequential

