#include "approx_nn.h"

void ApproxNN::ComputeBaseCase_(TreeType* query_node,
				TreeType* reference_node) {
   

  // do either if it is training or exact test or you have 
  // leaves left for approx test.
  if (test_ann_ == 0 || number_of_leaves_ < max_leaves_) {

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
    std::vector<std::pair<double, index_t> > neighbors(knns_);

    // Get the query point from the matrix
    Vector query_point;

    // FIX HERE: queries_ -> test_queries_
    queries_.MakeColumnVector(query_, &query_point);
      
    // FIX HERE: query_ -> test_query_
    index_t ind = query_*knns_;
    for(index_t i=0; i<knns_; i++) {
      neighbors[i]=std::make_pair(neighbor_distances_[ind+i],
				  neighbor_indices_[ind+i]);
    }

    // We'll do the same for the references
    for (index_t reference_index = reference_node->begin(); 
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
    for(index_t i=0; i<knns_; i++) {
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

    // FIX HERE:
    // updating the number of distance computations done
    nn_dc_[query_] += reference_node->end() - reference_node->begin();

    // storing the information regarding the first nn candidate
    // which is also considered the ann......
    if (train_nn_ == 1) { // first leaf encountered

      //       for(index_t i=0; i<knns_; i++) {

      // 	ann_dist_[ind+i] = neighbor_distances_[ind+i];
      // 	ann_ind_[ind+i]  = neighbor_indices_[ind+i];
      //       }

      //       ann_mc_[query_] = nn_mc_[query_];
      //       ann_dc_[query_] = nn_dc_[query_];

      //     if (query_ == 449)
      // 	NOTIFY("here %"LI"d, %"LI"d", ind, number_of_leaves_);


      ///// BIG FIX HERE: generalize for knns

      // updating the u_q value for this query
      DEBUG_ASSERT(query_ == (index_t) u_q_->size() - 1);

      // storing the u_q value
      // IMP: for u_q_ to work with all different kinds of errors
      // 0-1 error, dist error, rank error,
      // we need the u_q_ value to be dependent on those error

      double dist_eps = sqrt(neighbor_distances_[ind] 
			     / calc_nn_dists_[ind]) - 1;

      if (((*u_q_)[query_] == -1 )
	  && (calc_nn_dists_[ind] == neighbor_distances_[ind]))
	(*u_q_)[query_] = number_of_leaves_ + 1;


      DEBUG_ASSERT((index_t)error_list_->size() >= number_of_leaves_);

      if ((index_t)error_list_->size() <= number_of_leaves_) {

	DEBUG_ASSERT((index_t)error_list_->size() == number_of_leaves_);


	// have to add an element at the end of the vector
	// if (calc_nn_dists_[ind] < neighbor_distances_[ind]) {
	// HACK: if you care about exact, make dist_epsilon_ == 0.0
	if (dist_eps > dist_epsilon_) {
	  // error in the approx nn
	  error_list_->push_back(1);
	} else {
	  error_list_->push_back(0);
	}

	dist_error_list_->push_back(dist_eps);
	sq_dist_error_list_->push_back(dist_eps * dist_eps);
	max_dist_error_list_->push_back(dist_eps);

	ann_mc_->push_back(nn_mc_[query_] + last_mc_val_sum_);

	ann_dc_->push_back(nn_dc_[query_] + last_dc_val_sum_);
	sq_ann_dc_->push_back(nn_dc_[query_]*nn_dc_[query_]
			      + last_sq_dc_val_sum_);
	max_ann_dc_->push_back(nn_dc_[query_]);

      } else {
	// vector place holder already present for this number of leaves
	if (dist_eps > dist_epsilon_) {
	  // error in the approx nn
	  (*error_list_)[number_of_leaves_]++;
	  (*dist_error_list_)[number_of_leaves_] += dist_eps;
	  (*sq_dist_error_list_)[number_of_leaves_] 
	    += (dist_eps * dist_eps);

	  if ((*max_dist_error_list_)[number_of_leaves_] < dist_eps)
	    (*max_dist_error_list_)[number_of_leaves_] = dist_eps;
	}
	(*ann_mc_)[number_of_leaves_] += nn_mc_[query_];

	(*ann_dc_)[number_of_leaves_] += nn_dc_[query_];
	(*sq_ann_dc_)[number_of_leaves_] 
	  += (nn_dc_[query_] * nn_dc_[query_]);

	if ((*max_ann_dc_)[number_of_leaves_] < nn_dc_[query_])
	  (*max_ann_dc_)[number_of_leaves_] = nn_dc_[query_];
      }

      //       if (query_ == 449)
      // 	NOTIFY("there");

    } // diagnostics while training
    number_of_leaves_++;
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!//
    // Need to fix for the case where the leaf size might
    // be less than the knn - but later
  }         
} // ComputeBaseCase_


void ApproxNN::InitTrain(const Matrix& queries_in,
			 const Matrix& references_in,
			 struct datanode* module_in) {
    
  // set the module
  module_ = module_in;
    
  // track the number of prunes
  number_of_prunes_ = 0;
    
  // Get the leaf size from the module
  leaf_size_ = fx_param_int(module_, "leaf_size", 30);
  dist_epsilon_ = fx_param_double(module_, "dist_epsilon", 0.0);
  alpha_ = fx_param_double(module_, "alpha", 0.0);
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
  //     ann_ind_.Init(neighbor_indices_.size());

  nn_dc_.Init(queries_.n_cols());
  nn_mc_.Init(queries_.n_cols());

  error_list_ = new std::vector<index_t>();
  dist_error_list_ = new std::vector<double>();
  sq_dist_error_list_ = new std::vector<double>();
  max_dist_error_list_ = new std::vector<double>();
    
  ann_dc_ = new std::vector<long int>();
  sq_ann_dc_ = new std::vector<long int>();
  max_ann_dc_ = new std::vector<long int>();
  ann_mc_ = new std::vector<long int>();

  u_q_ = new std::vector<int>();
  v_q_ = new std::vector<int>();

  // Initialize the vector of upper bounds for each point.  
  neighbor_distances_.Init(queries_.n_cols() * knns_);
  calc_nn_dists_.Init(queries_.n_cols() * knns_);


  //     ann_dist_.Init(queries_.n_cols() * knns_);

  // We'll time tree building
  fx_timer_start(module_, "tree_building");

  // This call makes each tree from a matrix, leaf size, and two arrays 
  // that record the permutation of the data points
  // Instead of NULL, it is possible to specify an array new_from_old_

  // Here we need to change the query tree into N single-point
  // query trees
  for (index_t i = 0; i < queries_.n_cols(); i++) {
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
  reference_tree_
    = tree::MakeKdTreeMidpoint<TreeType>(references_, 
					 leaf_size_,
					 &old_from_new_references_,
					 NULL);
    
  // Stop the timer we started above
  fx_timer_stop(module_, "tree_building");

  // setting up training sequence here

  // make sure you clean up all the pointers which 
  // will be reused while testing

} // Init


void ApproxNN::InitTest(const Matrix& queries_in) {
    
  // track the number of prunes
  number_of_prunes_ = 0;
    
  // Copy the matrices to the class members since they will be rearranged.  
  queries_.Destruct();
  queries_.Copy(queries_in);

  // The data sets need to have the same number of points
  DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());
    
  // keep a track of the dataset
  fx_param_int(module_, "test_qsize", queries_.n_cols());

  // Initialize the list of nearest neighbor candidates
  neighbor_indices_.Renew();
  neighbor_indices_.Init(queries_.n_cols() * knns_);
  //     ann_ind_.Init(neighbor_indices_.size());

  nn_dc_.Renew();
  nn_dc_.Init(queries_.n_cols());

  nn_mc_.Renew();
  nn_mc_.Init(queries_.n_cols());

//   error_list_ = new std::vector<index_t>();
//   dist_error_list_ = new std::vector<double>();
//   sq_dist_error_list_ = new std::vector<double>();
//   max_dist_error_list_ = new std::vector<double>();
    
//   ann_dc_ = new std::vector<long int>();
//   sq_ann_dc_ = new std::vector<long int>();
//   max_ann_dc_ = new std::vector<long int>();
//   ann_mc_ = new std::vector<long int>();

//   u_q_ = new std::vector<int>();
//   v_q_ = new std::vector<int>();

  // Initialize the vector of upper bounds for each point.  
  neighbor_distances_.Destruct();
  neighbor_distances_.Init(queries_.n_cols() * knns_);

  calc_nn_dists_.Destruct();
  calc_nn_dists_.Init(queries_.n_cols() * knns_);

  //     ann_dist_.Init(queries_.n_cols() * knns_);

  // We'll time tree building
  fx_timer_start(module_, "tree_building");

  // This call makes each tree from a matrix, leaf size, and two arrays 
  // that record the permutation of the data points
  // Instead of NULL, it is possible to specify an array new_from_old_

  // Here we need to change the query tree into N single-point
  // query trees
  query_trees_.clear();
  for (index_t i = 0; i < queries_.n_cols(); i++) {
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
  // Stop the timer we started above
  fx_timer_stop(module_, "tree_building");

} // Init

inline double ConstrFun(std::vector<int> *data, int thres) {

  index_t num_over = 0;
  for (index_t i = 0; i < (index_t) data->size(); i++)
    if ((*data)[i] > thres)
      num_over++;

  return (double) num_over / (double) data->size();
}

inline double ObjFun(std::vector<int> *data, int thres) {

  int saved = 0;
  for (index_t i = 0; i < (index_t) data->size(); i++)
    if ((*data)[i] <= thres)
      saved += (thres - (*data)[i]);

  return (double) saved / (double) data->size();

}

/**
 * Computes the nearest neighbors and stores them in *results
 */
void ApproxNN::TrainNeighbors() {

  FILE *fp;
  fp = NULL;

  if (fx_param_exists(module_, "e_v_dc_file")) {
    std::string fname = fx_param_str_req(module_, "e_v_dc_file");
    fp = fopen(fname.c_str(), "w");
  }

  FILE *fp1;
  fp1 = NULL;

  if (fx_param_exists(module_, "uq_vq_file")) {
    std::string fname = fx_param_str_req(module_, "uq_vq_file");
    fp1 = fopen(fname.c_str(), "w");
  }

  // initialize the dc, mc for all the queries for every round
  for (index_t i = 0; i < queries_.n_cols(); i++) {
    nn_dc_[i] = 0;
    nn_mc_[i] = 0;
    // 	ann_dc_[i] = 0;
    // 	ann_mc_[i] = 0;
  }
    
  // initialize neighbor distances
  neighbor_distances_.SetAll(DBL_MAX);
  //       ann_dist_.SetAll(DBL_MAX);

  // initialize neighbor indices
  // do we need it?

  // if first round, compute nn
  //       if (numLeaves == 1)
  // 	compute_nn_ = 1;
  //       else
  // 	compute_nn_ = 0;

  // Start on the root of each tree
  // the index of the query in the queries_ matrix


  // computing the true nns
  train_nn_ = 0;
  test_ann_ = 0;

  query_ = 0;
  DEBUG_ASSERT((index_t)query_trees_.size() == queries_.n_cols());
  for (std::vector<TreeType*>::iterator query_tree = query_trees_.begin();
       query_tree < query_trees_.end(); ++query_tree, ++query_) {

    ComputeNeighborsRecursion_(*query_tree, reference_tree_, 
			       MinNodeDistSq_(*query_tree,
					      reference_tree_));
  }


  for (index_t i = 0; i < neighbor_indices_.size(); i++) {
    calc_nn_dists_[i] = neighbor_distances_[i];
  }

  double avg_dc = 0.0, avg_mc = 0.0;
  for (index_t i = 0; i < queries_.n_cols(); i++) {
    avg_dc += nn_dc_[i];
    avg_mc += nn_mc_[i];
  }

  avg_dc /= (double) queries_.n_cols();
  avg_mc /= (double) queries_.n_cols();
      
  NOTIFY("NN: Avg. DC: %lg, Avg. MC: %lg", avg_dc, avg_mc);

  // computing the anns progressively
  neighbor_distances_.SetAll(DBL_MAX);
  NOTIFY("Training Starting....");
  for (index_t i = 0; i < queries_.n_cols(); i++) {
    nn_dc_[i] = 0;
    nn_mc_[i] = 0;
  }

  train_nn_ = 1;
  last_dc_val_sum_ = 0;
  last_sq_dc_val_sum_ = 0;
 
  last_mc_val_sum_ = 0;

  index_t num_queries =  queries_.n_cols();

  query_ = 0;
  DEBUG_ASSERT((index_t)query_trees_.size() == queries_.n_cols());
  for (std::vector<TreeType*>::iterator query_tree = query_trees_.begin();
       query_tree < query_trees_.end(); ++query_tree, ++query_) {

    number_of_leaves_ = 0;
    u_q_->push_back(-1);

    ComputeNeighborsRecursion_(*query_tree, reference_tree_, 
			       MinNodeDistSq_(*query_tree,
					      reference_tree_));

    v_q_->push_back(number_of_leaves_);

    while (number_of_leaves_ < (index_t)error_list_->size()) {
      (*ann_dc_)[number_of_leaves_] += nn_dc_[query_];
      (*sq_ann_dc_)[number_of_leaves_] 
	+= (nn_dc_[query_] * nn_dc_[query_]);

      if ((*max_ann_dc_)[number_of_leaves_] < nn_dc_[query_])
	(*max_ann_dc_)[number_of_leaves_] = nn_dc_[query_];

      (*ann_mc_)[number_of_leaves_++] += nn_mc_[query_];
    }

    last_dc_val_sum_ += nn_dc_[query_];
    last_sq_dc_val_sum_ += (nn_dc_[query_]*nn_dc_[query_]);

    last_mc_val_sum_ += nn_mc_[query_];
  }

  NOTIFY("Training data obtained");

  DEBUG_ASSERT(error_list_->size() == ann_dc_->size());
  DEBUG_ASSERT(error_list_->size() == ann_mc_->size());


  for (index_t i = 0; i < (index_t)error_list_->size(); i++) {
    // 	NOTIFY("%"LI"d: E:%"LI"d/%"LI"d, Avg. DC: %lg, Avg. MC: %lg",
    // 	       i+1, (*error_list_)[i], queries_.n_cols(),
    // 	       (double) (*ann_dc_)[i] / (double) queries_.n_cols(),
    // 	       (double) (*ann_mc_)[i] / (double) queries_.n_cols());
    long double mean_dc = (long double) (*ann_dc_)[i]
      / (long double)num_queries;
    DEBUG_ASSERT((long double) (*sq_ann_dc_)[i] 
		 / (long double) num_queries >=  (mean_dc * mean_dc));
    long double sd_dc
      = sqrt((long double) (*sq_ann_dc_)[i]
	     / (long double) num_queries -  (mean_dc * mean_dc));

    double mean_dist_error = (*dist_error_list_)[i]
      / (double) num_queries;
    double sd_dist_error
      = sqrt((*sq_dist_error_list_)[i] / (double) num_queries
	     - (mean_dist_error * mean_dist_error));

    double mean_dist_error_var = 0.0, sd_dist_error_var = 0.0;
    if ((*error_list_)[i] != 0) {
      mean_dist_error_var = (*dist_error_list_)[i]
	/ (double) (*error_list_)[i];
      sd_dist_error_var
	= sqrt((*sq_dist_error_list_)[i] / (double) (*error_list_)[i]
	       - (mean_dist_error_var * mean_dist_error_var));
    }

    if (fp != NULL) 
      fprintf(fp, "%lg,%lg,%lg,%ld,"
	      "%ld,%lg,%lg,%lg,%lg,%lg\n",
	      (double) (*error_list_)[i] / (double) num_queries,
	      (double)mean_dc, (double)sd_dc, (*max_ann_dc_)[i],
	      (*ann_mc_)[i],
	      mean_dist_error, sd_dist_error, 
	      mean_dist_error_var, sd_dist_error_var,
	      (*max_dist_error_list_)[i]);
  }

  int g_min = -1, g_max = 0;
  for (index_t i = 0; i < num_queries; i++) {
    if (fp1 != NULL)
      fprintf(fp1, "%d,%d\n", (*u_q_)[i], (*v_q_)[i]);

    if (g_min == -1 || g_min > (*u_q_)[i])
      g_min = (*u_q_)[i];

    if (g_max < (*u_q_)[i])
      g_max = (*u_q_)[i];
  }

  // here you compute the gamma = max_leaves_ value using 
  // the u_q values
  DEBUG_ASSERT((index_t)u_q_->size() == num_queries);

  int gamma = g_max;
  int min_gam = gamma;
  double min_obj_fun = ObjFun(u_q_, gamma);

  max_leaves_ = min_gam;
  NOTIFY("Max.Gam: %d, Min. Gam: %d, Gam.: %d, Acc. prob.:%lg",
	 g_max, g_min, max_leaves_, ConstrFun(u_q_, max_leaves_));

  while(((double)(*error_list_)[gamma-1] / (double) num_queries
	<= alpha_) && (gamma > g_min -1)) {

    double obj_fun = ObjFun(u_q_, gamma);

    if (min_obj_fun > obj_fun) {
      min_obj_fun = obj_fun;
      min_gam = gamma;
    }

    double test_fail = (double) (*error_list_)[gamma -1]
      / (double) num_queries;
    DEBUG_ASSERT_MSG(test_fail == ConstrFun(u_q_, gamma), 
		     "ELE:%lg, ConsFun:%lg, gamma:%"LI"d",
		     test_fail, ConstrFun(u_q_, gamma), gamma);

    gamma--;

  }

  max_leaves_ = min_gam;
  NOTIFY("Max.Gam: %d, Min. Gam: %d, Gam.: %d, Acc. prob.:%lg",
	 g_max, g_min, max_leaves_, ConstrFun(u_q_, max_leaves_));

  NOTIFY("Error list entry: %lg",
	 (double) (*error_list_)[max_leaves_-1] 
	 / (double) num_queries);



  //       error = 0;
  //       double max_eps = -1.0, avg_eps = 0.0;
  //       for (index_t i = 0; i < nn_dist->size(); i++) {
  // 	if ((*nn_dist)[i] < ann_dist_[i]) {
  // 	  error++;
  // 	  double epsilon = (ann_dist_[i] / (*nn_dist)[i]) - 1;
  // 	  avg_eps += epsilon;
  // 	  if (epsilon > max_eps)
  // 	    max_eps = epsilon;
  // 	}
  //       }
  //       avg_eps /= (double) queries_.n_cols();

  //       NOTIFY("%"LI"d: E:%"LI"d/%"LI"d, AEps:%lg, MEps:%lg",
  // 	     numLeaves, error, queries_.n_cols(), avg_eps, max_eps); 

  //       double avg_dc = 0.0, avg_mc = 0.0;
  //       for (index_t i = 0; i < queries_.n_cols(); i++) {
  // 	avg_dc += ann_dc_[i];
  // 	avg_mc += ann_mc_[i];
  //       }

  //       avg_dc /= (double) queries_.n_cols();
  //       avg_mc /= (double) queries_.n_cols();

  //       NOTIFY("ANN: Avg. DC: %lg, Avg. MC: %lg", avg_dc, avg_mc);

  //       numLeaves++;

  //       if (fp != NULL) 
  // 	fprintf(fp, "%"LI"d,%lg,%lg\n", error, avg_dc, avg_mc);

  //     } while (error > 0);

  if (fp != NULL)
    fclose(fp);

  if (fp1 != NULL)
    fclose(fp1);

  return;

} // TrainNeighbors



// should be called after calling InitTest
void ApproxNN::TestNeighbors(ArrayList<index_t>* nn_ind,
			     ArrayList<double>* nn_dist, 
			     ArrayList<index_t>* ann_ind,
			     ArrayList<double>* ann_dist) {


  // We need to initialize the results list before filling it
  nn_ind->Init(neighbor_indices_.size());
  nn_dist->Init(neighbor_distances_.length());

  ann_ind->Init(neighbor_indices_.size());
  ann_dist->Init(neighbor_distances_.length());


  // initialize the dc, mc for all the queries for every round
  for (index_t i = 0; i < queries_.n_cols(); i++) {
    nn_dc_[i] = 0;
    nn_mc_[i] = 0;
  }
    
  // initialize neighbor distances
  neighbor_distances_.SetAll(DBL_MAX);

  // computing the true nns
  train_nn_ = 0;
  test_ann_ = 0;

  query_ = 0;
  DEBUG_ASSERT((index_t)query_trees_.size() == queries_.n_cols());
  for (std::vector<TreeType*>::iterator query_tree = query_trees_.begin();
       query_tree < query_trees_.end(); ++query_tree, ++query_) {

    ComputeNeighborsRecursion_(*query_tree, reference_tree_, 
			       MinNodeDistSq_(*query_tree,
					      reference_tree_));
  }

  for (index_t i = 0; i < neighbor_indices_.size(); i++) {
    index_t query = i/knns_;
    (*nn_ind)[query*knns_+ i%knns_]
      = old_from_new_references_[neighbor_indices_[i]];
    (*nn_dist)[query*knns_+ i%knns_] = neighbor_distances_[i];

    calc_nn_dists_[i] = neighbor_distances_[i];

    // 	  (*ann_ind)[query*knns_+ i%knns_]
    // 	    = old_from_new_references_[ann_ind_[i]];
    // 	  (*ann_dist)[query*knns_+ i%knns_] = ann_dist_[i];
  }

  double avg_dc = 0.0, avg_mc = 0.0;
  for (index_t i = 0; i < queries_.n_cols(); i++) {
    avg_dc += nn_dc_[i];
    avg_mc += nn_mc_[i];
  }

  avg_dc /= (double) queries_.n_cols();
  avg_mc /= (double) queries_.n_cols();
      
  NOTIFY("Test Exact NN: Avg. DC: %lg, Avg. MC: %lg", avg_dc, avg_mc);


  // computing the anns progressively
  neighbor_distances_.SetAll(DBL_MAX);
  for (index_t i = 0; i < queries_.n_cols(); i++) {
    nn_dc_[i] = 0;
    nn_mc_[i] = 0;
  }

  test_ann_ = 1;
  index_t num_queries =  queries_.n_cols();

  query_ = 0;
  DEBUG_ASSERT((index_t)query_trees_.size() == queries_.n_cols());
  for (std::vector<TreeType*>::iterator query_tree = query_trees_.begin();
       query_tree < query_trees_.end(); ++query_tree, ++query_) {

    number_of_leaves_ = 0;
    ComputeNeighborsRecursion_(*query_tree, reference_tree_, 
			       MinNodeDistSq_(*query_tree,
					      reference_tree_));
  }

  index_t error = 0;
  for (index_t i = 0; i < neighbor_indices_.size(); i++) {
    index_t query = i/knns_;

    (*ann_ind)[query*knns_+ i%knns_]
      = old_from_new_references_[neighbor_indices_[i]];
    (*ann_dist)[query*knns_+ i%knns_] = neighbor_distances_[i];
    
    double dist_eps = (neighbor_distances_[i] / calc_nn_dists_[i]) - 1.0;
    if (dist_eps > dist_epsilon_)
      error++;

  }

  NOTIFY("ANN search done");

  avg_dc = 0.0;
  avg_mc = 0.0;
  for (index_t i = 0; i < queries_.n_cols(); i++) {
    avg_dc += nn_dc_[i];
    avg_mc += nn_mc_[i];
  }

  avg_dc /= (double) queries_.n_cols();
  avg_mc /= (double) queries_.n_cols();
      
  NOTIFY("Test ANN: Avg. DC: %lg, Avg. MC: %lg", avg_dc, avg_mc);
  NOTIFY("Error: %"LI"d / %"LI"d = %lg",
	 error, num_queries, (double)error / (double) num_queries);


  return;

} // TestNeighbors

