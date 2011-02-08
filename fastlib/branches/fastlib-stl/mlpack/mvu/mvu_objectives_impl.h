/*
 * =====================================================================================
 * 
 *       Filename:  mvu_objectives_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  03/05/2008 04:20:27 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
#include <errno.h>

void MaxVariance::Init(datanode* module, arma::mat& data) {
  module_ = module;

  knns_ = fx_param_int(module_, "knns", 5);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  new_dimension_ = fx_param_int_req(module_, "new_dimension");
  num_of_points_ = data.n_cols;

  NOTIFY("Data loaded...\n");
  NOTIFY("Nearest neighbor constraints...\n");
  NOTIFY("Building tree with data...\n");

  if (knns_ == 0) {
    allknn_.Init(&data, leaf_size_, MAX_KNNS); 
  } else {
    allknn_.Init(&data, leaf_size_, knns_); 
  }

  NOTIFY("Tree built...\n");

  NOTIFY("Computing neighborhoods...\n");

  arma::Col<index_t> from_tree_neighbors;
  arma::vec          from_tree_distances;
  allknn_.ComputeNeighbors(from_tree_neighbors,
                           from_tree_distances);

  NOTIFY("Neighborhoods computed...\n");

  if (knns_ == 0) { // automatically estimate the correct number for k
    NOTIFY("Auto-tuning the knn...\n" );

    MaxVarianceUtils::EstimateKnns(from_tree_neighbors,
        from_tree_distances,
        MAX_KNNS, 
        data.n_cols,
        data.n_rows,
        knns_); 

    NOTIFY("Optimum knns is %i", knns_);
    fx_format_result(module_, "optimum_knns", "%i", knns_);

    NOTIFY("Consolidating neighbors...\n");
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
        from_tree_distances,
        MAX_KNNS,
        knns_,
        nearest_neighbor_pairs_,
        nearest_distances_,
        num_of_nearest_pairs_);
  } else { 
    NOTIFY("Consolidating neighbors...\n");
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
        from_tree_distances,
        knns_,
        knns_,
        nearest_neighbor_pairs_,
        nearest_distances_,
        num_of_nearest_pairs_);
  }
 
  eq_lagrange_mult_.ones(num_of_nearest_pairs_);
  double max_nearest_distance = 0;

  for(index_t i = 0; i < num_of_nearest_pairs_; i++) {
    max_nearest_distance = std::max(nearest_distances_[i], max_nearest_distance);
  }
  sum_of_furthest_distances_ = -max_nearest_distance * std::pow(data.n_cols, 2);
 
  NOTIFY("Lower bound for optimization is %lg", sum_of_furthest_distances_);
  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  fx_format_result(module_, "lower_optimal_bound", "%lg", sum_of_furthest_distances_);
}

void MaxVariance::Init(fx_module* module) {
  module_ = module;

  new_dimension_ = fx_param_int_req(module_, "new_dimension");
  std::string nearest_neighbor_file = fx_param_str(module, "nearest_neighbor_file", "nearest.txt");
  std::string furthest_neighbor_file = fx_param_str(module, "furthest_neighbor_file", "futhest.txt");

  // WHY ARE WE NOT USING DATA::LOAD()?  IS IT BECAUSE WE SUFFER FROM THE
  // STUPID?  YEAH?  IS THAT IT?
  FILE *fp = fopen(nearest_neighbor_file.c_str(), "r");
  if (fp == NULL) {
    FATAL("Error while opening %s...%s", nearest_neighbor_file.c_str(),
        strerror(errno));
  }
  nearest_neighbor_pairs_.clear();
  nearest_distances_.clear();
  num_of_points_ = 0;
  while(!feof(fp)) {
    index_t n1, n2;
    double distance;
    fscanf(fp, "%i %i %lg", &n1, &n2, &distance);
    nearest_neighbor_pairs_.push_back(std::make_pair(n1, n2));
    nearest_distances_.push_back(distance);
    if (n1 > num_of_points_) {
      num_of_points_ = n1;
    }
    if (n2 > num_of_points_) {
      num_of_points_ = n2;
    }
  }
  num_of_points_++;
  fclose(fp);   

  num_of_nearest_pairs_ = nearest_neighbor_pairs_.size();

  eq_lagrange_mult_.ones(num_of_nearest_pairs_);
  double max_nearest_distance = 0;
  for(index_t i = 0; i < num_of_nearest_pairs_; i++) {
    max_nearest_distance=std::max(nearest_distances_[i], max_nearest_distance);
  }
  sum_of_furthest_distances_ = -max_nearest_distance * std::pow(num_of_points_, 2);

  NOTIFY("Lower bound for optimization is %lg", sum_of_furthest_distances_);
  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  fx_format_result(module_, "lower_optimal_bound", "%lg", sum_of_furthest_distances_);
}

void MaxVariance::Destruct() {
  allknn_.Destruct(); // should be unnecessary
}

void MaxVariance::ComputeGradient(const arma::mat& coordinates, arma::mat& gradient) {
  gradient = -coordinates; // copy values and negate (we need to use -CRR^T
  // because we want to maximize CRR^T

  index_t dimension = coordinates.n_rows;
  for(index_t i = 0; i < num_of_nearest_pairs_; i++) {
    arma::vec a_i_r(dimension);
    index_t n1 = nearest_neighbor_pairs_[i].first;
    index_t n2 = nearest_neighbor_pairs_[i].second;
    const arma::vec point1 = coordinates.unsafe_col(n1);
    const arma::vec point2 = coordinates.unsafe_col(n2);
    double dist_diff = la::DistanceSqEuclidean(point1, point2) - nearest_distances_[i];
    
    a_i_r = point2 - point1;

    // equality constraints
    double scale = eq_lagrange_mult_[i] - dist_diff * sigma_;
    gradient.col(n1) -= scale * a_i_r;
    gradient.col(n2) += scale * a_i_r;
  }
}

void MaxVariance::ComputeObjective(const arma::mat& coordinates, double& objective) {
  objective = 0;

  index_t dimension = coordinates.n_rows;
  for(index_t i = 0; i < coordinates.n_cols; i++) {
    // could this be done more efficiently?
    objective -= dot(coordinates.unsafe_col(i), coordinates.unsafe_col(i));
  }
}

void MaxVariance::ComputeFeasibilityError(const arma::mat& coordinates, double& error) {
  error = 0;

  index_t dimension = coordinates.n_rows;
  for(index_t i = 0; i < num_of_nearest_pairs_; i++) {
    index_t n1 = nearest_neighbor_pairs_[i].first;
    index_t n2 = nearest_neighbor_pairs_[i].second;
    
    double distance = la::DistanceSqEuclidean(coordinates.unsafe_col(n1),
        coordinates.unsafe_col(n2));

    error += std::pow(distance - nearest_distances_[i], 2);
  }
}

double MaxVariance::ComputeLagrangian(const arma::mat& coordinates) {
  index_t dimension = coordinates.n_rows;

  double lagrangian = 0;
  ComputeObjective(coordinates, lagrangian);

  for(index_t i = 0; i < num_of_nearest_pairs_; i++) {
    index_t n1 = nearest_neighbor_pairs_[i].first;
    index_t n2 = nearest_neighbor_pairs_[i].second;
    
    double dist_diff = la::DistanceSqEuclidean(coordinates.unsafe_col(n1),
        coordinates.unsafe_col(n2)) - nearest_distances_[i];

    lagrangian += dist_diff * ((dist_diff * sigma_) -
        (eq_lagrange_mult_[i] * dist_diff));
  }

  return lagrangian;
}

void MaxVariance::UpdateLagrangeMult(const arma::mat& coordinates) {
  index_t dimension = coordinates.n_rows;

  for(index_t i = 0; i < num_of_nearest_pairs_; i++) {
    index_t n1 = nearest_neighbor_pairs_[i].first;
    index_t n2 = nearest_neighbor_pairs_[i].second;
    
    double dist_diff =la::DistanceSqEuclidean(coordinates.unsafe_col(n1),
        coordinates.unsafe_col(n2)) - nearest_distances_[i];
    eq_lagrange_mult_[i] -= (sigma_ * dist_diff);
  }
}

void MaxVariance::set_sigma(double sigma) {
  sigma_ = sigma;
}

bool MaxVariance::IsDiverging(double objective) {
  if (objective < sum_of_furthest_distances_) {
    NOTIFY("objective(%lg) < sum_of_furthest_distances (%lg)", objective,
        sum_of_furthest_distances_);
    return true;
  } else {
    return false;
  }
}

void MaxVariance::Project(arma::mat& coordinates) {
  OptUtils::RemoveMean(coordinates);
}

index_t MaxVariance::num_of_points() {
  return num_of_points_;
}

void MaxVariance::GiveInitMatrix(arma::mat& init_data) {
  init_data.randu(new_dimension_, num_of_points_);
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
void MaxFurthestNeighbors::Init(datanode* module, arma::mat& data) {
  module_ = module;
  new_dimension_ = fx_param_int_req(module_, "new_dimension");
  num_of_points_ = data.n_cols;

  infeasibility1_ = DBL_MAX;
  previous_infeasibility1_ = DBL_MAX;
  desired_feasibility_error_ = fx_param_double(module_, "desired_feasibility_error", 1);
  grad_tolerance_ = fx_param_double(module_, "grad_tolerance", 0.1);
  infeasibility_tolerance_=  fx_param_double(module_, "infeasibility_tolerance", 0.01);

  knns_ = fx_param_int(module_, "knns", 5);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);

  NOTIFY("Data loaded...\n");
  NOTIFY("Nearest neighbor constraints...\n");
  NOTIFY("Building tree with data...\n");

  if (knns_ == 0) {
     allknn_.Init(&data, leaf_size_, MAX_KNNS); 
  } else {
    allknn_.Init(&data, leaf_size_, knns_); 
  }

  NOTIFY("Tree built ...\n");
  NOTIFY("Computing neighborhoods ...\n");

  arma::Col<index_t> from_tree_neighbors;
  arma::vec from_tree_distances;
  allknn_.ComputeNeighbors(from_tree_neighbors,
                           from_tree_distances);

  NOTIFY("Neighborhoods computed...\n");

  if (knns_ == 0) { // automatically select k
    NOTIFY("Auto-tuning the knn...\n" );
    MaxVarianceUtils::EstimateKnns(from_tree_neighbors,
                                   from_tree_distances,
                                   MAX_KNNS, 
                                   data.n_cols,
                                   data.n_rows,
                                   knns_); 

    NOTIFY("Optimum knns is %i", knns_);
    fx_format_result(module_, "optimum_knns", "%i", knns_);
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
                                           from_tree_distances,
                                           MAX_KNNS,
                                           knns_,
                                           nearest_neighbor_pairs_,
                                           nearest_distances_,
                                           num_of_nearest_pairs_);
  } else { 
    NOTIFY("Consolidating neighbors...\n");
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
                                           from_tree_distances,
                                           knns_,
                                           knns_,
                                           nearest_neighbor_pairs_,
                                           nearest_distances_,
                                           num_of_nearest_pairs_);
  }

  // is there a better way to do this? (take dot product of distances)
  // this variable isn't even used before it's reset again!
//  sum_of_nearest_distances_ = 0;
//  for(int i = 0; i < nearest_distances_.size(); i++)
//    sum_of_nearest_distances_ += std::pow(nearest_distances_[i], 2.0);
//  sum_of_nearest_distances_ = std::pow(sum_of_nearest_distances_, 0.5);

  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);

  eq_lagrange_mult_.ones(num_of_nearest_pairs_);
  
  NOTIFY("Furthest neighbor constraints...\n");
  NOTIFY("Building tree with data...\n");

  allkfn_.Init(&data, leaf_size_, 1); 

  NOTIFY("Tree built...\n");
  NOTIFY("Computing furthest neighborhoods...\n");

  // mem:: madness?
//  from_tree_neighbors.Renew();
//  from_tree_distances.Renew();
  allkfn_.ComputeNeighbors(from_tree_neighbors,
                           from_tree_distances);

  NOTIFY("Furthest neighbors computed...\n");
  NOTIFY("Consolidating neighbors...\n");

  MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
                                         from_tree_distances,
                                         1,
                                         1,
                                         furthest_neighbor_pairs_,
                                         furthest_distances_,
                                         num_of_furthest_pairs_);

  double max_nearest_distance = 0;
  for(index_t i = 0; i < num_of_nearest_pairs_; i++) {
    max_nearest_distance = std::max(nearest_distances_[i], max_nearest_distance);
  }
  sum_of_furthest_distances_ = -(max_nearest_distance * data.n_cols * num_of_furthest_pairs_);
 
  NOTIFY("Lower bound for optimization: %lg", sum_of_furthest_distances_);
  fx_format_result(module_, "lower_optimal_bound", "%lg", sum_of_furthest_distances_);
}

void MaxFurthestNeighbors::Init(fx_module* module) {
  module_ = module;
  new_dimension_ = fx_param_int_req(module_, "new_dimension");

  infeasibility1_ = DBL_MAX;
  previous_infeasibility1_ = DBL_MAX;
  desired_feasibility_error_ = fx_param_double(module_, "desired_feasibility_error", 1);
  grad_tolerance_ = fx_param_double(module_, "grad_tolerance", 0.1);
  infeasibility_tolerance_ = fx_param_double(module_, "infeasibility_tolerance", 0.01);

  std::string nearest_neighbor_file = fx_param_str_req(module, "nearest_neighbor_file");
  std::string furthest_neighbor_file = fx_param_str_req(module, "furthest_neighbor_file");

  // need to use data::Load()
  FILE *fp = fopen(nearest_neighbor_file.c_str(), "r");
  if (fp == NULL) {
    FATAL("Error while opening %s...%s", nearest_neighbor_file.c_str(), strerror(errno));
  }
  num_of_points_ = 0;
  while(!feof(fp)) {
    index_t n1, n2;
    double distance;
    fscanf(fp, "%i %i %lg", &n1, &n2, &distance);
    nearest_neighbor_pairs_.push_back(std::make_pair(n1, n2));
    nearest_distances_.push_back(distance);
    if (n1 > num_of_points_) {
      num_of_points_=n1;
    }
    if (n2 > num_of_points_) {
      num_of_points_=n2;
    }
  }
  num_of_points_++;
  num_of_nearest_pairs_ = nearest_neighbor_pairs_.size();
  // this variable is not used before it is set again!
//  sum_of_nearest_distances_ = std::pow(dot(nearest_distances_[0],
//      nearest_distances_[0]), 0.5); // use norm()?

  fclose(fp);
  fp = fopen(furthest_neighbor_file.c_str(), "r");
  if (fp == NULL) {
    FATAL("Error while opening %s...%s", furthest_neighbor_file.c_str(), strerror(errno));
  }
  
  while(!feof(fp)) {
    index_t n1, n2;
    double distance;
    fscanf(fp, "%i %i %lg", &n1, &n2, &distance);
    furthest_neighbor_pairs_.push_back(std::make_pair(n1, n2));
    furthest_distances_.push_back(distance);
  }
  fclose(fp);
  num_of_furthest_pairs_ = furthest_neighbor_pairs_.size();

  eq_lagrange_mult_.ones(num_of_nearest_pairs_);
  double max_nearest_distance = 0;
  for(index_t i = 0; i < num_of_nearest_pairs_; i++) {
    max_nearest_distance = std::max(nearest_distances_[i], max_nearest_distance);
  }
  sum_of_furthest_distances_ = -(max_nearest_distance * num_of_points_ * num_of_points_);
 
  NOTIFY("Lower bound for optimization: %lg", sum_of_furthest_distances_);
  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  fx_format_result(module_, "lower_optimal_bound", "%lg", sum_of_furthest_distances_);
}

void MaxFurthestNeighbors::Destruct() {
  allknn_.Destruct(); // likely not necessary
  allkfn_.Destruct(); // likely not necessary
}

void MaxFurthestNeighbors::ComputeGradient(const arma::mat& coordinates, arma::mat& gradient) {
  gradient.zeros();
  // objective
  for(index_t i = 0; i < num_of_furthest_pairs_; i++) {
    arma::vec a_i_r(coordinates.n_rows);
    index_t n1 = furthest_neighbor_pairs_[i].first;
    index_t n2 = furthest_neighbor_pairs_[i].second;
    
    a_i_r = coordinates.unsafe_col(n2) - coordinates.unsafe_col(n1);

    gradient.col(n1) -= a_i_r;
    gradient.col(n2) += a_i_r;
  }
  // equality constraints
  for(index_t i = 0; i < num_of_nearest_pairs_; i++) {
    arma::vec a_i_r(coordinates.n_rows);

    index_t n1 = nearest_neighbor_pairs_[i].first;
    index_t n2 = nearest_neighbor_pairs_[i].second;
    arma::vec point1 = coordinates.unsafe_col(n1);
    arma::vec point2 = coordinates.unsafe_col(n2);
    double dist_diff = la::DistanceSqEuclidean(point1, point2) -
        nearest_distances_[i];

    a_i_r = point2 - point1;

    double scale = (dist_diff * sigma_ - eq_lagrange_mult_[i]);
    gradient.col(n1) += scale * a_i_r;
    gradient.col(n2) -= scale * a_i_r;
  }
}

void MaxFurthestNeighbors::ComputeObjective(const arma::mat& coordinates, double& objective) {
  objective = 0;
  
  for(index_t i = 0; i < num_of_furthest_pairs_; i++) {
    index_t n1 = furthest_neighbor_pairs_[i].first;
    index_t n2 = furthest_neighbor_pairs_[i].second;
    
    objective -= la::DistanceSqEuclidean(coordinates.unsafe_col(n1),
        coordinates.unsafe_col(n2));
  }
}

void MaxFurthestNeighbors::ComputeFeasibilityError(const arma::mat& coordinates, double& error) {
  error = 0;

  for(index_t i = 0; i < num_of_nearest_pairs_; i++) {
    index_t n1 = nearest_neighbor_pairs_[i].first;
    index_t n2 = nearest_neighbor_pairs_[i].second;
    
    double dist_diff = la::DistanceSqEuclidean(coordinates.unsafe_col(n1),
        coordinates.unsafe_col(n2)) - nearest_distances_[i];

    error += std::pow(dist_diff, 2);
  }

  error= 100 * (std::pow(error, 0.5) / sum_of_nearest_distances_);
}

double MaxFurthestNeighbors::ComputeLagrangian(const arma::mat& coordinates) {
  double lagrangian = 0;
  ComputeObjective(coordinates, lagrangian);

  for(index_t i = 0; i < num_of_nearest_pairs_; i++) {
    index_t n1 = nearest_neighbor_pairs_[i].first;
    index_t n2 = nearest_neighbor_pairs_[i].second;
    
    double dist_diff = la::DistanceSqEuclidean(coordinates.unsafe_col(n1),
        coordinates.unsafe_col(n2)) - nearest_distances_[i];

    lagrangian += (std::pow(dist_diff, 2) * sigma_ / 2) - (eq_lagrange_mult_[i] *
        dist_diff);
  }

  return lagrangian;
}

void MaxFurthestNeighbors::UpdateLagrangeMult(const arma::mat& coordinates) {
  for(index_t i = 0; i < num_of_nearest_pairs_; i++) {
    index_t n1 = nearest_neighbor_pairs_[i].first;
    index_t n2 = nearest_neighbor_pairs_[i].second;
    
    double dist_diff =la::DistanceSqEuclidean(coordinates.unsafe_col(n1),
        coordinates.unsafe_col(n2)) - nearest_distances_[i];

    eq_lagrange_mult_[i] -= sigma_ * dist_diff;
  }
}

void MaxFurthestNeighbors::set_sigma(double sigma) {
  sigma_ = sigma;
}

void MaxFurthestNeighbors::set_lagrange_mult(double val) {
  eq_lagrange_mult_.fill(val);
}

bool MaxFurthestNeighbors::IsDiverging(double objective) {
  if (objective < sum_of_furthest_distances_) {
    NOTIFY("objective(%lg) < sum_of_furthest_distances (%lg)", objective,
        sum_of_furthest_distances_);
    return true;
  } else {
    return false;
  }
}

void MaxFurthestNeighbors::Project(arma::mat& coordinates) {
  OptUtils::RemoveMean(coordinates);
}

index_t MaxFurthestNeighbors::num_of_points() {
  return num_of_points_;
}

void MaxFurthestNeighbors::GiveInitMatrix(arma::mat& init_data) {
  init_data.randu(new_dimension_, num_of_points_);
}

bool MaxFurthestNeighbors::IsOptimizationOver(arma::mat& coordinates, 
      arma::mat& gradient, double step) {
  ComputeFeasibilityError(coordinates, infeasibility1_);
  if (infeasibility1_ < desired_feasibility_error_ || 
      fabs(infeasibility1_ - previous_infeasibility1_) < infeasibility_tolerance_)  {
    NOTIFY("Optimization is over");
    return true;
  } else {
    previous_infeasibility1_ = infeasibility1_;
    return false; 
  }
}

bool MaxFurthestNeighbors::IsIntermediateStepOver(arma::mat& coordinates, 
      arma::mat& gradient, double step) {
  double norm_gradient = std::pow(dot(gradient, gradient), 0.5);
  double feasibility_error;
  ComputeFeasibilityError(coordinates, feasibility_error);
  if (norm_gradient * step < grad_tolerance_ 
      || feasibility_error < desired_feasibility_error_) {
    return true;
  }
  return false;
}


///////////////////////////////////////////////////////////////
void MaxVarianceUtils::ConsolidateNeighbors(const arma::Col<index_t>& from_tree_ind,
                                            const arma::vec&  from_tree_dist,
                                            index_t num_of_neighbors,
                                            index_t chosen_neighbors,
                                            std::vector<std::pair<index_t, index_t> >& neighbor_pairs,
                                            std::vector<double>& distances,
                                            index_t& num_of_pairs) {
  
  num_of_pairs = 0;
  index_t num_of_points = from_tree_ind.n_elem / num_of_neighbors;

  // just in case they aren't clear already
  neighbor_pairs.clear();
  distances.clear();

  bool skip = false;
  for(index_t i = 0; i < num_of_points; i++) {
    for(index_t k = 0; k < chosen_neighbors; k++) {  
      index_t n1 = i; // neighbor 1
      index_t n2 = from_tree_ind[(i * num_of_neighbors) + k]; // neighbor 2
      if(n1 > n2) {
        for(index_t n = 0; n < chosen_neighbors; n++) {
          if(from_tree_ind[(n2 * num_of_neighbors) + n] == n1) {
            skip = true;
            break;
          }
        }
      }  
      if(!skip) {
        num_of_pairs += 1;
        neighbor_pairs.push_back(std::make_pair(n1, n2));
        distances.push_back(from_tree_dist[(i * num_of_neighbors) + k]);
      }
      skip = false;
    }
  }
}

void MaxVarianceUtils::EstimateKnns(const arma::Col<index_t>& nearest_neighbors,
                                    const arma::vec& nearest_distances,
                                    index_t maximum_knns,
                                    index_t num_of_points,
                                    index_t dimension,
                                    index_t& optimum_knns) {
  double max_loocv_score = -DBL_MAX;
  double loocv_score = 0;
  // double unit_sphere_volume = math::SphereVolume(1.0, dimension);

  optimum_knns = 0;

  for(index_t k = 2; k < maximum_knns; k++) {
    loocv_score = 0.0;
    double mean_band = 0.0;
    for(index_t i = 0; i < num_of_points; i++) {
      double scale_factor = std::pow(nearest_distances[(i * maximum_knns) + k], (double) dimension / 2.0);
      double probability = 0;

      for(index_t j = 0; j < k; j++) {
        probability += exp(-nearest_distances[(i * maximum_knns) + j] /
            (2 * std::pow(nearest_distances[(i * maximum_knns) + k], 0.5))) / scale_factor;
      }

      loocv_score += log(probability);
      mean_band += nearest_distances[(i * maximum_knns) + k];
    }
    NOTIFY("Knn=%i mean_band=%lg score=%lg, dimension=%i", k, mean_band / num_of_points, loocv_score, dimension);
    if (loocv_score > max_loocv_score) {
      max_loocv_score = loocv_score;
      optimum_knns = k;
    }
  }
}
