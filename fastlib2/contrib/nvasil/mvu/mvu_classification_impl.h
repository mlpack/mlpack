/*
 * =====================================================================================
 * 
 *       Filename:  mvu_classification_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  10/08/2008 05:29:23 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

void MaxFurthestNeighborsSemiSupervised::Init(fx_module *module, Matrix &labeled_data, 
      Matrix &unlabeled_data) {
  module_=module;
  knns_ = fx_param_int(module_, "knns", 5);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  num_of_labeled_ = labeled_data.n_cols();
  num_of_unlabeled_ = unlabeled_data.n_cols();
  labeled_offset_=0;
  unlabeled_offset_=num_of_labeled_;
  DEBUG_ASSERT_MSG(labeled_data.n_rows()==unlabeled_data.n_rows(), 
      "Labeled data points don't have the same dimension with unlabeled");
  Matrix  data_points;
  data_points.Init(labeled_data.n_rows(), num_of_labeled_+num_of_unlabeled_);
  data_points.CopyColumnFromMat(labeled_offset_, 0, labeled_data.n_rows(), labeled_data);
  data_points.CopyColumnFromMat(unlabeled_offset_, 0, unlabeled_data.n_rows(), unlabeled_data);

  NOTIFY("Nearest neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  if (knns_==0) {
     allknn_.Init(data_points, leaf_size_, MAX_KNNS); 
  } else {
    allknn_.Init(data_points, leaf_size_, knns_); 
  }
  NOTIFY("Tree built ...\n");
  NOTIFY("Computing neighborhoods ...\n");
  ArrayList<index_t> from_tree_neighbors;
  ArrayList<double>  from_tree_distances;
  allknn_.ComputeNeighbors(&from_tree_neighbors,
                           &from_tree_distances);

  NOTIFY("Neighborhoods computed...\n");
  if (knns_==0) {
    NOTIFY("Auto-tuning the knn...\n" );
    MaxVarianceUtils::EstimateKnns(from_tree_neighbors,
        from_tree_distances,
        MAX_KNNS, 
        data_points.n_cols(),
        data_points.n_rows(),
        &knns_); 
    NOTIFY("Optimum knns is %i", knns_);
    fx_format_result(module_, "optimum_knns", "%i",knns_);
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
        from_tree_distances,
        MAX_KNNS,
        knns_,
        &nearest_neighbor_pairs_,
        &nearest_distances_,
        &num_of_nearest_pairs_);
  } else { 
    NOTIFY("Consolidating neighbors...\n");
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
        from_tree_distances,
        knns_,
        knns_,
        &nearest_neighbor_pairs_,
        &nearest_distances_,
        &num_of_nearest_pairs_);
  }
 
 fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  eq_lagrange_mult_.Init(num_of_nearest_pairs_);
  eq_lagrange_mult_.SetAll(1.0);
  NOTIFY("Furtherst neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  allkfn_.Init(data_points, leaf_size_, 1); 
  NOTIFY("Tree built ...\n");
  NOTIFY("Computing furthest neighborhoods ...\n");
  from_tree_neighbors.Renew();
  from_tree_distances.Renew();
  allkfn_.ComputeNeighbors(&from_tree_neighbors,
                           &from_tree_distances);
  NOTIFY("Furthest Neighbors computed...\n");
  NOTIFY("Consolidating neighbors...\n");
  MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
      from_tree_distances,
      1,
      1,
      &furthest_neighbor_pairs_,
      &furthest_distances_,
      &num_of_furthest_pairs_);
  double max_nearest_distance=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    max_nearest_distance=std::max(nearest_distances_[i], max_nearest_distance);
  }
  sum_of_furthest_distances_=-max_nearest_distance*
      data_points.n_cols()*num_of_furthest_pairs_;
 
  NOTIFY("Lower bound for optimization %lg", sum_of_furthest_distances_);
  fx_format_result(module_, "lower_optimal_bound", "%lg", sum_of_furthest_distances_);
}
/*
void MaxFurthestNeighborsSemiSupervised::Init(fx_module *module) {
  module_=module;
  std::string nearest_neighbor_file=fx_param_str_req(module, 
      "nearest_neighbor_file");
  std::string furthest_neighbor_file=fx_param_str_req(module, 
      "furthest_neighbor_file");
  FILE *fp=fopen(nearest_neighbor_file.c_str(), "r");
  if (fp==NULL) {
    FATAL("Error while opening %s...%s", nearest_neighbor_file.c_str(),
        strerror(errno));
  }
  nearest_neighbor_pairs_.Init();
  nearest_distances_.Init();
  num_of_points_=0;
  while(!feof(fp)) {
    index_t n1, n2;
    double distance;
    fscanf(fp,"%i %i %lg", &n1, &n2, &distance);
    nearest_neighbor_pairs_.PushBackCopy(std::make_pair(n1, n2));
    nearest_distances_.PushBackCopy(distance);
    if (n1>num_of_points_) {
      num_of_points_=n1;
    }
    if (n2>num_of_points_) {
      num_of_points_=n2;
    }
  }
  num_of_points_++;
  num_of_nearest_pairs_=nearest_neighbor_pairs_.size();
  fclose(fp);
  fp=fopen(furthest_neighbor_file.c_str(), "r");
  if (fp==NULL) {
    FATAL("Error while opening %s...%s", furthest_neighbor_file.c_str(),
        strerror(errno));
  }
  furthest_neighbor_pairs_.Init();
  furthest_distances_.Init();
  while(!feof(fp)) {
    index_t n1, n2;
    double distance;
    fscanf(fp,"%i %i %lg", &n1, &n2, &distance);
    furthest_neighbor_pairs_.PushBackCopy(std::make_pair(n1, n2));
    furthest_distances_.PushBackCopy(distance);
  }
  fclose(fp);
  num_of_furthest_pairs_=furthest_neighbor_pairs_.size();
  eq_lagrange_mult_.Init(num_of_nearest_pairs_);
  eq_lagrange_mult_.SetAll(1.0);
  double max_nearest_distance=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    max_nearest_distance=std::max(nearest_distances_[i], max_nearest_distance);
  }
  sum_of_furthest_distances_=-max_nearest_distance*
      num_of_points_*num_of_points_;
 
  NOTIFY("Lower bound for optimization %lg", sum_of_furthest_distances_);
  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  fx_format_result(module_, "lower_optimal_bound", "%lg", sum_of_furthest_distances_);
}
*/
void MaxFurthestNeighborsSemiSupervised::Destruct() {
  allknn_.Destruct();
  allkfn_.Destruct();
  nearest_neighbor_pairs_.Renew();
  nearest_distances_.Renew();
  eq_lagrange_mult_.Destruct();
  furthest_neighbor_pairs_.Renew();
  furthest_distances_.Renew();
}

void MaxFurthestNeighborsSemiSupervised::ComputeGradient(Matrix &coordinates, Matrix *gradient) {
  index_t dimension=coordinates.n_rows();
  gradient->SetAll(0.0);
  // objective
  for(index_t i=0; i<num_of_furthest_pairs_; i++) {
    double a_i_r[dimension];
    index_t n1=furthest_neighbor_pairs_[i].first;
    index_t n2=furthest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    la::SubOverwrite(dimension, point2, point1, a_i_r);

    la::AddExpert(dimension, -1.0, a_i_r,
        gradient->GetColumnPtr(n1));
    la::AddTo(dimension, a_i_r,
        gradient->GetColumnPtr(n2));
  }
  // equality constraints
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    double a_i_r[dimension];
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(dimension, point1, point2) 
                           -nearest_distances_[i];
    la::SubOverwrite(dimension, point2, point1, a_i_r);

    la::AddExpert(dimension,
        -eq_lagrange_mult_[i]+dist_diff*sigma_,
        a_i_r, 
        gradient->GetColumnPtr(n1));
    la::AddExpert(dimension,
        eq_lagrange_mult_[i]-dist_diff*sigma_,
        a_i_r, 
        gradient->GetColumnPtr(n2));
  }
}

void MaxFurthestNeighborsSemiSupervised::ComputeObjective(Matrix &coordinates, 
    double *objective) {
  *objective=0;
  index_t dimension = coordinates.n_rows();
  for(index_t i=0; i<num_of_furthest_pairs_; i++) {
    index_t n1=furthest_neighbor_pairs_[i].first;
    index_t n2=furthest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double diff = la::DistanceSqEuclidean(dimension, point1, point2);
    *objective-=diff;   
  }
}

void MaxFurthestNeighborsSemiSupervised::ComputeFeasibilityError(Matrix &coordinates, double *error) {
  index_t dimension=coordinates.n_rows();
  *error=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(dimension, 
                                               point1, point2) 
                           -nearest_distances_[i];
    *error+=dist_diff*dist_diff;
  }
}

double MaxFurthestNeighborsSemiSupervised::ComputeLagrangian(Matrix &coordinates) {
  index_t dimension=coordinates.n_rows();
  double lagrangian=0;
  ComputeObjective(coordinates, &lagrangian);
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(dimension, point1, point2) 
                           -nearest_distances_[i];
    lagrangian+=dist_diff*dist_diff*sigma_/2
        -eq_lagrange_mult_[i]*dist_diff;
  }
  return lagrangian;
}

void MaxFurthestNeighborsSemiSupervised::UpdateLagrangeMult(Matrix &coordinates) {
  index_t dimension=coordinates.n_rows();
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff =la::DistanceSqEuclidean(dimension, point1, point2) 
                            -nearest_distances_[i];
    eq_lagrange_mult_[i]-=sigma_*dist_diff;
  }
}

void MaxFurthestNeighborsSemiSupervised::set_sigma(double sigma) {
  sigma_=sigma;
}

void MaxFurthestNeighborsSemiSupervised::set_lagrange_mult(double val) {
  eq_lagrange_mult_.SetAll(val);
}
bool MaxFurthestNeighborsSemiSupervised::IsDiverging(double objective) {
  if (objective < sum_of_furthest_distances_) {
    NOTIFY("objective(%lg) < sum_of_furthest_distances (%lg)", objective,
        sum_of_furthest_distances_);
    return true;
  } else {
    return false;
  }
}

void MaxFurthestNeighborsSemiSupervised::Project(Matrix *coordinates) {
  OptUtils::RemoveMean(coordinates);
}

index_t  MaxFurthestNeighborsSemiSupervised::num_of_points() {
  return num_of_points_;
}


