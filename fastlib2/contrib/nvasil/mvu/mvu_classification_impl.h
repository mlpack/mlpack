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
  grad_tolerance_ = fx_param_double(module_, "grad_tolerance", 1e-3);
  desired_feasibility_error_ = fx_param_double(module_, "desired_feasibility_error", 10);
  new_dimension_=fx_param_int_req(module_, "new_dimension");
  num_of_labeled_ = labeled_data.n_cols();
  num_of_unlabeled_ = unlabeled_data.n_cols();
  num_of_points_=num_of_labeled_+num_of_unlabeled_;
  previous_infeasibility1_=DBL_MAX;
  labeled_offset_=0;
  unlabeled_offset_=num_of_labeled_;
  DEBUG_ASSERT_MSG(labeled_data.n_rows()==unlabeled_data.n_rows(), 
      "Labeled data points don't have the same dimension with unlabeled");
  Matrix  data_points;
  data_points.Init(labeled_data.n_rows(), num_of_labeled_+num_of_unlabeled_);
  data_points.CopyColumnFromMat(labeled_offset_, 0, labeled_data.n_cols(), labeled_data);
  data_points.CopyColumnFromMat(unlabeled_offset_, 0, unlabeled_data.n_cols(), unlabeled_data);

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
  sum_of_nearest_distances_=0;
  for(index_t i=0; i<nearest_distances_.size(); i++) {
   sum_of_nearest_distances_+=math::Pow<1,2>(nearest_distances_[i]);
  }
  NOTIFY("Sum of all nearest distances:%lg", sum_of_nearest_distances_);
  fx_result_double(module_, "sum_of_nearest_distances", sum_of_nearest_distances_);
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
    *error+=fabs(dist_diff);
  }
  *error = *error *100.0/sum_of_nearest_distances_;
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

bool MaxFurthestNeighborsSemiSupervised::IsOptimizationOver(
    Matrix &coordinates, Matrix &gradient, double step) {
  
  ComputeFeasibilityError(coordinates, &infeasibility1_);
  if (infeasibility1_<desired_feasibility_error_ || 
      fabs(infeasibility1_-previous_infeasibility1_)<0.1)  {
    NOTIFY("Optimization is over");
    return true;
  } else {
    previous_infeasibility1_=infeasibility1_;
    return false; 
  }
}

bool MaxFurthestNeighborsSemiSupervised::IsIntermediateStepOver(
    Matrix &coordinates, Matrix &gradient, double step) {
  double norm_gradient=math::Pow<1,2>(la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr()));
  double feasibility_error;
  ComputeFeasibilityError(coordinates, &feasibility_error);

  if (norm_gradient*step < grad_tolerance_ 
      ||  feasibility_error<desired_feasibility_error_) {
    return true;
  }
  return false;

}

void MaxFurthestNeighborsSemiSupervised::GiveInitMatrix(Matrix *init_matrix) {
  init_matrix->Init(new_dimension_, num_of_points_);
  for(index_t i=0; i<num_of_points_; i++) {
    for(index_t j=0; j<new_dimension_; j++) {
      init_matrix->set(j, i, math::Random(0.0, 1.0));
    }
  }
}

index_t  MaxFurthestNeighborsSemiSupervised::num_of_points() {
  return num_of_points_;
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

void MaxFurthestNeighborsSvmSemiSupervised::Init(fx_module *module, 
    Matrix &labeled_data, 
    Matrix &unlabeled_data,
    Matrix &labels) {

  module_=module;
  knns_ = fx_param_int(module_, "knns", 5);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  grad_tolerance_ = fx_param_double(module_, "grad_tolerance", 1e-3);
  desired_feasibility_error_ = fx_param_double(module_, "desired_feasibility_error", 10);
  new_dimension_=fx_param_int_req(module_, "new_dimension");
  regularizer_ = fx_param_double(module_, "regularizer", 0);
  num_of_labeled_ = labeled_data.n_cols();
  num_of_unlabeled_ = unlabeled_data.n_cols();
  num_of_points_=num_of_labeled_+num_of_unlabeled_;
  previous_infeasibility1_=DBL_MAX;
  labeled_offset_=0;
  unlabeled_offset_=num_of_labeled_;
  // get the number of classes
  num_of_classes_ = index_t(*std::max_element(labels.ptr(), 
                     labels.ptr()+labels.n_cols()));
  svm_signs_.Init(num_of_classes_, labels.n_cols());
  anchors_.Init(num_of_classes_);
  for(index_t i=0; i<num_of_classes_; i++) {
    double *p = std::find(labels.ptr(), labels.ptr()+labels.n_cols(), double(i));
    anchors_[i] = ptrdiff_t(p - labels.ptr());
  }
  
  ineq_lagrange_mult_.Init(num_of_classes_ *  num_of_labeled_);
  ineq_lagrange_mult_.SetAll(1);
  for(index_t i=0; i<num_of_classes_; i++) {
    for(index_t j=0; j<labels.n_cols(); j++) {
      if (labels.get(0, j)==labels.get(0, anchors_[i])) {
        svm_signs_.set(i, j ,1.0);
      } else {
        svm_signs_.set(i, j ,-1.0);
      }
    }
  }
  DEBUG_ASSERT_MSG(labeled_data.n_rows()==unlabeled_data.n_rows(), 
      "Labeled data points don't have the same dimension with unlabeled");
  Matrix  data_points;
  data_points.Init(labeled_data.n_rows(), num_of_labeled_+num_of_unlabeled_);
  data_points.CopyColumnFromMat(labeled_offset_, 0, labeled_data.n_cols(), labeled_data);
  if (unlabeled_data.n_cols()!=0) {
    data_points.CopyColumnFromMat(unlabeled_offset_, 0, unlabeled_data.n_cols(), unlabeled_data);
  }
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
  sum_of_nearest_distances_=0;
  for(index_t i=0; i<nearest_distances_.size(); i++) {
   sum_of_nearest_distances_+=math::Pow<1,2>(nearest_distances_[i]);
  }
  NOTIFY("Sum of all nearest distances:%lg", sum_of_nearest_distances_);
  fx_result_double(module_, "sum_of_nearest_distances", sum_of_nearest_distances_);
  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  eq_lagrange_mult_.Init(num_of_nearest_pairs_);
  eq_lagrange_mult_.SetAll(0.0);
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

void MaxFurthestNeighborsSvmSemiSupervised::Init(fx_module *module, 
                                                 Matrix &labels) {

  module_=module;
  knns_ = fx_param_int(module_, "knns", 5);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  grad_tolerance_ = fx_param_double(module_, "grad_tolerance", 1e-3);
  desired_feasibility_error_ = fx_param_double(module_, "desired_feasibility_error", 10);
  new_dimension_=fx_param_int_req(module_, "new_dimension");
  num_of_labeled_ = labels.n_cols();
  num_of_points_=num_of_labeled_+num_of_unlabeled_;
  previous_infeasibility1_=DBL_MAX;
  labeled_offset_=0;
  unlabeled_offset_=num_of_labeled_;
  // get the number of classes
  num_of_classes_ = index_t(*std::max_element(labels.ptr(), labels.ptr()+labels.n_cols()))+1;
  svm_signs_.Init(num_of_classes_, labels.n_cols());
  anchors_.Init(num_of_classes_);
  for(index_t i=0; i<num_of_classes_; i++) {
    double *p = std::find(labels.ptr(), labels.ptr()+labels.n_cols(), double(i));
    anchors_[i] = ptrdiff_t(p - labels.ptr());
  }
  
  ineq_lagrange_mult_.Init(num_of_classes_ *  num_of_labeled_);
  ineq_lagrange_mult_.SetAll(100);
  for(index_t i=0; i<num_of_classes_; i++) {
    for(index_t j=0; j<labels.n_cols(); j++) {
      if (labels.get(0, j)==labels.get(0, anchors_[i])) {
        svm_signs_.set(i, j ,1.0);
      } else {
        svm_signs_.set(i, j ,-1.0);
      }
    }
  }
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
  fclose(fp);
  num_of_unlabeled_ = num_of_points_-num_of_labeled_;  
  num_of_nearest_pairs_=nearest_neighbor_pairs_.size(); 
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


void MaxFurthestNeighborsSvmSemiSupervised::Destruct() {
  allknn_.Destruct();
  allkfn_.Destruct();
  nearest_neighbor_pairs_.Renew();
  nearest_distances_.Renew();
  eq_lagrange_mult_.Destruct();
  ineq_lagrange_mult_.Destruct();
  furthest_neighbor_pairs_.Renew();
  furthest_distances_.Renew();
}

void MaxFurthestNeighborsSvmSemiSupervised::ComputeGradient(Matrix &coordinates, Matrix *gradient) {
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

    la::AddExpert(dimension, -2.0, a_i_r,
        gradient->GetColumnPtr(n1));
    la::AddExpert(dimension, 2.0, a_i_r,
        gradient->GetColumnPtr(n2));
  }
  
  // maximize the margin too
/*  for(index_t i=0; i<num_of_classes_; i++) {
    la::AddExpert(dimension, 2*regularizer_, 
                  coordinates.GetColumnPtr(anchors_[i]),
                  gradient->GetColumnPtr(anchors_[i]));
  }
*/  
  
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
        -2*eq_lagrange_mult_[i]+dist_diff*sigma_,
        a_i_r, 
        gradient->GetColumnPtr(n1));
    la::AddExpert(dimension,
        2*eq_lagrange_mult_[i]-dist_diff*sigma_,
        a_i_r, 
        gradient->GetColumnPtr(n2));
  }
  
  // inequality constraints
  for(index_t i=0; i<num_of_classes_; i++) {
    for(index_t j=0; j<num_of_labeled_; j++) {
      double *p1=coordinates.GetColumnPtr(anchors_[i]);
      double *p2=coordinates.GetColumnPtr(j);
      double dot_prod=la::Dot(dimension,
                              p1,
                              p2);
      double ineq=dot_prod*svm_signs_.get(i, j)-MARGIN;
      if (sigma1_*ineq <= ineq_lagrange_mult_[i*num_of_labeled_+j] ) {
        double factor=(-ineq_lagrange_mult_[i*num_of_labeled_+j]+ sigma1_*ineq)*svm_signs_.get(i, j);
        la::AddExpert(dimension, factor, p2, 
            gradient->GetColumnPtr(anchors_[i]));
        la::AddExpert(dimension, factor, p1, 
            gradient->GetColumnPtr(j));
      }
    }
  }
  // End of computations
}

void MaxFurthestNeighborsSvmSemiSupervised::ComputeObjective(Matrix &coordinates, 
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

// Maximize the margin  
/*  for(index_t i=0; i<num_of_classes_; i++) {
    *objective+=regularizer_ * 
                 la::Dot(dimension, coordinates.GetColumnPtr(anchors_[i]),
                         coordinates.GetColumnPtr(anchors_[i]));
  }
*/  
}

void MaxFurthestNeighborsSvmSemiSupervised::ComputeFeasibilityError(Matrix &coordinates, double *error) {
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
    *error+=fabs(dist_diff);
  }
  *error = *error *100.0/sum_of_nearest_distances_;
  NOTIFY("Feasibility error:%lg", *error);
  double error1=0;
  for(index_t i=0; i<num_of_classes_; i++) {
    for(index_t j=0; j<num_of_labeled_; j++) {
      double dot_prod=la::Dot(dimension,
                              coordinates.GetColumnPtr(anchors_[i]),
                              coordinates.GetColumnPtr(j)) ; 
      if (dot_prod*svm_signs_.get(i, j)<=0) {
        error1+=1;
      }
    } 
  }
  error1=(100.0 * error1)/svm_signs_.n_elements();
  *error=(*error+error1)/2.0;
  NOTIFY("Classification Error:%lg", error1);
}

double MaxFurthestNeighborsSvmSemiSupervised::ComputeLagrangian(Matrix &coordinates) {
  index_t dimension=coordinates.n_rows();
  double lagrangian=0;
  ComputeObjective(coordinates, &lagrangian);
  //Equality constraints
 
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(dimension, point1, point2) 
                           -nearest_distances_[i];
    lagrangian+=dist_diff*dist_diff*sigma_/2.0
        -eq_lagrange_mult_[i]*dist_diff;
  }

  // inequality constraint
  for(index_t i=0; i<num_of_classes_; i++) {
    for(index_t j=0; j<num_of_labeled_; j++) {
      double dot_prod=la::Dot(dimension,
                              coordinates.GetColumnPtr(anchors_[i]),
                              coordinates.GetColumnPtr(j)); 
      double ineq=dot_prod*svm_signs_.get(i, j)-MARGIN;
      if (sigma1_ * ineq  <= ineq_lagrange_mult_[i*num_of_labeled_+j]) {
        lagrangian+=(-ineq_lagrange_mult_[i*num_of_labeled_+j] + sigma1_/2*ineq)*ineq;
      } else {
        lagrangian+=-math::Pow<2,1>(ineq_lagrange_mult_[i*num_of_labeled_+j])/(2*sigma1_);
      }
    }   
  }
  
  return lagrangian;
}

void MaxFurthestNeighborsSvmSemiSupervised::UpdateLagrangeMult(Matrix &coordinates) {
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
  
  for(index_t i=0; i<num_of_classes_; i++) {
    for(index_t j=0; j<num_of_labeled_; j++) {
      double dot_prod=la::Dot(dimension,
                              coordinates.GetColumnPtr(anchors_[i]),
                              coordinates.GetColumnPtr(j));
      double ineq=dot_prod*svm_signs_.get(i, j)-MARGIN; 
      ineq_lagrange_mult_[i*num_of_labeled_+j] = 
          std::max(ineq_lagrange_mult_[i*num_of_labeled_+j]-sigma1_*ineq, 0.0);
    }
  }
}

void MaxFurthestNeighborsSvmSemiSupervised::set_sigma(double sigma) {
  sigma_=sigma;
  sigma1_=sigma_;
}

void MaxFurthestNeighborsSvmSemiSupervised::set_lagrange_mult(double val) {
  eq_lagrange_mult_.SetAll(val);
  ineq_lagrange_mult_.SetAll(val);
}
bool MaxFurthestNeighborsSvmSemiSupervised::IsDiverging(double objective) {
  if (objective < sum_of_furthest_distances_) {
    NOTIFY("objective(%lg) < sum_of_furthest_distances (%lg)", objective,
        sum_of_furthest_distances_);
    return true;
  } else {
    return false;
  }
}

void MaxFurthestNeighborsSvmSemiSupervised::Project(Matrix *coordinates) {
  OptUtils::RemoveMean(coordinates);
}

bool MaxFurthestNeighborsSvmSemiSupervised::IsOptimizationOver(
  Matrix &coordinates, Matrix &gradient, double step) {
  ComputeFeasibilityError(coordinates, &infeasibility1_);
  if (infeasibility1_<desired_feasibility_error_ || 
      fabs(infeasibility1_-previous_infeasibility1_)<0.001)  {
    NOTIFY("Optimization is over");
    return true;
  } else {
    previous_infeasibility1_=infeasibility1_;
    return false; 
  }
}

bool MaxFurthestNeighborsSvmSemiSupervised::IsIntermediateStepOver(
    Matrix &coordinates, Matrix &gradient, double step) {
  double norm_gradient=math::Pow<1,2>(la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr()));
  double feasibility_error;
  ComputeFeasibilityError(coordinates, &feasibility_error);
  if (norm_gradient*step < grad_tolerance_ 
      ||  feasibility_error<desired_feasibility_error_) {
    return true;
  }
  return false;

}

void MaxFurthestNeighborsSvmSemiSupervised::GiveInitMatrix(Matrix *init_matrix) {
  init_matrix->Init(new_dimension_, num_of_points_);
  for(index_t i=0; i<num_of_points_; i++) {
    for(index_t j=0; j<new_dimension_; j++) {
      init_matrix->set(j, i, math::Random(0.0, 1.0));
    }
  }
}

index_t  MaxFurthestNeighborsSvmSemiSupervised::num_of_points() {
  return num_of_points_;
}

void MaxFurthestNeighborsSvmSemiSupervised::anchors(ArrayList<index_t> *anch) {
  for(index_t i=0; i<num_of_classes_; i++) {
    anch->PushBackCopy(anchors_[i]);
  }
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
void MaxFurthestNeighborsSvmSemiSupervised1::Init(fx_module *module, 
    Matrix &labeled_data, 
    Matrix &unlabeled_data,
    Matrix &labels) {

  module_=module;
  knns_ = fx_param_int(module_, "knns", 5);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  grad_tolerance_ = fx_param_double(module_, "grad_tolerance", 1e-3);
  desired_feasibility_error_ = fx_param_double(module_, "desired_feasibility_error", 10);
  new_dimension_=fx_param_int_req(module_, "new_dimension");
  regularizer_ = fx_param_double(module_, "regularizer", 1);
  num_of_labeled_ = labeled_data.n_cols();
  num_of_unlabeled_ = unlabeled_data.n_cols();
  num_of_points_=num_of_labeled_+num_of_unlabeled_;
  previous_infeasibility1_=DBL_MAX;
  labeled_offset_=0;
  unlabeled_offset_=num_of_labeled_;
  // get the number of classes
  num_of_classes_ = index_t(*std::max_element(labels.ptr(), 
                     labels.ptr()+labels.n_cols()));
  svm_signs_.Init(num_of_classes_, labels.n_cols());
  anchors_.Init(num_of_classes_);
  for(index_t i=0; i<num_of_classes_; i++) {
    anchors_[i] = num_of_points_+i;
  }
  
  ineq_lagrange_mult_.Init(num_of_classes_ *  num_of_labeled_);
  ineq_lagrange_mult_.SetAll(1);
  for(index_t i=0; i<num_of_classes_; i++) {
    for(index_t j=0; j<labels.n_cols(); j++) {
      if (labels.get(0, j)==i) {
        svm_signs_.set(i, j ,1.0);
      } else {
        svm_signs_.set(i, j ,-1.0);
      }
    }
  }
  DEBUG_ASSERT_MSG(labeled_data.n_rows()==unlabeled_data.n_rows(), 
      "Labeled data points don't have the same dimension with unlabeled");
  Matrix  data_points;
  data_points.Init(labeled_data.n_rows(), num_of_labeled_+num_of_unlabeled_);
  data_points.CopyColumnFromMat(labeled_offset_, 0, labeled_data.n_cols(), labeled_data);
  if (unlabeled_data.n_cols()!=0) {
    data_points.CopyColumnFromMat(unlabeled_offset_, 0, unlabeled_data.n_cols(), unlabeled_data);
  }
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
  sum_of_nearest_distances_=0;
  for(index_t i=0; i<nearest_distances_.size(); i++) {
   sum_of_nearest_distances_+=math::Pow<1,2>(nearest_distances_[i]);
  }
  NOTIFY("Sum of all nearest distances:%lg", sum_of_nearest_distances_);
  fx_result_double(module_, "sum_of_nearest_distances", sum_of_nearest_distances_);
  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  eq_lagrange_mult_.Init(num_of_nearest_pairs_);
  eq_lagrange_mult_.SetAll(0.0);
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

void MaxFurthestNeighborsSvmSemiSupervised1::Init(fx_module *module, 
                                                 Matrix &labels) {

  module_=module;
  knns_ = fx_param_int(module_, "knns", 5);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  grad_tolerance_ = fx_param_double(module_, "grad_tolerance", 1e-3);
  desired_feasibility_error_ = fx_param_double(module_, "desired_feasibility_error", 10);
  new_dimension_=fx_param_int_req(module_, "new_dimension");
  num_of_labeled_ = labels.n_cols();
  num_of_points_=num_of_labeled_+num_of_unlabeled_;
  previous_infeasibility1_=DBL_MAX;
  labeled_offset_=0;
  unlabeled_offset_=num_of_labeled_;
  // get the number of classes
  num_of_classes_ = index_t(*std::max_element(labels.ptr(), labels.ptr()+labels.n_cols()))+1;
  svm_signs_.Init(num_of_classes_, labels.n_cols());
  anchors_.Init(num_of_classes_);
  for(index_t i=0; i<num_of_classes_; i++) {
    double *p = std::find(labels.ptr(), labels.ptr()+labels.n_cols(), double(i));
    anchors_[i] = ptrdiff_t(p - labels.ptr());
  }
  
  ineq_lagrange_mult_.Init(num_of_classes_ *  num_of_labeled_);
  ineq_lagrange_mult_.SetAll(100);
  for(index_t i=0; i<num_of_classes_; i++) {
    for(index_t j=0; j<labels.n_cols(); j++) {
      if (labels.get(0, j)==labels.get(0, anchors_[i])) {
        svm_signs_.set(i, j ,1.0);
      } else {
        svm_signs_.set(i, j ,-1.0);
      }
    }
  }
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
  fclose(fp);
  num_of_unlabeled_ = num_of_points_-num_of_labeled_;  
  num_of_nearest_pairs_=nearest_neighbor_pairs_.size(); 
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


void MaxFurthestNeighborsSvmSemiSupervised1::Destruct() {
  allknn_.Destruct();
  allkfn_.Destruct();
  nearest_neighbor_pairs_.Renew();
  nearest_distances_.Renew();
  eq_lagrange_mult_.Destruct();
  ineq_lagrange_mult_.Destruct();
  furthest_neighbor_pairs_.Renew();
  furthest_distances_.Renew();
}

void MaxFurthestNeighborsSvmSemiSupervised1::ComputeGradient(Matrix &coordinates, Matrix *gradient) {
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

    la::AddExpert(dimension, -2.0, a_i_r,
        gradient->GetColumnPtr(n1));
    la::AddExpert(dimension, 2.0, a_i_r,
        gradient->GetColumnPtr(n2));
  }
  
  // maximize the margin too
  for(index_t i=0; i<num_of_classes_; i++) {
    la::AddExpert(dimension, 2*regularizer_, 
                  coordinates.GetColumnPtr(anchors_[i]),
                  gradient->GetColumnPtr(anchors_[i]));
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
        -2*eq_lagrange_mult_[i]+dist_diff*sigma_,
        a_i_r, 
        gradient->GetColumnPtr(n1));
    la::AddExpert(dimension,
        2*eq_lagrange_mult_[i]-dist_diff*sigma_,
        a_i_r, 
        gradient->GetColumnPtr(n2));
  }
  
  // inequality constraints
  for(index_t i=0; i<num_of_classes_; i++) {
    for(index_t j=0; j<num_of_labeled_; j++) {
      double *p1=coordinates.GetColumnPtr(anchors_[i]);
      double *p2=coordinates.GetColumnPtr(j);
      double dot_prod=la::Dot(dimension,
                              p1,
                              p2);
      double ineq=dot_prod*svm_signs_.get(i, j)-MARGIN;
      if (sigma1_*ineq <= ineq_lagrange_mult_[i*num_of_labeled_+j] ) {
        double factor=(-ineq_lagrange_mult_[i*num_of_labeled_+j]+ sigma1_*ineq)*svm_signs_.get(i, j);
        la::AddExpert(dimension, factor, p2, 
            gradient->GetColumnPtr(anchors_[i]));
        la::AddExpert(dimension, factor, p1, 
            gradient->GetColumnPtr(j));
      }
    }
  }
  // End of computations
}

void MaxFurthestNeighborsSvmSemiSupervised1::ComputeObjective(Matrix &coordinates, 
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

// Maximize the margin  
  for(index_t i=0; i<num_of_classes_; i++) {
    *objective+=regularizer_ * 
                 la::Dot(dimension, coordinates.GetColumnPtr(anchors_[i]),
                         coordinates.GetColumnPtr(anchors_[i]));
  }
  
}

void MaxFurthestNeighborsSvmSemiSupervised1::ComputeFeasibilityError(Matrix &coordinates, double *error) {
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
    *error+=fabs(dist_diff);
  }
  *error = *error *100.0/sum_of_nearest_distances_;
  NOTIFY("Feasibility error:%lg", *error);
  double error1=0;
  for(index_t i=0; i<num_of_classes_; i++) {
    for(index_t j=0; j<num_of_labeled_; j++) {
      double dot_prod=la::Dot(dimension,
                              coordinates.GetColumnPtr(anchors_[i]),
                              coordinates.GetColumnPtr(j)) ; 
      if (dot_prod*svm_signs_.get(i, j)<=0) {
        error1+=1;
      }
    } 
  }
  error1=(100.0 * error1)/svm_signs_.n_elements();
  *error=(*error+error1)/2.0;
  NOTIFY("Classification Error:%lg", error1);
}

double MaxFurthestNeighborsSvmSemiSupervised1::ComputeLagrangian(Matrix &coordinates) {
  index_t dimension=coordinates.n_rows();
  double lagrangian=0;
  ComputeObjective(coordinates, &lagrangian);
  //Equality constraints
 
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(dimension, point1, point2) 
                           -nearest_distances_[i];
    lagrangian+=dist_diff*dist_diff*sigma_/2.0
        -eq_lagrange_mult_[i]*dist_diff;
  }

  // inequality constraint
  for(index_t i=0; i<num_of_classes_; i++) {
    for(index_t j=0; j<num_of_labeled_; j++) {
      double dot_prod=la::Dot(dimension,
                              coordinates.GetColumnPtr(anchors_[i]),
                              coordinates.GetColumnPtr(j)); 
      double ineq=dot_prod*svm_signs_.get(i, j)-MARGIN;
      if (sigma1_ * ineq  <= ineq_lagrange_mult_[i*num_of_labeled_+j]) {
        lagrangian+=(-ineq_lagrange_mult_[i*num_of_labeled_+j] + sigma1_/2*ineq)*ineq;
      } else {
        lagrangian+=-math::Pow<2,1>(ineq_lagrange_mult_[i*num_of_labeled_+j])/(2*sigma1_);
      }
    }   
  }
  
  return lagrangian;
}

void MaxFurthestNeighborsSvmSemiSupervised1::UpdateLagrangeMult(Matrix &coordinates) {
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
  
  for(index_t i=0; i<num_of_classes_; i++) {
    for(index_t j=0; j<num_of_labeled_; j++) {
      double dot_prod=la::Dot(dimension,
                              coordinates.GetColumnPtr(anchors_[i]),
                              coordinates.GetColumnPtr(j));
      double ineq=dot_prod*svm_signs_.get(i, j)-MARGIN; 
      ineq_lagrange_mult_[i*num_of_labeled_+j] = 
          std::max(ineq_lagrange_mult_[i*num_of_labeled_+j]-sigma1_*ineq, 0.0);
    }
  }
}

void MaxFurthestNeighborsSvmSemiSupervised1::set_sigma(double sigma) {
  sigma_=sigma;
  sigma1_=sigma_;
}

void MaxFurthestNeighborsSvmSemiSupervised1::set_lagrange_mult(double val) {
  eq_lagrange_mult_.SetAll(val);
  ineq_lagrange_mult_.SetAll(val);
}
bool MaxFurthestNeighborsSvmSemiSupervised1::IsDiverging(double objective) {
  if (objective < sum_of_furthest_distances_) {
    NOTIFY("objective(%lg) < sum_of_furthest_distances (%lg)", objective,
        sum_of_furthest_distances_);
    return true;
  } else {
    return false;
  }
}

void MaxFurthestNeighborsSvmSemiSupervised1::Project(Matrix *coordinates) {
  OptUtils::RemoveMean(coordinates);
}

bool MaxFurthestNeighborsSvmSemiSupervised1::IsOptimizationOver(
  Matrix &coordinates, Matrix &gradient, double step) {
  ComputeFeasibilityError(coordinates, &infeasibility1_);
  if (infeasibility1_<desired_feasibility_error_ || 
      fabs(infeasibility1_-previous_infeasibility1_)<0.001)  {
    NOTIFY("Optimization is over");
    return true;
  } else {
    previous_infeasibility1_=infeasibility1_;
    return false; 
  }
}

bool MaxFurthestNeighborsSvmSemiSupervised1::IsIntermediateStepOver(
    Matrix &coordinates, Matrix &gradient, double step) {
  double norm_gradient=math::Pow<1,2>(la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr()));
  double feasibility_error;
  ComputeFeasibilityError(coordinates, &feasibility_error);
  if (norm_gradient*step < grad_tolerance_ 
      ||  feasibility_error<desired_feasibility_error_) {
    return true;
  }
  return false;

}

void MaxFurthestNeighborsSvmSemiSupervised1::GiveInitMatrix(Matrix *init_matrix) {
  init_matrix->Init(new_dimension_, num_of_points_+num_of_classes_);
  for(index_t i=0; i<num_of_points_+num_of_classes_; i++) {
    for(index_t j=0; j<new_dimension_; j++) {
      init_matrix->set(j, i, math::Random(0.0, 1.0));
    }
  }
}

index_t  MaxFurthestNeighborsSvmSemiSupervised1::num_of_points() {
  return num_of_points_;
}

void MaxFurthestNeighborsSvmSemiSupervised1::anchors(ArrayList<index_t> *anch) {
  for(index_t i=0; i<num_of_classes_; i++) {
    anch->PushBackCopy(anchors_[i]);
  }
}
