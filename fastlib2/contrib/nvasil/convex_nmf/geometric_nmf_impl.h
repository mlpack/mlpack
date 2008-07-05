/*
 * =====================================================================================
 * 
 *       Filename:  geometric_nmf_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  07/02/2008 02:41:34 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

void GeometricNmf::Init(fx_module *module, 
			ArrayList<index_t> &rows,
			ArrayList<index_t> &columns,
      ArrayList<double>  &values) {
  module_=module;
  // we need to put the data back to a matrix to build the 
  // tree and etc
  Matrix data_mat;
  data_mat.Init(num_of_rows_, num_of_columns_);
  data_mat.SetAll(0.0);
  index_t count=0;
  for(index_t i=0; i<data_mat.n_rows(); i++) {
    for(index_t j=0; j< data_mat.n_cols(); j++) {
      data_mat.set(i, j, values[count]);
      count++;
    }
  } 
  knns_ = fx_param_int(module_, "knns", 5);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  NOTIFY("Data loaded ...\n");
  NOTIFY("Nearest neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  allknn_.Init(data_mat, leaf_size_, knns_); 
  NOTIFY("Tree built ...\n");
  NOTIFY("Computing neighborhoods ...\n");
  ArrayList<index_t> from_tree_neighbors;
  ArrayList<double>  from_tree_distances;
  allknn_.ComputeNeighbors(&from_tree_neighbors,
                           &from_tree_distances);
  NOTIFY("Neighborhoods computed...\n");
  NOTIFY("Consolidating neighbors...\n");
  MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
      from_tree_distances,
      knns_,
      knns_,
      &nearest_neighbor_pairs_,
      &nearest_distances_,
      &num_of_nearest_pairs_);
  // Now compute the nearest neighbor dot products
  nearest_dot_products_.Init(nearest_distances_.size());
  for(index_t i=0; i<nearest_distances_.size(); i++) {
    index_t p1=nearest_neighbor_pairs_[i].first;
    index_t p2=nearest_neighbor_pairs_[i].second;
    nearest_dot_products_[i]=la::Dot(data_mat.n_rows(),
        data_mat.GetColumnPtr(p1),
        data_mat.GetColumnPtr(p2));
  }
	new_dim_=fx_param_int(module_, "new_dim", 5); 
  desired_duality_gap_=fx_param_double(module_, "desired_duality_gap", 1e-4);
  gradient_tolerance_=fx_param_double(module_, "gradient_tolerance", 1);
  v_accuracy_=fx_param_double(module_, "v_accuracy", 1e-4);
  rows_.InitCopy(rows);
	columns_.InitCopy(columns);
	values_.InitCopy(values);
	num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
	num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
	// It assumes an N x new_dim_ array, where N=num_rows+num_columns
  offset_h_ = num_of_rows_;
  offset_epsilon_=num_of_rows_+num_of_columns_;
  num_of_logs_=values_.size()*new_dim_+nearest_dot_products_.size();
}

void GeometricNmf::ComputeGradient(Matrix &coordinates, 
    Matrix *gradient) {
  gradient->SetAll(0.0);
  double epsilon1=coordinates.get(0, offset_epsilon_);
  double epsilon2=coordinates.get(1, offset_epsilon_);
  double depsilon1=2*epsilon1;
  double depsilon2=2*epsilon2;
  // mathcing dot products and the objective
  for(index_t i=0; i<values_.size(); i++) {
    index_t w_i=rows_[i];
    index_t h_i=columns_[i]+offset_h_;
    double v=values_[i];
    double new_v=0;
    for(index_t j=0; j<new_dim_; j++) {
      double w=coordinates.get(j, w_i);
      double h=coordinates.get(j, h_i);
      new_v+=exp(w+h);
    }
    for(index_t j=0; j<new_dim_; j++) {
      double w=coordinates.get(w_i, j);
      double h=coordinates.get(h_i, j);
      double grad_w=gradient->get(w_i, j);
      double grad_h=gradient->get(h_i, j);    
      double grad=sigma_*exp(w+h)-exp(w+h)/(v+epsilon1-new_v);
      depsilon1=-1/(v+epsilon1-new_v);
      grad_w+=grad;
      grad_h+=grad;
      gradient->set(j, w_i, grad);
      gradient->set(j, h_i, grad);
    }
  }
  // neighborhood constraints
  for(index_t i=0; i<nearest_neighbor_pairs_.size(); i++) {
    index_t w_1=nearest_neighbor_pairs_[i].first;
    index_t w_2=nearest_neighbor_pairs_[i].second;
    double v=nearest_dot_products_[i];
    double new_v=0;
    for(index_t j=0; j<new_dim_; j++) {
      double w1=coordinates.get(w_1, w_1);
      double w2=coordinates.get(w_2, w_2);
      new_v+=exp(w1+w2);
    }
    for(index_t j=0; j<new_dim_; j++) {
      double w1=coordinates.get(w_1, j);
      double w2=coordinates.get(w_2, j);
      double grad_w1=gradient->get(w_1, j);
      double grad_w2=gradient->get(w_2, j);    
      double grad=sigma_*exp(w1+w2)-exp(w1+w2)/(epsilon2+new_v-v);
      depsilon2=-1/(epsilon2+new_v-v);
      grad_w1+=grad;
      grad_w2+=grad;
      gradient->set(j, w_1, grad);
      gradient->set(j, w_2, grad);
    }
  }
  gradient->set(0, offset_epsilon_, depsilon1);
  gradient->set(1, offset_epsilon_, depsilon2);

}

void GeometricNmf::ComputeObjective(Matrix &coordinates, 
                                    double *objective) {
  *objective=0;
  for(index_t i=0; i<values_.size(); i++) {
    index_t w_i=rows_[i];
    index_t h_i=columns_[i]+offset_h_;
    for(index_t j=0; j<new_dim_; j++) {
      double w=coordinates.get(j, w_i);
      double h=coordinates.get(j, h_i);
      *objective+=exp(w+h);
    }
  }
  // add the epsilons
  *objective+=math::Sqr(coordinates.get(0, offset_epsilon_));
  *objective+=math::Sqr(coordinates.get(1, offset_epsilon_));
}

void GeometricNmf::ComputeFeasibilityError(Matrix &coordinates, 
                                           double *error) {
  // return duality gap instead
  *error=num_of_logs_/sigma_;
  double lagrangian=ComputeLagrangian(coordinates);
  double objective;
  ComputeObjective(coordinates, &objective);
  NOTIFY("sum of log barriers:%lg ", lagrangian-sigma_*objective);
}

double GeometricNmf::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian=0;
  // from the objective functions
  ComputeObjective(coordinates, &lagrangian);
  lagrangian*=sigma_;
  double epsilon1=coordinates.get(0, offset_epsilon_);
  double epsilon2=coordinates.get(1, offset_epsilon_);
  // mathcing dot products and the objective
  for(index_t i=0; i<values_.size(); i++) {
    index_t w_i=rows_[i];
    index_t h_i=columns_[i]+offset_h_;
    double v=values_[i];
    double new_v=0;
    for(index_t j=0; j<new_dim_; j++) {
      double w=coordinates.get(j, w_i);
      double h=coordinates.get(j, h_i);
      new_v+=exp(w+h);
    }
    lagrangian+=-log(epsilon1+v-new_v);
  }
  // neighborhood constraints
  for(index_t i=0; i<nearest_neighbor_pairs_.size(); i++) {
    index_t w_1=nearest_neighbor_pairs_[i].first;
    index_t w_2=nearest_neighbor_pairs_[i].second;
    double v=nearest_dot_products_[i];
    double new_v=0;
    for(index_t j=0; j<new_dim_; j++) {
      double w1=coordinates.get(w_1, w_1);
      double w2=coordinates.get(w_2, w_2);
      new_v+=exp(w1+w2);
    }
    lagrangian+=-log(epsilon2+v-new_v);
  }
 
  return lagrangian;
}

void GeometricNmf::UpdateLagrangeMult(Matrix &coordinates) {

}

void GeometricNmf::Project(Matrix *coordinates) {
  //OptUtils::NonNegativeProjection(coordinates);
  double epsilon1=coordinates->get(0, offset_epsilon_);
  double epsilon2=coordinates->get(1, offset_epsilon_);
  if (epsilon1<0) {
    coordinates->set(0, offset_epsilon_, 0);
  }
  if (epsilon2<0) {
    coordinates->set(1, offset_epsilon_, 0);
  }
}

void GeometricNmf::set_sigma(double sigma) {
  sigma_=sigma;
}

void GeometricNmf::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(new_dim_, num_of_rows_+num_of_columns_);
  init_data->SetAll(0.0);
  double epsilon1=*std::max_element(values_.begin(), values_.end());
  double epsilon2=*std::max_element(nearest_dot_products_.begin(), 
      nearest_dot_products_.end());
  init_data->set(0, offset_epsilon_, epsilon1);
  init_data->set(1, offset_epsilon_, epsilon2);
}

bool GeometricNmf::IsDiverging(double objective) {
  return false;
}

bool GeometricNmf::IsOptimizationOver(Matrix &coordinates, 
    Matrix &gradient, double step) {
/*  double norm_gradient = la::Dot(gradient.n_elements(), 
      gradient.ptr(), gradient.ptr());
  //  one of our barriers is zero
  if (norm_gradient>=DBL_MAX) {
    return true;
  }
  if (number_of_cones_/sigma_ < desired_duality_gap_) {
    return true;
  } else {
    return false;
  }
  */
  return true;
}

bool GeometricNmf::IsIntermediateStepOver(Matrix &coordinates, 
    Matrix &gradient, double step) {
  double norm_gradient = la::Dot(gradient.n_elements(), 
      gradient.ptr(), gradient.ptr());
  //NOTIFY("norm_gradient:%lg step:%lg , gradient_tolerance_:%lg", 
  //    norm_gradient, step, gradient_tolerance_);
  if (norm_gradient*step < gradient_tolerance_ || step==0.0) {
    return true;
  } else {
    return false;
  }
}
 
