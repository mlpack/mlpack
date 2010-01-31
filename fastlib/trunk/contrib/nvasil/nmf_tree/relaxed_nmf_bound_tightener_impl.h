/*
 * =====================================================================================
 * 
 *       Filename:  relaxed_nmf_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  08/19/2008 10:11:06 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifdef  RELAXED_NMF_BOUND_TIGHTENER_H_
#ifndef RELAXED_NMF_BOUND_IMPL_TIGHTENER_H_
#define RELAXED_NMF_BOUND_IMPL_TIGHTENER_H_

void RelaxedNmfBoundTightener::Init(fx_module *module,
                      ArrayList<index_t> &rows,
                      ArrayList<index_t> &columns,
                      ArrayList<double> &values,
                      Matrix &x_lower_bound, 
                      Matrix &x_upper_bound,  
                      index_t opt_var_row,
                      index_t opt_var_column,
                      index_t opt_var_sign,
                      double function_upper_bound) {
  module_=module;
  rows_.InitAlias(rows);
  num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
  columns_.InitAlias(columns);
  num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
  w_offset_=num_of_rows_;
  h_offset_=0;
  values_.InitAlias(values);
  values_sq_norm_=la::Dot(values_.size(), values_.begin(), values_.begin());
  x_lower_bound_.Copy(x_lower_bound);
  x_upper_bound_.Copy(x_upper_bound);
  soft_lower_bound_=values_sq_norm_;
  new_dimension_=fx_param_int(module_, "new_dimension",  5);
  grad_tolerance_=fx_param_double(module_, "grad_tolerance", 0.01);
  desired_duality_gap_=fx_param_double(module_, "duality_gap", 0.001);
  function_upper_bound_=function_upper_bound; 
  opt_var_row_=opt_var_row;
  opt_var_column_=opt_var_column;
  opt_var_sign_=opt_var_sign;
  // Generate the linear terms
  // and compute the soft lower bound
  a_linear_term_.Init(new_dimension_*values_.size());
  b_linear_term_.Init(new_dimension_*values_.size());
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row+w_offset_;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dimension_; j++) {
      double y_lower=x_lower_bound_.get(j, w) + x_lower_bound_.get(j, h);
      double y_upper=x_upper_bound_.get(j, w) + x_upper_bound_.get(j, h);      
      a_linear_term_[new_dimension_*i+j]=
          (y_upper*exp(y_lower)-y_lower*exp(y_upper))/(y_upper-y_lower);
      b_linear_term_[new_dimension_*i+j]=(exp(y_upper)-exp(y_lower))/
          (y_upper-y_lower);
      DEBUG_ASSERT(b_linear_term_[new_dimension_*i+j]>=0);
      convex_part+=exp(y_lower);
      soft_lower_bound_+= -2*values_[i]*(
                            a_linear_term_[new_dimension_*i+j]
                            +b_linear_term_[new_dimension_*i+j]*y_upper);
    }
    soft_lower_bound_+=convex_part*convex_part;
  }
}

void RelaxedNmfBoundTightener::SetOptVarRowColumn(index_t row, index_t column) {
  opt_var_row_=row;
  opt_var_column_=column;
}

void RelaxedNmfBoundTightener::SetOptVarSign(double sign) {
  opt_var_sign_=sign;
}

void RelaxedNmfBoundTightener::Destruct() {
  num_of_rows_=-1;;
  num_of_columns_=-1;
  new_dimension_=-1;
  rows_.Renew();
  columns_.Renew();
  values_.Renew();
  a_linear_term_.Destruct();
  b_linear_term_.Destruct();
  soft_lower_bound_=-DBL_MAX;
  values_sq_norm_=-DBL_MAX;
} 


void RelaxedNmfBoundTightener::ComputeGradient(Matrix &coordinates, Matrix *gradient) {
  fx_timer_start(NULL, "gradient");
  gradient->SetAll(0.0);
 
  double norm_error=values_sq_norm_;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row+w_offset_;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dimension_; j++) {
      norm_error+=-2*values_[i]*(a_linear_term_[new_dimension_*i+j]
                                +b_linear_term_[new_dimension_*i+j]
                                 *(coordinates.get(j, w)+coordinates.get(j, h)));

      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    norm_error+=convex_part*convex_part;
  } 
  DEBUG_ASSERT_MSG(norm_error <= function_upper_bound_, 
                   "Something is wrong, solution out of the interior!"); 
  
  
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row+w_offset_;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dimension_; j++) {
      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    for(index_t j=0; j<new_dimension_; j++) {
      double grad=2*convex_part*exp(coordinates.get(j, w)
                                    +coordinates.get(j, h))
                      -2*values_[i]*b_linear_term_[new_dimension_*i+j];
      gradient->set(j, w, gradient->get(j, w)+grad);
      gradient->set(j, h, gradient->get(j, h)+grad);
    }
  }

 
  la::Scale(1.0/(function_upper_bound_-norm_error), gradient);
  // gradient from the objective
  // only for the particular variable that we are optimizing the bound
  gradient->set(opt_var_row_, opt_var_column_, 
      gradient->get(opt_var_row_, opt_var_column_)+sigma_*opt_var_sign_);
 
  fx_timer_stop(NULL, "gradient");
}

void RelaxedNmfBoundTightener::ComputeObjective(Matrix &coordinates, double *objective) {
  fx_timer_start(NULL, "objective");
  *objective=opt_var_sign_*coordinates.get(opt_var_row_, opt_var_column_);
  fx_timer_stop(NULL, "objective");
}

void RelaxedNmfBoundTightener::ComputeNonRelaxedObjective(Matrix &coordinates, 
                                            double *objective) {
  fx_timer_start(NULL, "non_relaxed_objective");
  *objective=values_sq_norm_;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row+w_offset_;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dimension_; j++) {
      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    *objective+=convex_part*(convex_part-2*values_[i]);
  } 
  fx_timer_stop(NULL, "non_relaxed_objective");
}


void RelaxedNmfBoundTightener::ComputeFeasibilityError(Matrix &coordinates, 
                                         double *error) {
  *error=0;
}

double RelaxedNmfBoundTightener::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian;
  ComputeObjective(coordinates, &lagrangian);
  lagrangian*=sigma_;
  double norm_error=values_sq_norm_;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row+w_offset_;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dimension_; j++) {
      norm_error+=-2*values_[i]*(a_linear_term_[new_dimension_*i+j]
                                +b_linear_term_[new_dimension_*i+j]
                                 *(coordinates.get(j, w)+coordinates.get(j, h)));

      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    norm_error+=convex_part*convex_part;
  } 
  if (norm_error >= function_upper_bound_) {
    return DBL_MAX;
  }
  lagrangian+=-log(function_upper_bound_-norm_error);
  return lagrangian;
}

void RelaxedNmfBoundTightener::UpdateLagrangeMult(Matrix &coordinates) {
  
}

void RelaxedNmfBoundTightener::Project(Matrix *coordinates) {
  fx_timer_start(NULL, "project");
  for(index_t i=0; i<x_lower_bound_.n_rows(); i++) {
    for(index_t j=0; j<x_lower_bound_.n_cols(); j++) {
      if (coordinates->get(i, j) < x_lower_bound_.get(i, j)) {
        coordinates->set(i, j, x_lower_bound_.get(i, j));
      } else {
        if (coordinates->get(i, j)> x_upper_bound_.get(i, j)) {
          coordinates->set(i, j, x_upper_bound_.get(i, j));
        }
      }
    }
  }
  fx_timer_stop(NULL, "project");
}

void RelaxedNmfBoundTightener::set_sigma(double sigma) {
  sigma_=sigma;
}

void RelaxedNmfBoundTightener::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(new_dimension_, num_of_rows_ + num_of_columns_);
  for(index_t i=0; i<init_data->n_rows(); i++) {
    for(index_t j=0; j<init_data->n_cols(); j++) {
      init_data->set(i, j, 
          (x_lower_bound_.get(i, j) + x_upper_bound_.get(i, j))/2);
    }
  }
}

bool RelaxedNmfBoundTightener::IsDiverging(double objective) {
  return false;
} 

bool RelaxedNmfBoundTightener::IsOptimizationOver(Matrix &coordinates, 
                                    Matrix &gradient, double step) {

  index_t num_of_constraints=1;
  if (num_of_constraints/sigma_ < desired_duality_gap_) {
    return true;
  } else {
    return false;
  }
 
}
/*  double objective;
  ComputeObjective(coordinates, &objective);
  if (fabs(objective-previous_objective_)/objective<0.01) {
    previous_objective_=objective;
    return true;
  } else  {
     previous_objective_=objective;
     return false;
   
  }
*/  

  


bool RelaxedNmfBoundTightener::IsIntermediateStepOver(Matrix &coordinates, 
                                        Matrix &gradient, 
                                        double step) {
  double norm_gradient=la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr());
  if (norm_gradient*step < grad_tolerance_) {
    return true;
  }
  return false;


/*
  double norm_gradient=la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr());
  if (norm_gradient*step < grad_tolerance_) {
    return true;
  }
  return false;
*/
}

double RelaxedNmfBoundTightener::GetSoftLowerBound() {
  return soft_lower_bound_;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
void RelaxedNmfIsometricBoundTightener::Init(fx_module *module,
                      ArrayList<index_t> &rows,
                      ArrayList<index_t> &columns,
                      ArrayList<double> &values,
                      Matrix &x_lower_bound, 
                      Matrix &x_upper_bound,  
                      index_t opt_var_row,
                      index_t opt_var_column,
                      index_t opt_var_sign,
                      double function_upper_bound) {
  module_=module;
  rows_.InitAlias(rows);
  num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
  columns_.InitAlias(columns);
  num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
  w_offset_=num_of_rows_;
  h_offset_=0;
  values_.InitAlias(values);
  values_sq_norm_=la::Dot(values_.size(), values_.begin(), values_.begin());
  x_lower_bound_.Copy(x_lower_bound);
  x_upper_bound_.Copy(x_upper_bound);
  soft_lower_bound_=values_sq_norm_;
  new_dimension_=fx_param_int(module_, "new_dimension",  5);
  grad_tolerance_=fx_param_double(module_, "grad_tolerance", 0.01);
  desired_duality_gap_=fx_param_double(module_, "duality_gap", 0.001);
  function_upper_bound_=function_upper_bound; 
  opt_var_row_=opt_var_row;
  opt_var_column_=opt_var_column;
  opt_var_sign_=opt_var_sign;
  // Generate the linear terms
  // and compute the soft lower bound
  objective_a_linear_term_.Init(new_dimension_*values_.size());
  objective_b_linear_term_.Init(new_dimension_*values_.size());
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=col+w_offset_;
    index_t h=row+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dimension_; j++) {
      double y_lower=x_lower_bound_.get(j, w) + x_lower_bound_.get(j, h);
      double y_upper=x_upper_bound_.get(j, w) + x_upper_bound_.get(j, h);      
      objective_a_linear_term_[new_dimension_*i+j]=
          (y_upper*exp(y_lower)-y_lower*exp(y_upper))/(y_upper-y_lower);
      objective_b_linear_term_[new_dimension_*i+j]=(exp(y_upper)-exp(y_lower))/
          (y_upper-y_lower);
      DEBUG_ASSERT(objective_b_linear_term_[new_dimension_*i+j]>=0);
      convex_part+=exp(y_lower);
      soft_lower_bound_+= -2*values_[i]*(
                            objective_a_linear_term_[new_dimension_*i+j]
                            +objective_b_linear_term_[new_dimension_*i+j]*y_upper);
    }
    soft_lower_bound_+=convex_part*convex_part;
  }
  // Compute the relaxations for the nearest neighbor distance constraints
    // we need to put the data back to a matrix to build the 
    // tree and etc
  Matrix data_mat;
  data_mat.Init(num_of_rows_, num_of_columns_);
  data_mat.SetAll(0.0);

  for(index_t i=0; i<rows_.size(); i++) {
    data_mat.set(rows_[i], columns_[i], values_[i]);
  }
  // Initializations for the local isometries
  index_t knns = fx_param_int(module_, "knns", 3);
  index_t leaf_size = fx_param_int(module_, "leaf_size", 20);
  NOTIFY("Data loaded ...\n");
  NOTIFY("Nearest neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  allknn_.Init(data_mat, leaf_size, knns); 
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
       knns,
       knns,
       &nearest_neighbor_pairs_,
       &nearest_distances_,
       &num_of_nearest_pairs_);
  is_infeasible_=false; 
  constraint_a_linear_term_.Init(new_dimension_*num_of_nearest_pairs_);
  constraint_b_linear_term_.Init(new_dimension_*num_of_nearest_pairs_);
  for(index_t i=0; i<nearest_neighbor_pairs_.size(); i++) {
    double soft_lower_bound=0.0;
    for(index_t j=0; j<new_dimension_; j++) {
      index_t n1 = w_offset_ + nearest_neighbor_pairs_[i].first;
      index_t n2 = w_offset_ + nearest_neighbor_pairs_[i].second; 
      double y_lower=x_lower_bound_.get(j, n1) + x_lower_bound_.get(j, n2);
      double y_upper=x_upper_bound_.get(j, n1) + x_upper_bound_.get(j, n2);      
      constraint_a_linear_term_[new_dimension_*i+j]=
          (y_upper*exp(y_lower)-y_lower*exp(y_upper))/(y_upper-y_lower);
      constraint_b_linear_term_[new_dimension_*i+j]=(exp(y_upper)-exp(y_lower))/
          (y_upper-y_lower);
      DEBUG_ASSERT(constraint_b_linear_term_[new_dimension_*i+j]>=0);
      soft_lower_bound+= exp(2*x_lower_bound_.get(j, n1)) 
                         +exp(2*x_lower_bound_.get(j, n2)); 
                         -2*(constraint_a_linear_term_[new_dimension_*i+j]
                             +constraint_b_linear_term_[new_dimension_*i+j]*y_upper);
                            
    }
     if (soft_lower_bound_-nearest_distances_[i]>0) {
       is_infeasible_=true;
       break;
     }
  }  
}

void RelaxedNmfIsometricBoundTightener::SetOptVarRowColumn(index_t row, index_t column) {
  opt_var_row_=row;
  opt_var_column_=column;
}

void RelaxedNmfIsometricBoundTightener::SetOptVarSign(double sign) {
  opt_var_sign_=sign;
}

void RelaxedNmfIsometricBoundTightener::Destruct() {
  num_of_rows_=-1;;
  num_of_columns_=-1;
  new_dimension_=-1;
  rows_.Renew();
  columns_.Renew();
  values_.Renew();
  objective_a_linear_term_.Destruct();
  objective_b_linear_term_.Destruct();
  constraint_a_linear_term_.Destruct();
  constraint_b_linear_term_.Destruct();
  soft_lower_bound_=-DBL_MAX;
  values_sq_norm_=-DBL_MAX;
  allknn_.Destruct();
} 


void RelaxedNmfIsometricBoundTightener::ComputeGradient(Matrix &coordinates, Matrix *gradient) {
  fx_timer_start(NULL, "gradient");
  gradient->SetAll(0.0);
 
  double norm_error=values_sq_norm_;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=col+w_offset_;
    index_t h=row+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dimension_; j++) {
      norm_error+=-2*values_[i]*(objective_a_linear_term_[new_dimension_*i+j]
                                +objective_b_linear_term_[new_dimension_*i+j]
                                 *(coordinates.get(j, w)+coordinates.get(j, h)));

      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    norm_error+=convex_part*convex_part;
  } 
  DEBUG_ASSERT_MSG(norm_error <= function_upper_bound_, 
                   "Something is wrong, solution out of the interior!"); 
  
  
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=col+w_offset_;
    index_t h=row+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dimension_; j++) {
      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    for(index_t j=0; j<new_dimension_; j++) {
      double grad=2*convex_part*exp(coordinates.get(j, w)
                                    +coordinates.get(j, h))
                      -2*values_[i]*objective_b_linear_term_[new_dimension_*i+j];
      gradient->set(j, w, gradient->get(j, w)+grad);
      gradient->set(j, h, gradient->get(j, h)+grad);
    }
  }

 
  la::Scale(1.0/(function_upper_bound_-norm_error), gradient);
  // gradient from the distance constraints
  for(index_t i=0; i<nearest_neighbor_pairs_.size(); i++) {
    index_t n1 = w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2 = w_offset_ + nearest_neighbor_pairs_[i].second; 
    double distance=0;
    for(index_t j=0; j<new_dimension_; j++) {
      distance+=exp(2*coordinates.get(j, n1))+exp(2*coordinates.get(j, n2));
      distance+=-2*(constraint_a_linear_term_[new_dimension_*i+j]
                    +constraint_b_linear_term_[new_dimension_*i+j]
                     *(coordinates.get(j, n1)+coordinates.get(j, n2)));
    }
    double denominator=nearest_distances_[i]-distance;
    DEBUG_ASSERT_MSG(denominator>0,
       "Something is wrong, solution out of the interior");
    for(index_t j=0; j<new_dimension_; j++){
      double nominator=-2*constraint_b_linear_term_[new_dimension_*i+j];
      gradient->set(j, n1, gradient->get(j, n1) 
          +(nominator+2*exp(2*coordinates.get(j, n1)))/denominator);
      gradient->set(j, n2, gradient->get(j, n2) 
          +(nominator+2*exp(2*coordinates.get(j, n2)))/denominator);

    }    
  }

  // gradient from the objective
  // only for the particular variable that we are optimizing the bound
  gradient->set(opt_var_row_, opt_var_column_, 
      gradient->get(opt_var_row_, opt_var_column_)+sigma_*opt_var_sign_);

 
  
  fx_timer_stop(NULL, "gradient");
}

void RelaxedNmfIsometricBoundTightener::ComputeObjective(Matrix &coordinates, double *objective) {
  fx_timer_start(NULL, "objective");
  *objective=opt_var_sign_*coordinates.get(opt_var_row_, opt_var_column_);
  fx_timer_stop(NULL, "objective");
}

void RelaxedNmfIsometricBoundTightener::ComputeNonRelaxedObjective(Matrix &coordinates, 
                                            double *objective) {
  fx_timer_start(NULL, "non_relaxed_objective");
  *objective=values_sq_norm_;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=col+w_offset_;
    index_t h=row+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dimension_; j++) {
      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    *objective+=convex_part*(convex_part-2*values_[i]);
  } 
  fx_timer_stop(NULL, "non_relaxed_objective");
}


void RelaxedNmfIsometricBoundTightener::ComputeFeasibilityError(Matrix &coordinates, 
                                         double *error) {
  *error=0;
}

double RelaxedNmfIsometricBoundTightener::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian;
  ComputeObjective(coordinates, &lagrangian);
  lagrangian*=sigma_;
  double norm_error=values_sq_norm_;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=col+w_offset_;
    index_t h=row+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dimension_; j++) {
      norm_error+=-2*values_[i]*(objective_a_linear_term_[new_dimension_*i+j]
                                +objective_b_linear_term_[new_dimension_*i+j]
                                 *(coordinates.get(j, w)+coordinates.get(j, h)));

      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    norm_error+=convex_part*convex_part;
  } 
  if (norm_error >= function_upper_bound_) {
    return DBL_MAX;
  }
  lagrangian+=-log(function_upper_bound_-norm_error);
  double prod=1.0;
  for(index_t i=0; i<nearest_neighbor_pairs_.size(); i++) {
    index_t n1 = w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2 = w_offset_ + nearest_neighbor_pairs_[i].second; 
    double distance=0;
    for(index_t j=0; j<new_dimension_; j++) {
      distance+=exp(2*coordinates.get(j, n1))+exp(2*coordinates.get(j, n2));
      distance+=-2*(constraint_a_linear_term_[new_dimension_*i+j]
                    +constraint_b_linear_term_[new_dimension_*i+j]
                     *(coordinates.get(j, n1)+coordinates.get(j, n2)));
    }
    double denominator=nearest_distances_[i]-distance;
    if (denominator<=0) {
      return DBL_MAX;
    }
    prod*=denominator;
    if (unlikely(prod<=1e-30 || prod >=1e+30)) {
      lagrangian+=-log(prod);
      prod=1.0;
    }
  }
  lagrangian+=-log(prod);
  return lagrangian;
}

void RelaxedNmfIsometricBoundTightener::UpdateLagrangeMult(Matrix &coordinates) {
  
}

void RelaxedNmfIsometricBoundTightener::Project(Matrix *coordinates) {
  fx_timer_start(NULL, "project");
  for(index_t i=0; i<x_lower_bound_.n_rows(); i++) {
    for(index_t j=0; j<x_lower_bound_.n_cols(); j++) {
      if (coordinates->get(i, j) < x_lower_bound_.get(i, j)) {
        coordinates->set(i, j, x_lower_bound_.get(i, j));
      } else {
        if (coordinates->get(i, j)> x_upper_bound_.get(i, j)) {
          coordinates->set(i, j, x_upper_bound_.get(i, j));
        }
      }
    }
  }
  fx_timer_stop(NULL, "project");
}

void RelaxedNmfIsometricBoundTightener::set_sigma(double sigma) {
  sigma_=sigma;
}

void RelaxedNmfIsometricBoundTightener::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(new_dimension_, num_of_rows_ + num_of_columns_);
  for(index_t i=0; i<init_data->n_rows(); i++) {
    for(index_t j=0; j<init_data->n_cols(); j++) {
      init_data->set(i, j, 
          (x_lower_bound_.get(i, j) + x_upper_bound_.get(i, j))/2);
    }
  }
}

bool RelaxedNmfIsometricBoundTightener::IsDiverging(double objective) {
  return false;
} 

bool RelaxedNmfIsometricBoundTightener::IsOptimizationOver(Matrix &coordinates, 
                                    Matrix &gradient, double step) {

  index_t num_of_constraints=1+num_of_nearest_pairs_;
  if (num_of_constraints/sigma_ < desired_duality_gap_) {
    return true;
  } else {
    return false;
  }
 
}
/*  double objective;
  ComputeObjective(coordinates, &objective);
  if (fabs(objective-previous_objective_)/objective<0.01) {
    previous_objective_=objective;
    return true;
  } else  {
     previous_objective_=objective;
     return false;
   
  }
*/  

  


bool RelaxedNmfIsometricBoundTightener::IsIntermediateStepOver(Matrix &coordinates, 
                                        Matrix &gradient, 
                                        double step) {
  double norm_gradient=math::Pow<1,2>(la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr()));
  if (norm_gradient*step < grad_tolerance_) {
    return true;
  }
  return false;


/*
  double norm_gradient=la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr());
  if (norm_gradient*step < grad_tolerance_) {
    return true;
  }
  return false;
*/
}

double RelaxedNmfIsometricBoundTightener::GetSoftLowerBound() {
  return soft_lower_bound_;
}

bool RelaxedNmfIsometricBoundTightener::IsInfeasible() {
  return is_infeasible_;
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
void RelaxedNmfIsometricBoxTightener::Init(fx_module *module,
                      ArrayList<index_t> &rows,
                      ArrayList<index_t> &columns,
                      ArrayList<double> &values,
                      Vector &x_lower_bound, 
                      Vector &x_upper_bound,  
                      index_t opt_var_sign,
                      double function_upper_bound) {
  module_=module;
  rows_.InitAlias(rows);
  num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
  columns_.InitAlias(columns);
  num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
  w_offset_=num_of_rows_;
  h_offset_=0;
  values_.InitAlias(values);
  values_sq_norm_=la::Dot(values_.size(), values_.begin(), values_.begin());
  x_lower_bound_.Copy(x_lower_bound);
  x_upper_bound_.Copy(x_upper_bound);
  soft_lower_bound_=values_sq_norm_;
  new_dimension_=fx_param_int(module_, "new_dimension",  5);
  grad_tolerance_=fx_param_double(module_, "grad_tolerance", 0.01);
  desired_duality_gap_=fx_param_double(module_, "duality_gap", 0.001);
  function_upper_bound_=function_upper_bound; 
  opt_var_sign_=opt_var_sign;
  // Generate the linear terms
  // and compute the soft lower bound
  objective_a_linear_term_.Init(new_dimension_*values_.size());
  objective_b_linear_term_.Init(new_dimension_*values_.size());
  for(index_t i=0; i<values_.size(); i++) {
//    index_t row=rows_[i];
//    index_t col=columns_[i];
//    index_t w=col+w_offset_;
//    index_t h=row+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dimension_; j++) {
      double y_lower=x_lower_bound_[j] + x_lower_bound_[j];
      double y_upper=x_upper_bound_[j] + x_upper_bound_[j];      
      objective_a_linear_term_[new_dimension_*i+j]=
          (y_upper*exp(y_lower)-y_lower*exp(y_upper))/(y_upper-y_lower);
      objective_b_linear_term_[new_dimension_*i+j]=(exp(y_upper)-exp(y_lower))/
          (y_upper-y_lower);
      DEBUG_ASSERT(objective_b_linear_term_[new_dimension_*i+j]>=0);
      convex_part+=exp(y_lower);
      soft_lower_bound_+= -2*values_[i]*(
                            objective_a_linear_term_[new_dimension_*i+j]
                            +objective_b_linear_term_[new_dimension_*i+j]*y_upper);
    }
    soft_lower_bound_+=convex_part*convex_part;
  }
  // Compute the relaxations for the nearest neighbor distance constraints
    // we need to put the data back to a matrix to build the 
    // tree and etc
  Matrix data_mat;
  data_mat.Init(num_of_rows_, num_of_columns_);
  data_mat.SetAll(0.0);

  for(index_t i=0; i<rows_.size(); i++) {
    data_mat.set(rows_[i], columns_[i], values_[i]);
  }
  // Initializations for the local isometries
  index_t knns = fx_param_int(module_, "knns", 3);
  index_t leaf_size = fx_param_int(module_, "leaf_size", 20);
  NOTIFY("Data loaded ...\n");
  NOTIFY("Nearest neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  allknn_.Init(data_mat, leaf_size, knns); 
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
       knns,
       knns,
       &nearest_neighbor_pairs_,
       &nearest_distances_,
       &num_of_nearest_pairs_);
  is_infeasible_=false; 
  constraint_a_linear_term_.Init(new_dimension_*num_of_nearest_pairs_);
  constraint_b_linear_term_.Init(new_dimension_*num_of_nearest_pairs_);
  for(index_t i=0; i<nearest_neighbor_pairs_.size(); i++) {
    double soft_lower_bound=0.0;
    for(index_t j=0; j<new_dimension_; j++) {
//      index_t n1 = w_offset_ + nearest_neighbor_pairs_[i].first;
//      index_t n2 = w_offset_ + nearest_neighbor_pairs_[i].second; 
      double y_lower=x_lower_bound_[j] + x_lower_bound_[j];
      double y_upper=x_upper_bound_[j] + x_upper_bound_[j];      
      constraint_a_linear_term_[new_dimension_*i+j]=
          (y_upper*exp(y_lower)-y_lower*exp(y_upper))/(y_upper-y_lower);
      constraint_b_linear_term_[new_dimension_*i+j]=(exp(y_upper)-exp(y_lower))/
          (y_upper-y_lower);
      DEBUG_ASSERT(constraint_b_linear_term_[new_dimension_*i+j]>=0);
      soft_lower_bound+= exp(2*x_lower_bound_.get(j)) 
                         +exp(2*x_lower_bound_.get(j)); 
                         -2*(constraint_a_linear_term_[new_dimension_*i+j]
                             +constraint_b_linear_term_[new_dimension_*i+j]*y_upper);
                            
    }
     if (soft_lower_bound_-nearest_distances_[i]>0) {
       is_infeasible_=true;
       break;
     }
  }  
}

void RelaxedNmfIsometricBoxTightener::SetOptVarSign(double sign) {
  opt_var_sign_=sign;
}

void RelaxedNmfIsometricBoxTightener::Destruct() {
  num_of_rows_=-1;;
  num_of_columns_=-1;
  new_dimension_=-1;
  rows_.Renew();
  columns_.Renew();
  values_.Renew();
  objective_a_linear_term_.Destruct();
  objective_b_linear_term_.Destruct();
  constraint_a_linear_term_.Destruct();
  constraint_b_linear_term_.Destruct();
  soft_lower_bound_=-DBL_MAX;
  values_sq_norm_=-DBL_MAX;
  allknn_.Destruct();
} 


void RelaxedNmfIsometricBoxTightener::ComputeGradient(Matrix &coordinates, Matrix *gradient) {
  fx_timer_start(NULL, "gradient");
  gradient->SetAll(0.0);
 
  double norm_error=values_sq_norm_;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=col+w_offset_;
    index_t h=row+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dimension_; j++) {
      norm_error+=-2*values_[i]*(objective_a_linear_term_[new_dimension_*i+j]
                                +objective_b_linear_term_[new_dimension_*i+j]
                                 *(coordinates.get(j, w)+coordinates.get(j, h)));

      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    norm_error+=convex_part*convex_part;
  } 
  DEBUG_ASSERT_MSG(norm_error <= function_upper_bound_, 
                   "Something is wrong, solution out of the interior!"); 
  
  
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=col+w_offset_;
    index_t h=row+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dimension_; j++) {
      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    for(index_t j=0; j<new_dimension_; j++) {
      double grad=2*convex_part*exp(coordinates.get(j, w)
                                    +coordinates.get(j, h))
                      -2*values_[i]*objective_b_linear_term_[new_dimension_*i+j];
      gradient->set(j, w, gradient->get(j, w)+grad);
      gradient->set(j, h, gradient->get(j, h)+grad);
    }
  }

 
  la::Scale(1.0/(function_upper_bound_-norm_error), gradient);
  // gradient from the distance constraints
  for(index_t i=0; i<nearest_neighbor_pairs_.size(); i++) {
    index_t n1 = w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2 = w_offset_ + nearest_neighbor_pairs_[i].second; 
    double distance=0;
    for(index_t j=0; j<new_dimension_; j++) {
      distance+=exp(2*coordinates.get(j, n1))+exp(2*coordinates.get(j, n2));
      distance+=-2*(constraint_a_linear_term_[new_dimension_*i+j]
                    +constraint_b_linear_term_[new_dimension_*i+j]
                     *(coordinates.get(j, n1)+coordinates.get(j, n2)));
    }
    double denominator=nearest_distances_[i]-distance;
    DEBUG_ASSERT_MSG(denominator>0,
       "Something is wrong, solution out of the interior");
    for(index_t j=0; j<new_dimension_; j++){
      double nominator=-2*constraint_b_linear_term_[new_dimension_*i+j];
      gradient->set(j, n1, gradient->get(j, n1) 
          +(nominator+2*exp(2*coordinates.get(j, n1)))/denominator);
      gradient->set(j, n2, gradient->get(j, n2) 
          +(nominator+2*exp(2*coordinates.get(j, n2)))/denominator);

    }    
  }

  // gradient from the objective
  Matrix ones;
  ones.Init(new_dimension_, num_of_rows_+num_of_columns_);
  ones.SetAll(sigma_*opt_var_sign_);
  la::AddTo(ones, gradient);
  
  fx_timer_stop(NULL, "gradient");
}

void RelaxedNmfIsometricBoxTightener::ComputeObjective(Matrix &coordinates, double *objective) {
  fx_timer_start(NULL, "objective");
  Matrix ones;
  ones.Init(new_dimension_, num_of_rows_+num_of_columns_);
  ones.SetAll(opt_var_sign_);
  *objective=la::Dot(coordinates.n_elements(),
                     coordinates.ptr(), 
                     ones.ptr());
  fx_timer_stop(NULL, "objective");
}

void RelaxedNmfIsometricBoxTightener::ComputeNonRelaxedObjective(Matrix &coordinates, 
                                            double *objective) {
  fx_timer_start(NULL, "non_relaxed_objective");
  *objective=values_sq_norm_;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=col+w_offset_;
    index_t h=row+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dimension_; j++) {
      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    *objective+=convex_part*(convex_part-2*values_[i]);
  } 
  fx_timer_stop(NULL, "non_relaxed_objective");
}


void RelaxedNmfIsometricBoxTightener::ComputeFeasibilityError(Matrix &coordinates, 
                                         double *error) {
  *error=0;
}

double RelaxedNmfIsometricBoxTightener::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian;
  ComputeObjective(coordinates, &lagrangian);
  lagrangian*=sigma_;
  double norm_error=values_sq_norm_;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=col+w_offset_;
    index_t h=row+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dimension_; j++) {
      norm_error+=-2*values_[i]*(objective_a_linear_term_[new_dimension_*i+j]
                                +objective_b_linear_term_[new_dimension_*i+j]
                                 *(coordinates.get(j, w)+coordinates.get(j, h)));

      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    norm_error+=convex_part*convex_part;
  } 
  if (norm_error >= function_upper_bound_) {
    return DBL_MAX;
  }
  lagrangian+=-log(function_upper_bound_-norm_error);
  double prod=1.0;
  for(index_t i=0; i<nearest_neighbor_pairs_.size(); i++) {
    index_t n1 = w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2 = w_offset_ + nearest_neighbor_pairs_[i].second; 
    double distance=0;
    for(index_t j=0; j<new_dimension_; j++) {
      distance+=exp(2*coordinates.get(j, n1))+exp(2*coordinates.get(j, n2));
      distance+=-2*(constraint_a_linear_term_[new_dimension_*i+j]
                    +constraint_b_linear_term_[new_dimension_*i+j]
                     *(coordinates.get(j, n1)+coordinates.get(j, n2)));
    }
    double denominator=nearest_distances_[i]-distance;
    if (denominator<=0) {
      return DBL_MAX;
    }
    prod*=denominator;
    if (unlikely(prod<=1e-30 || prod >=1e+30)) {
      lagrangian+=-log(prod);
      prod=1.0;
    }
  }
  lagrangian+=-log(prod);
  return lagrangian;
}

void RelaxedNmfIsometricBoxTightener::UpdateLagrangeMult(Matrix &coordinates) {
  
}

void RelaxedNmfIsometricBoxTightener::Project(Matrix *coordinates) {
  fx_timer_start(NULL, "project");
  for(index_t i=0; i<num_of_rows_+num_of_columns_; i++) {
    for(index_t j=0; j<x_lower_bound_.length(); j++) {
      if (coordinates->get(j, i) < x_lower_bound_[j]) {
        coordinates->set(j, i, x_lower_bound_[j]);
      } else {
        if (coordinates->get(j, i)> x_upper_bound_[j]) {
          coordinates->set(j, i, x_upper_bound_[j]);
        }
      }
    }
  }
  fx_timer_stop(NULL, "project");
}

void RelaxedNmfIsometricBoxTightener::set_sigma(double sigma) {
  sigma_=sigma;
}

void RelaxedNmfIsometricBoxTightener::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(new_dimension_, num_of_rows_ + num_of_columns_);
  for(index_t i=0; i<init_data->n_cols(); i++) {
    for(index_t j=0; j<init_data->n_rows(); j++) {
      init_data->set(j, i, 
          (x_lower_bound_[j] + x_upper_bound_[j])/2);
    }
  }
}

bool RelaxedNmfIsometricBoxTightener::IsDiverging(double objective) {
  return false;
} 

bool RelaxedNmfIsometricBoxTightener::IsOptimizationOver(Matrix &coordinates, 
                                    Matrix &gradient, double step) {

  index_t num_of_constraints=1+num_of_nearest_pairs_;
  if (num_of_constraints/sigma_ < desired_duality_gap_) {
    return true;
  } else {
    return false;
  }
 
}
/*  double objective;
  ComputeObjective(coordinates, &objective);
  if (fabs(objective-previous_objective_)/objective<0.01) {
    previous_objective_=objective;
    return true;
  } else  {
     previous_objective_=objective;
     return false;
   
  }
*/  

  


bool RelaxedNmfIsometricBoxTightener::IsIntermediateStepOver(Matrix &coordinates, 
                                        Matrix &gradient, 
                                        double step) {
  double norm_gradient=math::Pow<1,2>(la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr()));
  if (norm_gradient*step < grad_tolerance_) {
    return true;
  }
  return false;


/*
  double norm_gradient=la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr());
  if (norm_gradient*step < grad_tolerance_) {
    return true;
  }
  return false;
*/
}

double RelaxedNmfIsometricBoxTightener::GetSoftLowerBound() {
  return soft_lower_bound_;
}

bool RelaxedNmfIsometricBoxTightener::IsInfeasible() {
  return is_infeasible_;
}

#endif
#endif
