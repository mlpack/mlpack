/*
 * =====================================================================================
 * 
 *       Filename:  gop_nmf_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  07/21/2008 11:10:48 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
const fx_submodule_doc gop_nmf_submodules[] = {
  {"l_bfgs", &lbfgs_doc,
   "  Responsible for the lbfgs.\n"},
  {"geometric_nmf_objective", &geometric_nmf_objective_doc,
   "  Stores results for geometric nmf objective.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_entry_doc gop_nmf_engine_entries[] = {
  {"new_dimension", FX_PARAM, FX_INT, NULL,
   "  New dimension for the nmf.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc gop_nmf_engine_doc = {
  gop_nmf_engine_entries, gop_nmf_submodules,
  "All the entries for the engine.\n"
};



void RelaxedNmf::Init(ArrayList<index_t> &rows,
    ArrayList<index_t> &columns,
    ArrayList<double> &values,
    index_t new_dim,
    double grad_tolerance,
    Matrix &x_lower_bound,
    Matrix &x_upper_bound) {
  
  grad_tolerance_=grad_tolerance;
  previous_objective_=DBL_MAX;
  rows_.InitCopy(rows);
  num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
  columns_.InitCopy(columns);
  num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
  w_offset_=num_of_rows_;
  h_offset_=0;
  values_.InitCopy(values);
  values_sq_norm_=la::Dot(values_.size(), values_.begin(), values_.begin());
  x_lower_bound_.Copy(x_lower_bound);
  x_upper_bound_.Copy(x_upper_bound);
  soft_lower_bound_=values_sq_norm_;
  new_dim_=new_dim;
  // Generate the linear terms
  // and compute the soft lower bound
  a_linear_term_.Init(new_dim_*values_.size());
  b_linear_term_.Init(new_dim_*values_.size());
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row+w_offset_;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dim_; j++) {
      double y_lower=x_lower_bound_.get(j, w) + x_lower_bound_.get(j, h);
      double y_upper=x_upper_bound_.get(j, w) + x_upper_bound_.get(j, h);      
      a_linear_term_[new_dim_*i+j]=
          (y_upper*exp(y_lower)-y_lower*exp(y_upper))/(y_upper-y_lower);
      b_linear_term_[new_dim_*i+j]=(exp(y_upper)-exp(y_lower))/
          (y_upper-y_lower);
      DEBUG_ASSERT(b_linear_term_[new_dim_*i+j]>=0);
      convex_part+=exp(y_lower);
      soft_lower_bound_+= -2*values_[i]*(
                            a_linear_term_[new_dim_*i+j]
                            +b_linear_term_[new_dim_*i+j]*y_upper);
    }
    soft_lower_bound_+=convex_part*convex_part;
  }
}
void RelaxedNmf::Init(fx_module *module,
            ArrayList<index_t> &rows,
            ArrayList<index_t> &columns,
            ArrayList<double> &values,
            Matrix &x_lower_bound, // the initial lower bound for x (optimization variable)
            Matrix &x_upper_bound  // the initial upper bound for x (optimization variable)
           ) {
  grad_tolerance_=fx_param_double(module, "grad_tolerance", 0.1);
  new_dim_=fx_param_int(module, "new_dimension", 5);
  Init(rows, columns, values, new_dim_, grad_tolerance_, 
       x_lower_bound, x_upper_bound);
}

void RelaxedNmf::Destruct() {
  num_of_rows_=-1;;
  num_of_columns_=-1;
  new_dim_=-1;
  x_lower_bound_.Destruct();
  x_upper_bound_.Destruct();
  a_linear_term_.Destruct();
  b_linear_term_.Destruct();
  rows_.Renew();
  columns_.Renew();
  values_.Renew();
  soft_lower_bound_=-DBL_MAX;
  values_sq_norm_=-DBL_MAX;
} 


void RelaxedNmf::ComputeGradient(Matrix &coordinates, Matrix *gradient) {
  fx_timer_start(NULL, "gradient");
  gradient->SetAll(0.0);
  // gradient from the objective
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row+w_offset_;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dim_; j++) {
      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
     // convex_part+=ComputeExpTaylorApproximation(coordinates.get(j, w)+coordinates.get(j, h), 10);
    }
    for(index_t j=0; j<new_dim_; j++) {
      double grad=2*convex_part*exp(coordinates.get(j, w)
                                    +coordinates.get(j, h))
                      -2*values_[i]*b_linear_term_[new_dim_*i+j];
      gradient->set(j, w, gradient->get(j, w)+grad);
      gradient->set(j, h, gradient->get(j, h)+grad);
    }
  }
  fx_timer_stop(NULL, "gradient");
}

void RelaxedNmf::ComputeObjective(Matrix &coordinates, double *objective) {
  fx_timer_start(NULL, "objective");
  *objective=values_sq_norm_;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row+w_offset_;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dim_; j++) {
      *objective+=-2*values_[i]*(a_linear_term_[new_dim_*i+j]
                                +b_linear_term_[new_dim_*i+j]
                                 *(coordinates.get(j, w)+coordinates.get(j, h)));

      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
      // convex_part+=ComputeExpTaylorApproximation(coordinates.get(j, w)+coordinates.get(j, h), 10);
    }
    *objective+=convex_part*convex_part;
  } 
  fx_timer_stop(NULL, "objective");
}

void RelaxedNmf::ComputeNonRelaxedObjective(Matrix &coordinates, 
                                            double *objective) {
  fx_timer_start(NULL, "non_relaxed_objective");
  *objective=values_sq_norm_;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row+w_offset_;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dim_; j++) {
      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    *objective+=convex_part*(convex_part-2*values_[i]);
  } 
  fx_timer_stop(NULL, "non_relaxed_objective");
}


void RelaxedNmf::ComputeFeasibilityError(Matrix &coordinates, 
                                         double *error) {
  *error=0;
}

double RelaxedNmf::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian;
  ComputeObjective(coordinates, &lagrangian);
  return lagrangian;
}

void RelaxedNmf::UpdateLagrangeMult(Matrix &coordinates) {
  
}

void RelaxedNmf::Project(Matrix *coordinates) {
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

void RelaxedNmf::set_sigma(double sigma) {

}

void RelaxedNmf::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(new_dim_, num_of_rows_ + num_of_columns_);
  for(index_t i=0; i<init_data->n_rows(); i++) {
    for(index_t j=0; j<init_data->n_cols(); j++) {
      init_data->set(i, j, 
          (x_lower_bound_.get(i, j) + x_upper_bound_.get(i, j))/2);
    }
  }
}

bool RelaxedNmf::IsDiverging(double objective) {
  return false;
} 

bool RelaxedNmf::IsOptimizationOver(Matrix &coordinates, 
                                    Matrix &gradient, double step) {

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
  double norm_gradient=la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr());
  if (norm_gradient*step < grad_tolerance_) {
    return true;
  }
  return false;
  
}

bool RelaxedNmf::IsIntermediateStepOver(Matrix &coordinates, 
                                        Matrix &gradient, 
                                        double step) {
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
  double norm_gradient=la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr());
  if (norm_gradient*step < grad_tolerance_) {
    return true;
  }
  return false;

}

double RelaxedNmf::GetSoftLowerBound() {
  return soft_lower_bound_;
}

inline double RelaxedNmf::ComputeExpTaylorApproximation(double x, index_t order) {
  double value=1.0;
  double factorial=1.0;
  double power=1.0;
  for(index_t i=0; i<order; i++) {
    power*=power;
    factorial*=i;
    value+=power/factorial; 
  }
  return value;
}


void RelaxedNmf1::Init(ArrayList<index_t> &rows,
    ArrayList<index_t> &columns,
    ArrayList<double> &values,
    index_t new_dim,
    double grad_tolerance,
    Matrix &x_lower_bound,
    Matrix &x_upper_bound) {
  
  grad_tolerance_=grad_tolerance;
  rows_.InitCopy(rows);
  num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
  columns_.InitCopy(columns);
  num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
  w_offset_=num_of_rows_;
  h_offset_=0;
  values_.InitCopy(values);
  values_sq_norm_=la::Dot(values_.size(), values_.begin(), values_.begin());
  x_lower_bound_.Copy(x_lower_bound);
  x_upper_bound_.Copy(x_upper_bound);
  soft_lower_bound_=values_sq_norm_;
  new_dim_=new_dim;
  // Generate the linear terms
  // and compute the soft lower bound
  a_linear_term_.Init(new_dim_*values_.size());
  b_linear_term_.Init(new_dim_*values_.size());
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row+w_offset_;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dim_; j++) {
      double y_lower=x_lower_bound_.get(j, w) + x_lower_bound_.get(j, h);
      double y_upper=x_upper_bound_.get(j, w) + x_upper_bound_.get(j, h);      
      a_linear_term_[new_dim_*i+j]=
          (y_upper*exp(y_lower)-y_lower*exp(y_upper))/(y_upper-y_lower);
      b_linear_term_[new_dim_*i+j]=(exp(y_upper)-exp(y_lower))/
          (y_upper-y_lower);
      DEBUG_ASSERT(b_linear_term_[new_dim_*i+j]>=0);
      convex_part+=exp(y_lower);
      soft_lower_bound_+= -2*values_[i]*(
                            a_linear_term_[new_dim_*i+j]
                            +b_linear_term_[new_dim_*i+j]*y_upper);
    }
    soft_lower_bound_+=convex_part*convex_part;
  }
}

void RelaxedNmf1::Destruct() {
  num_of_rows_=-1;;
  num_of_columns_=-1;
  new_dim_=-1;
  x_lower_bound_.Destruct();
  x_upper_bound_.Destruct();
  a_linear_term_.Destruct();
  b_linear_term_.Destruct();
  rows_.Renew();
  columns_.Renew();
  values_.Renew();
  soft_lower_bound_=-DBL_MAX;
  values_sq_norm_=-DBL_MAX;
} 


void RelaxedNmf1::ComputeGradient(Matrix &coordinates, Matrix *gradient) {
  fx_timer_start(NULL, "gradient");
  gradient->SetAll(0.0);
  // gradient from the objective
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row+w_offset_;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dim_; j++) {
      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    for(index_t j=0; j<new_dim_; j++) {
      double grad=2*convex_part*exp(coordinates.get(j, w)
                                    +coordinates.get(j, h))
                      -2*values_[i]*b_linear_term_[new_dim_*i+j];
      gradient->set(j, w, gradient->get(j, w)+grad);
      gradient->set(j, h, gradient->get(j, h)+grad);
    }
  }
  la::Scale(sigma_, gradient); 
  // gradient for the bound constraints
  for(index_t i=0; i<gradient->n_cols(); i++) {
    for(index_t j=0; j<gradient->n_rows(); j++) {
      double denominator1=coordinates.get(j, i)-x_lower_bound_.get(j, i);
      double denominator2=x_upper_bound_.get(j, i)-coordinates.get(j, i);
      gradient->set(j, i,-1.0/denominator1+1.0/denominator2 
                         +gradient->get(j, i));
    }
  } 
 
  fx_timer_stop(NULL, "gradient");
}

void RelaxedNmf1::ComputeObjective(Matrix &coordinates, double *objective) {
  fx_timer_start(NULL, "objective");
  *objective=values_sq_norm_;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row+w_offset_;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dim_; j++) {
      *objective+=-2*values_[i]*(a_linear_term_[new_dim_*i+j]
                                +b_linear_term_[new_dim_*i+j]
                                 *(coordinates.get(j, w)+coordinates.get(j, h)));

      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    *objective+=convex_part*convex_part;
  } 
  fx_timer_stop(NULL, "objective");
}

void RelaxedNmf1::ComputeNonRelaxedObjective(Matrix &coordinates, 
                                            double *objective) {
  fx_timer_start(NULL, "non_relaxed_objective");
  *objective=values_sq_norm_;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row+w_offset_;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dim_; j++) {
      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    *objective+=convex_part*(convex_part-2*values_[i]);
  } 
  fx_timer_stop(NULL, "non_relaxed_objective");
}


void RelaxedNmf1::ComputeFeasibilityError(Matrix &coordinates, 
                                         double *error) {
  *error=0;
}

double RelaxedNmf1::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian;
  ComputeObjective(coordinates, &lagrangian);
  lagrangian*=sigma_;
  // penalty for the bounds
  double temp_product=1;
  for(index_t i=0; i<coordinates.n_cols(); i++) {
    for(index_t j=0; j<coordinates.n_rows(); j++) {
      double denominator1=coordinates.get(j, i)-x_lower_bound_.get(j, i);
      double denominator2=x_upper_bound_.get(j, i)-coordinates.get(j, i);
      if (denominator1<=0 || denominator2<=0) {
        return DBL_MAX;
      } else {
        temp_product*=denominator1*denominator2;
        if (unlikely(temp_product>1e+40 || temp_product<1e-40)) {
          lagrangian+=-log(temp_product);
          temp_product=1;    
        }
      }
    }  
  } 
  lagrangian+=-log(temp_product);
 
  return lagrangian;
}

void RelaxedNmf1::UpdateLagrangeMult(Matrix &coordinates) {
  
}

void RelaxedNmf1::Project(Matrix *coordinates) {
}

void RelaxedNmf1::set_sigma(double sigma) {
  sigma_=sigma;
}

void RelaxedNmf1::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(new_dim_, num_of_rows_ + num_of_columns_);
  for(index_t i=0; i<init_data->n_rows(); i++) {
    for(index_t j=0; j<init_data->n_cols(); j++) {
      init_data->set(i, j, 
          (x_lower_bound_.get(i, j) + x_upper_bound_.get(i, j))/2);
    }
  }
}

bool RelaxedNmf1::IsDiverging(double objective) {
  return false;
} 

bool RelaxedNmf1::IsOptimizationOver(Matrix &coordinates, 
                                    Matrix &gradient, double step) {
  
  index_t num_of_constraints=2*coordinates.n_elements();
  if (sigma_/num_of_constraints > 10) {
    return true;
  } else {
    return false;
  }
}

bool RelaxedNmf1::IsIntermediateStepOver(Matrix &coordinates, 
                                        Matrix &gradient, 
                                        double step) {
  double norm_gradient=la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr());
  if (norm_gradient*step < grad_tolerance_) {
    return true;
  }
  return false;
}

double RelaxedNmf1::GetSoftLowerBound() {
  return soft_lower_bound_;
}

///////////////////////////////////////////////////////////////////////
////////RelaxedNmfIsometric////////////////////////////////////////////
void RelaxedNmfIsometric::Init(fx_module *module,
                      ArrayList<index_t> &rows,
                      ArrayList<index_t> &columns,
                      ArrayList<double> &values,
                      Matrix &x_lower_bound, 
                      Matrix &x_upper_bound) {
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
  // Generate the linear terms
  // and compute the soft lower bound
  objective_a_linear_term_.Init(new_dimension_*values_.size());
  objective_b_linear_term_.Init(new_dimension_*values_.size());
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row+w_offset_;
    index_t h=col+h_offset_;
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


void RelaxedNmfIsometric::Destruct() {
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


void RelaxedNmfIsometric::ComputeGradient(Matrix &coordinates, Matrix *gradient) {
  fx_timer_start(NULL, "gradient");
  gradient->SetAll(0.0);
 
  
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
                      -2*values_[i]*objective_b_linear_term_[new_dimension_*i+j];
      gradient->set(j, w, gradient->get(j, w)+grad);
      gradient->set(j, h, gradient->get(j, h)+grad);
    }
  }
  la::Scale(sigma_, gradient);

 
  // gradient from the distance constraints
  for(index_t i=0; i<nearest_neighbor_pairs_.size(); i++) {
    index_t n1 = w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2 = w_offset_ + nearest_neighbor_pairs_[i].second; 
    double distance=0;
    for(index_t j=0; j<new_dimension_; j++) {
      distance+=exp(2*coordinates.get(j, n1))+exp(2*coordinates.get(j, n2));
      distance+=-2*(constraint_a_linear_term_[new_dimension_*i+j]
                    +constraint_b_linear_term_[new_dimension_*i+j]
                     *exp(coordinates.get(j, n1)+coordinates.get(j, n2)));
    }
    double denominator=nearest_distances_[i]-distance;
    DEBUG_ASSERT_MSG(denominator<=0,
       "Something is wrong, solution out of the interior");
    for(index_t j=0; j<new_dimension_; j++){
      double nominator=-2*constraint_b_linear_term_[new_dimension_*i+j]
                         *exp(coordinates.get(j, n1)+coordinates.get(j, n2));
      gradient->set(j, n1, gradient->get(i, n1) 
          +(nominator+2*exp(2*coordinates.get(j, n1)))/denominator);
      gradient->set(j, n2, gradient->get(i, n2) 
          +(nominator+2*exp(2*coordinates.get(j, n2)))/denominator);

    }    
  }

  fx_timer_stop(NULL, "gradient");
}

void RelaxedNmfIsometric::ComputeObjective(Matrix &coordinates, double *objective) {
  fx_timer_start(NULL, "objective");
  double norm_error=values_sq_norm_;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row+w_offset_;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dimension_; j++) {
      norm_error+=-2*values_[i]*(objective_a_linear_term_[new_dimension_*i+j]
                                +objective_b_linear_term_[new_dimension_*i+j]
                                 *(coordinates.get(j, w)+coordinates.get(j, h)));

      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    norm_error+=convex_part*convex_part;
  } 
 
  *objective=norm_error;
  fx_timer_stop(NULL, "objective");
}

void RelaxedNmfIsometric::ComputeNonRelaxedObjective(Matrix &coordinates, 
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


void RelaxedNmfIsometric::ComputeFeasibilityError(Matrix &coordinates, 
                                         double *error) {
  *error=0;
}

double RelaxedNmfIsometric::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian;
  ComputeObjective(coordinates, &lagrangian);
  lagrangian*=sigma_;
  double prod=1.0;
  for(index_t i=0; i<nearest_neighbor_pairs_.size(); i++) {
    index_t n1 = w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2 = w_offset_ + nearest_neighbor_pairs_[i].second; 
    double distance=0;
    for(index_t j=0; j<new_dimension_; j++) {
      distance+=exp(2*coordinates.get(j, n1))+exp(2*coordinates.get(j, n2));
      distance+=-2*(constraint_a_linear_term_[new_dimension_*i+j]
                    +constraint_b_linear_term_[new_dimension_*i+j]
                     *exp(coordinates.get(j, n1)+coordinates.get(j, n2)));
    }
    double denominator=nearest_distances_[i]-distance;
    if (denominator<=0) {
      return DBL_MAX;
    }
    prod*=denominator;
    if (unlikely(prod<=1e-30 || prod >=1e+30)) {
      lagrangian+=log(prod);
      prod=1.0;
    }
  }
  lagrangian+=log(prod);
  return lagrangian;
}

void RelaxedNmfIsometric::UpdateLagrangeMult(Matrix &coordinates) {
  
}

void RelaxedNmfIsometric::Project(Matrix *coordinates) {
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

void RelaxedNmfIsometric::set_sigma(double sigma) {
  sigma_=sigma;
}

void RelaxedNmfIsometric::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(new_dimension_, num_of_rows_ + num_of_columns_);
  for(index_t i=0; i<init_data->n_rows(); i++) {
    for(index_t j=0; j<init_data->n_cols(); j++) {
      init_data->set(i, j, 
          (x_lower_bound_.get(i, j) + x_upper_bound_.get(i, j))/2);
    }
  }
}

bool RelaxedNmfIsometric::IsDiverging(double objective) {
  return false;
} 

bool RelaxedNmfIsometric::IsOptimizationOver(Matrix &coordinates, 
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

  


bool RelaxedNmfIsometric::IsIntermediateStepOver(Matrix &coordinates, 
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

double RelaxedNmfIsometric::GetSoftLowerBound() {
  return soft_lower_bound_;
}

bool RelaxedNmfIsometric::IsInfeasible() {
  return is_infeasible_;
}




//////////////////////////////////////////////////////////////////////
//GopNmfEngine/////////////////////////////////////////////////////
void GopNmfEngine::Init(fx_module *module, Matrix &data_points) {
  module_=module;
  l_bfgs_module_=fx_submodule(module_, "l_bfgs");
  new_dim_ = fx_param_int(module_, "new_dimension", 5);
  grad_tolerance_ = fx_param_double(module_, "grad_tolerance", 1e-3);
  desired_global_optimum_gap_= fx_param_double(module_, "opt_gap", 1e-1);
  PreprocessData(data_points);
  fx_set_param_int(l_bfgs_module_, "new_dimension", new_dim_);
  fx_set_param_int(l_bfgs_module_, "num_of_points", num_of_rows_+num_of_columns_);
  fx_set_param_int(l_bfgs_module_, "mem_bfgs", 8);
  fx_set_param_bool(l_bfgs_module_, "use_default_termination", false);
  fx_set_param_bool(l_bfgs_module_, "silent", false);
}

void GopNmfEngine::ComputeGlobalOptimum() {
   
  soft_prunes_=0;
  hard_prunes_=0;
  double norm_values=la::Dot(values_.size(), values_.begin(), values_.begin());
  NOTIFY("Values norm:%lg", norm_values);
  // Solve for  this bounding box
  Matrix lower_bound;
  Matrix upper_bound;
  lower_bound.Copy(x_lower_bound_);
  upper_bound.Copy(x_upper_bound_);
  total_volume_=ComputeVolume(lower_bound, upper_bound);
  soft_pruned_volume_=0.0;
  hard_pruned_volume_=0.0;  
  RelaxedNmf opt_fun;
  opt_fun.Init(rows_, columns_, values_,  
               new_dim_, grad_tolerance_,
               lower_bound, upper_bound); 
  LowerOptimizer lower_optimizer;
  lower_optimizer.Init(&opt_fun, l_bfgs_module_);
  Matrix init_data;
  opt_fun.GiveInitMatrix(&init_data);
  lower_optimizer.set_coordinates(init_data);
  lower_optimizer.ComputeLocalOptimumBFGS();
  return;
  double new_upper_global_optimum;
  // Compute a new upper bound for the global maximum
  opt_fun.ComputeNonRelaxedObjective(*lower_optimizer.coordinates(), &new_upper_global_optimum);
  double new_lower_global_optimum;
  opt_fun.ComputeObjective(init_data, &new_lower_global_optimum);
  upper_solution_.first=new_upper_global_optimum;
  upper_solution_.second.Own(lower_optimizer.coordinates());
/*  SolutionPack pack;
  pack.solution_.Copy(*optimizer.coordinates());
  pack.box_.first.Copy(lower_bound);
  pack.box_.second.Copy(upper_bound);
  lower_solution_.insert(
      std::make_pair(new_lower_global_optimum,pack));
*/

  iteration_=0;
  NOTIFY("iteration:%i upper_global_optimum:%lg lower_global_optimum:%lg",
      iteration_, upper_solution_.first, new_lower_global_optimum); 
  while (true) {
    // these bounds correspond to the optimization variables
    Matrix left_upper_bound;
    Matrix left_lower_bound;
    Matrix right_upper_bound;
    Matrix right_lower_bound;
    
    Split(lower_bound, upper_bound, 
          &left_lower_bound, &left_upper_bound,
          &right_lower_bound, &right_upper_bound);

    lower_optimizer.Destruct();
    opt_fun.Destruct();
    // optimize left
    opt_fun.Init(rows_, columns_, values_,  
                 new_dim_, grad_tolerance_,
                 left_lower_bound, left_upper_bound); 
    lower_optimizer.Init(&opt_fun, l_bfgs_module_);
    if (opt_fun.GetSoftLowerBound() > upper_solution_.first) {
      soft_prunes_++;
      soft_pruned_volume_+=ComputeVolume(left_lower_bound, left_upper_bound);
    } else {
      init_data.Destruct();
      opt_fun.GiveInitMatrix(&init_data);
      lower_optimizer.set_coordinates(init_data);
      lower_optimizer.ComputeLocalOptimumBFGS();
      double new_upper_global_optimum;
      // Compute a new upper global maximum
      opt_fun.ComputeNonRelaxedObjective(init_data, &new_upper_global_optimum);
      if (new_upper_global_optimum <upper_solution_.first) {
        upper_solution_.first=new_upper_global_optimum;
        upper_solution_.second.Destruct();
        upper_solution_.second.Own(lower_optimizer.coordinates());
      }
      double new_lower_global_optimum;
      opt_fun.ComputeObjective(*lower_optimizer.coordinates(), &new_lower_global_optimum);
      if (new_lower_global_optimum < upper_solution_.first) {
        SolutionPack pack;
        pack.box_.first.Init(1,1);
        pack.box_.second.Init(1, 1);
        pack.solution_.Init(1, 1);
            std::map<double, SolutionPack>::iterator it; 
        it=lower_solution_.insert(
            std::make_pair(new_lower_global_optimum, pack));
 
        it->second.solution_.Destruct();
        it->second.box_.first.Destruct();
        it->second.box_.second.Destruct();
       
        it->second.solution_.Copy(*lower_optimizer.coordinates());
        it->second.box_.first.Own(&left_lower_bound);
        it->second.box_.second.Own(&left_upper_bound);
 
      } else {
        hard_prunes_++;
        hard_pruned_volume_+=ComputeVolume(left_lower_bound, left_upper_bound);
      }
      NOTIFY("left_iteration:%i upper_global_optimum:%lg lower_global_optimum:%lg "
             "upper_bound:%lg lower_bound:%lg",
          iteration_, upper_solution_.first, lower_solution_.begin()->first,
          new_upper_global_optimum, new_lower_global_optimum);
   }

    // optimize right
    lower_optimizer.Destruct();
    opt_fun.Destruct();
    init_data.Destruct();
    opt_fun.Init(rows_, columns_, values_,  
                 new_dim_, grad_tolerance_, right_lower_bound, right_upper_bound); 
    lower_optimizer.Init(&opt_fun, l_bfgs_module_);
    opt_fun.GiveInitMatrix(&init_data);
    lower_optimizer.set_coordinates(init_data);
    if (opt_fun.GetSoftLowerBound() > upper_solution_.first) {
      soft_prunes_++;
      soft_pruned_volume_+=ComputeVolume(right_lower_bound, right_upper_bound);
    } else {
      init_data.Destruct();
      opt_fun.GiveInitMatrix(&init_data);
      lower_optimizer.set_coordinates(init_data);
      lower_optimizer.Reset();
      lower_optimizer.ComputeLocalOptimumBFGS();
      double new_upper_global_optimum;
      // Compute a new upper global maximum
      opt_fun.ComputeNonRelaxedObjective(init_data, &new_upper_global_optimum);
      if (new_upper_global_optimum <upper_solution_.first) {
        upper_solution_.first=new_upper_global_optimum;
        upper_solution_.second.Destruct();
        upper_solution_.second.Own(lower_optimizer.coordinates());
      }
      double new_lower_global_optimum;
      opt_fun.ComputeObjective(*lower_optimizer.coordinates(), &new_lower_global_optimum);
      if (new_lower_global_optimum < upper_solution_.first) {
        SolutionPack pack;
        pack.box_.first.Init(1,1);
        pack.box_.second.Init(1, 1);
        pack.solution_.Init(1, 1);
         std::multimap<double, SolutionPack>::iterator it; 
        it=lower_solution_.insert(
            std::make_pair(new_lower_global_optimum,pack));
        it->second.solution_.Destruct();
        it->second.box_.first.Destruct();
        it->second.box_.second.Destruct();
       
        it->second.solution_.Copy(*lower_optimizer.coordinates());
        it->second.box_.first.Own(&right_lower_bound);
        it->second.box_.second.Own(&right_upper_bound);
      
      } else {
        hard_prunes_++;
        hard_pruned_volume_+=ComputeVolume(right_lower_bound, right_upper_bound);
      }
      NOTIFY("right_iteration:%i upper_global_optimum:%lg lower_global_optimum:%lg "
             "upper_bound:%lg lower_bound:%lg",
          iteration_, upper_solution_.first, lower_solution_.begin()->first,
          new_upper_global_optimum, new_lower_global_optimum);
      NOTIFY("hard_prunes:%i soft_prunes:%i", hard_prunes_, soft_prunes_);

    }
     
    // Check for convergence
    DEBUG_ASSERT(upper_solution_.first - lower_solution_.begin()->first); 
    if (upper_solution_.first - lower_solution_.begin()->first < 
        desired_global_optimum_gap_) {
      std::multimap<double, SolutionPack>::iterator it;
      for(it=lower_solution_.begin(); it!=lower_solution_.end(); it++) {
        soft_pruned_volume_+=ComputeVolume(it->second.box_.first, 
            it->second.box_.second);
      }
      NOTIFY("Algorithm converged global optimum found");
      fx_result_double(module_, "upper_global_optimum", upper_solution_.first);
      fx_result_double(module_, "lower_global_optimum", lower_solution_.begin()->first);
      fx_result_int(module_, "soft_prunes", soft_prunes_);
      fx_result_int(module_, "hard_prunes", hard_prunes_);
      fx_result_int(module_, "iterations", iteration_);
      fx_result_double(module_, "soft_pruned_volume", soft_pruned_volume_);
      fx_result_double(module_, "hard_pruned_volume", hard_pruned_volume_);
      fx_result_double(module_, "total_volume", total_volume_);
      fx_result_double(module_, "volume_percentage_of_pruning", 
          100*(soft_pruned_volume_+hard_pruned_volume_)/total_volume_);
      fx_result_double(module_, "prunes_over_iterations_percentage", 
          100.0*(hard_prunes_+soft_prunes_)/iteration_);

      for(index_t i=0; i<upper_solution_.second.n_rows(); i++) {
        for(index_t j=0; j<upper_solution_.second.n_cols(); j++) {
          upper_solution_.second.set(i, j, exp(upper_solution_.second.get(i, j)));
        }
      }
      data::Save("result.csv", upper_solution_.second);
      return;
    }    
    
    // choose next box to split and work   
    lower_bound.Destruct(); 
    lower_bound.Own(&lower_solution_.begin()->second.box_.first);
    upper_bound.Destruct();
    upper_bound.Own(&lower_solution_.begin()->second.box_.second);  
    DEBUG_ASSERT(lower_solution_.size()>0);
    lower_solution_.erase(lower_solution_.begin()); 
    iteration_++;
  }
   
}

void GopNmfEngine::ComputeTighterGlobalOptimum() {
  
  soft_prunes_=0;
  hard_prunes_=0;
  // Solve for  this bounding box
  Matrix lower_bound;
  Matrix upper_bound;
  lower_bound.Copy(x_lower_bound_);
  upper_bound.Copy(x_upper_bound_);
  
  fx_module *geometric_nmf_module;
  geometric_nmf_module=fx_submodule(module_, "geometric_nmf_objective");
  fx_set_param_int(geometric_nmf_module, "new_dimension", new_dim_); 
  RelaxedNmf lower_opt_fun;
  GeometricNmf upper_opt_fun;
  LowerOptimizer lower_optimizer;
  UpperOptimizer upper_optimizer;
 
  Matrix init_data;
  lower_opt_fun.Init(rows_, columns_, values_,  
                     new_dim_, grad_tolerance_,
                     lower_bound, upper_bound); 
  lower_opt_fun.GiveInitMatrix(&init_data);
  lower_optimizer.Init(&lower_opt_fun, l_bfgs_module_);
  lower_optimizer.set_coordinates(init_data);
  init_data.Destruct();
  lower_optimizer.ComputeLocalOptimumBFGS();
  return;
  double new_lower_global_optimum;
  lower_opt_fun.ComputeObjective(*lower_optimizer.coordinates(), &new_lower_global_optimum);
  double new_upper_global_optimum;
  upper_opt_fun.Init(geometric_nmf_module, rows_, columns_, values_,  
                     lower_bound, upper_bound); 
  upper_opt_fun.GiveInitMatrix(&init_data);
  upper_optimizer.Init(&upper_opt_fun, l_bfgs_module_);
  upper_optimizer.set_coordinates(init_data);
  init_data.Destruct();
  upper_optimizer.Reset();
  upper_optimizer.ComputeLocalOptimumBFGS();
  upper_opt_fun.ComputeObjective(*upper_optimizer.coordinates(), 
      &new_upper_global_optimum);
  upper_opt_fun.Destruct();
  upper_solution_.first=new_upper_global_optimum;
  upper_solution_.second.Own(upper_optimizer.coordinates());
  
/*  solutionpack pack;
  pack.solution_.copy(*optimizer.coordinates());
  pack.box_.first.copy(lower_bound);
  pack.box_.second.copy(upper_bound);
  lower_solution_.insert(
      std::make_pair(new_lower_global_optimum,pack));
*/

  iteration_=0;
  NOTIFY("iteration:%i upper_global_optimum:%lg lower_global_optimum:%lg",
      iteration_, upper_solution_.first, new_lower_global_optimum); 
  while (true) {
    // these bounds correspond to the optimization variables
    Matrix left_upper_bound;
    Matrix left_lower_bound;
    Matrix right_upper_bound;
    Matrix right_lower_bound;
    
    Split(lower_bound, upper_bound, 
          &left_lower_bound, &left_upper_bound,
          &right_lower_bound, &right_upper_bound);

    lower_optimizer.Destruct();
    lower_opt_fun.Destruct();
    // optimize left
    lower_opt_fun.Init(rows_, columns_, values_,  
                       new_dim_, grad_tolerance_,
                       left_lower_bound, left_upper_bound); 
    lower_optimizer.Init(&lower_opt_fun, l_bfgs_module_);
    if (lower_opt_fun.GetSoftLowerBound() > upper_solution_.first) {
      soft_prunes_++;
    } else {
      lower_opt_fun.GiveInitMatrix(&init_data);
      lower_optimizer.set_coordinates(init_data);
      init_data.Destruct();
      lower_optimizer.ComputeLocalOptimumBFGS();
      double new_upper_global_optimum;
      // Compute the upper bound here
      upper_opt_fun.Init(geometric_nmf_module, rows_, columns_, values_,  
                         left_lower_bound, left_upper_bound);
      Matrix temp_coordinates1; 
      if (upper_opt_fun.GiveInitMatrix(&init_data)==true) {
        upper_optimizer.set_coordinates(init_data);
        init_data.Destruct();
        upper_optimizer.Reset();
        upper_optimizer.ComputeLocalOptimumBFGS();
        upper_opt_fun.ComputeObjective(*upper_optimizer.coordinates(), 
            &new_upper_global_optimum);
        temp_coordinates1.Alias(*upper_optimizer.coordinates());
      } else {
         init_data.Destruct();
         lower_opt_fun.ComputeNonRelaxedObjective(*lower_optimizer.coordinates(), 
            &new_upper_global_optimum);
        temp_coordinates1.Alias(*lower_optimizer.coordinates());
      }
      upper_opt_fun.Destruct();
      if (new_upper_global_optimum <upper_solution_.first) {
        upper_solution_.first=new_upper_global_optimum;
        upper_solution_.second.Destruct();
        upper_solution_.second.Own(&temp_coordinates1);
      }
      double new_lower_global_optimum;
      lower_opt_fun.ComputeObjective(*lower_optimizer.coordinates(), &new_lower_global_optimum);
      //new_lower_global_optimum=std::max(0.0, new_lower_global_optimum);
      if (new_lower_global_optimum < upper_solution_.first) {
        SolutionPack pack;
        lower_optimizer.CopyCoordinates(&pack.solution_);
        pack.box_.first.Own(&left_lower_bound);
        pack.box_.second.Own(&left_upper_bound);
        lower_solution_.insert(
            std::make_pair(new_lower_global_optimum, pack));
      } else {
        hard_prunes_++;
      }
      NOTIFY("left_iteration:%i upper_global_optimum:%lg lower_global_optimum:%lg "
             "upper_bound:%lg lower_bound:%lg",
          iteration_, upper_solution_.first, lower_solution_.begin()->first,
          new_upper_global_optimum, new_lower_global_optimum);
   }
   // optimize right
   lower_optimizer.Destruct();
   lower_opt_fun.Destruct();
   lower_opt_fun.Init(rows_, columns_, values_,  
                      new_dim_, grad_tolerance_, right_lower_bound, right_upper_bound); 
   lower_optimizer.Init(&lower_opt_fun, l_bfgs_module_);
   lower_opt_fun.GiveInitMatrix(&init_data);
   lower_optimizer.set_coordinates(init_data);
   init_data.Destruct();
   if (lower_opt_fun.GetSoftLowerBound() > upper_solution_.first) {
     soft_prunes_++;
   } else {
     lower_opt_fun.GiveInitMatrix(&init_data);
     lower_optimizer.set_coordinates(init_data);
     init_data.Destruct();
     lower_optimizer.ComputeLocalOptimumBFGS();
     double new_upper_global_optimum;
     // Compute the upper bound here
     upper_opt_fun.Init(geometric_nmf_module, rows_, columns_, values_,  
                        right_lower_bound, right_upper_bound); 
     Matrix temp_coordinates2;
     if (upper_opt_fun.GiveInitMatrix(&init_data) == true) {
       upper_optimizer.set_coordinates(init_data);
       init_data.Destruct();
       upper_optimizer.Reset();
       upper_optimizer.ComputeLocalOptimumBFGS();
       upper_opt_fun.ComputeObjective(*upper_optimizer.coordinates(), 
           &new_upper_global_optimum);
       temp_coordinates2.Alias(*upper_optimizer.coordinates());
      } else {
        init_data.Destruct();
        lower_opt_fun.ComputeNonRelaxedObjective(*lower_optimizer.coordinates(), 
            &new_upper_global_optimum);
        temp_coordinates2.Alias(*lower_optimizer.coordinates());
      }
      upper_opt_fun.Destruct();
      
      if (new_upper_global_optimum <upper_solution_.first) {
        upper_solution_.first=new_upper_global_optimum;
        upper_solution_.second.Destruct();
        upper_solution_.second.Copy(temp_coordinates2);
      }
      double new_lower_global_optimum;
      lower_opt_fun.ComputeObjective(*lower_optimizer.coordinates(), &new_lower_global_optimum);
      //new_lower_global_optimum=std::max(0.0, new_lower_global_optimum);
      if (new_lower_global_optimum < upper_solution_.first) {
        SolutionPack pack;
        lower_optimizer.CopyCoordinates(&pack.solution_);
        pack.box_.first.Own(&right_lower_bound);
        pack.box_.second.Own(&right_upper_bound);
        lower_solution_.insert(
            std::make_pair(new_lower_global_optimum, pack));
      } else {
        hard_prunes_++;
      }
      NOTIFY("right_iteration:%i upper_global_optimum:%lg lower_global_optimum:%lg "
             "upper_bound:%lg lower_bound:%lg",
          iteration_, upper_solution_.first, lower_solution_.begin()->first,
          new_upper_global_optimum, new_lower_global_optimum);
      NOTIFY("hard_prunes:%i soft_prunes:%i", hard_prunes_, soft_prunes_);

    }
     
    // check for convergence
    DEBUG_ASSERT(upper_solution_.first - lower_solution_.begin()->first); 
    if (upper_solution_.first - lower_solution_.begin()->first < 
        desired_global_optimum_gap_) {
      NOTIFY("algorithm converged global optimum found");
      fx_result_double(module_, "upper_global_optimum", upper_solution_.first);
      fx_result_double(module_, "lower_global_optimum", lower_solution_.begin()->first);
      fx_result_int(module_, "soft_prunes", soft_prunes_);
      fx_result_int(module_, "hard_prunes", hard_prunes_);
      fx_result_int(module_, "iterations", iteration_);
      for(index_t i=0; i<upper_solution_.second.n_rows(); i++) {
        for(index_t j=0; j<upper_solution_.second.n_cols(); j++) {
          upper_solution_.second.set(i, j, exp(upper_solution_.second.get(i, j)));
        }
      }
      data::Save("result.csv", upper_solution_.second);
      return;
    }    
    
    // choose next box to split and work   
    lower_bound.Destruct(); 
    lower_bound.Own(&lower_solution_.begin()->second.box_.first);
    upper_bound.Destruct();
    upper_bound.Own(&lower_solution_.begin()->second.box_.second);  
    DEBUG_ASSERT(lower_solution_.size()>0);
    lower_solution_.erase(lower_solution_.begin()); 
    iteration_++;
  }
   
}

void GopNmfEngine::Split(Matrix &lower_bound, Matrix &upper_bound, 
          Matrix *left_lower_bound, Matrix *left_upper_bound,
          Matrix *right_lower_bound, Matrix *right_upper_bound) {
  double widest_range=0;
  index_t widest_range_i=-1;
  index_t widest_range_j=-1;
  for(index_t i=0; i<lower_bound.n_rows(); i++) {
    for(index_t j=0; j<lower_bound.n_cols(); j++) {
      DEBUG_ASSERT(upper_bound.get(i, j) >= lower_bound.get(i,j));
      if (upper_bound.get(i, j) - lower_bound.get(i, j) > widest_range) {
        widest_range=upper_bound.get(i, j) - lower_bound.get(i, j);
        widest_range_i=i;
        widest_range_j=j;
      }
    }
  }
  left_lower_bound->Copy(lower_bound);
  left_upper_bound->Copy(upper_bound);
  right_lower_bound->Copy(lower_bound);
  right_upper_bound->Copy(upper_bound);
 
  for(index_t i=0; i<left_upper_bound->n_rows(); i++) {
    double split_value=(upper_bound.get(i, widest_range_j)
        +lower_bound.get(i, widest_range_j))/2;

    left_upper_bound->set(i, widest_range_j, split_value);
    right_lower_bound->set(i, widest_range_j, split_value); 
  }
 
}


void GopNmfEngine::PreprocessData(Matrix &data_mat) {
  values_.Init();
	rows_.Init();
	columns_.Init();
	for(index_t i=0; i<data_mat.n_rows(); i++) {
  	for(index_t j=0; j<data_mat.n_cols(); j++) {
		  values_.PushBackCopy(data_mat.get(i, j));
			rows_.PushBackCopy(i);
			columns_.PushBackCopy(j);
		}
	}
  num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
  num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
  x_lower_bound_.Init(new_dim_, num_of_rows_+num_of_columns_);
  x_upper_bound_.Init(new_dim_, num_of_rows_+num_of_columns_);
  Vector l1_norms;
  l1_norms.Init(num_of_columns_);
  l1_norms.SetAll(0.0);
  for(index_t i=0; i<num_of_columns_; i++) {
    for(index_t j=0; j<num_of_rows_; j++) {
      l1_norms[i]+=data_mat.get(j, i);
    }
  }
  x_lower_bound_.SetAll(-10.0);
  
  for(index_t i=0; i<num_of_rows_; i++) {
    for(index_t j=0; j<new_dim_; j++) {
      x_upper_bound_.set(j, i, log(1.0));
    }
  }
 
  for(index_t i=num_of_rows_; i<num_of_rows_+num_of_columns_; i++) {
    for(index_t j=0; j<new_dim_; j++) {
      x_upper_bound_.set(j, i, log(l1_norms[i-num_of_rows_]));
    }
  }
 
/*  
  x_lower_bound_.set(0,0, log(1)-5);
  x_upper_bound_.set(0,0, log(1)+5);
  x_lower_bound_.set(1,0, log(2)-5);
  x_upper_bound_.set(1,0, log(2)+5);
  x_lower_bound_.set(0,1, log(3)-5);
  x_upper_bound_.set(0,1, log(3)+5);
  x_lower_bound_.set(1,1, log(4)-5);
  x_upper_bound_.set(1,1, log(4)+5);
  x_lower_bound_.set(0,2, log(1)-5);
  x_upper_bound_.set(0,2, log(1)+5);
  x_lower_bound_.set(1,2, log(2)-5);
  x_upper_bound_.set(1,2, log(2)+5);
  x_lower_bound_.set(0,3, log(2)-5);
  x_upper_bound_.set(0,3, log(2)+5);
  x_lower_bound_.set(1,3, log(1)-5);
  x_upper_bound_.set(1,3, log(1)+5);
*/
/*
   x_lower_bound_.set(0,0, log(math::Random(0.1,1))-3);
  x_upper_bound_.set(0,0, log(math::Random(0.1,1))+3);
  x_lower_bound_.set(1,0, log(math::Random(0.1,1))-3);
  x_upper_bound_.set(1,0, log(math::Random(0.1,1))+3);
  x_lower_bound_.set(0,1, log(math::Random(0.1,1))-3);
  x_upper_bound_.set(0,1, log(math::Random(0.1,1))+3);
  x_lower_bound_.set(1,1, log(math::Random(0.1,1))-3);
  x_upper_bound_.set(1,1, log(math::Random(0.1,1))+3);
  x_lower_bound_.set(0,2, log(math::Random(0.1,1))-3);
  x_upper_bound_.set(0,2, log(math::Random(0.1,1))+3);
  x_lower_bound_.set(1,2, log(math::Random(0.1,1))-3);
  x_upper_bound_.set(1,2, log(math::Random(0.1,1))+3);
  x_lower_bound_.set(0,3, log(math::Random(0.1,1))-3);
  x_upper_bound_.set(0,3, log(math::Random(0.1,1))+3);
  x_lower_bound_.set(1,3, log(math::Random(0.1,1))-3);
  x_upper_bound_.set(1,3, log(math::Random(0.1,1))+3);
 */
}

double GopNmfEngine::ComputeVolume(Matrix &lower_bound, Matrix &upper_bound) {
  double volume=1.0;
  DEBUG_ASSERT(lower_bound.n_rows()==upper_bound.n_rows());
  DEBUG_ASSERT(lower_bound.n_cols()==upper_bound.n_cols());

  for(index_t i=0; i<lower_bound.n_rows(); i++) {
    for(index_t j=0; j<lower_bound.n_cols(); j++){
      DEBUG_ASSERT(lower_bound.get(i, j) <= upper_bound.get(i,j));
      volume*=upper_bound.get(i, j)-lower_bound.get(i,j);
    }
  }
  return volume;
}

