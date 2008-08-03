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

void RelaxedNmf::Init(ArrayList<index_t> &rows,
    ArrayList<index_t> &columns,
    ArrayList<double> &values,
    index_t new_dim,
    double grad_tolerance,
    Matrix &x_lower_bound,
    Matrix &x_upper_bound) {
  
  grad_tolerance_=grad_tolerance;
  rows_.InitCopy(rows);
  num_of_rows_=*std::max_element(rows_.begin(), rows_.end());
  columns_.InitCopy(columns);
  num_of_columns_=*std::max(columns_.begin(), columns_.end());
  h_offset_=num_of_rows_;
  values_.InitCopy(values);
  x_lower_bound_.Copy(x_lower_bound);
  x_upper_bound_.Copy(x_upper_bound);
  soft_lower_bound_=0;
  // Generate the linear terms
  a_linear_term_.Init(new_dim_*values_.size());
  b_linear_term_.Init(new_dim_*values_.size());
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dim_; j++) {
      convex_part+=exp(x_lower_bound_.get(j, w)+x_lower_bound_.get(j, h));
    }
    soft_lower_bound_+=convex_part*convex_part;
    for(index_t j=0; j<new_dim_; j++) {
      double y_lower=x_lower_bound_.get(j, w) + x_lower_bound_.get(j, h);
      double y_upper=x_upper_bound_.get(j, w) + x_upper_bound_.get(i, h);      
      a_linear_term_[new_dim_*i+j]=
          (y_upper*exp(y_lower)-y_lower*exp(y_upper))/(y_upper-y_lower);
      b_linear_term_[new_dim_*i+j]=(exp(y_upper)-exp(y_lower))/
          (y_upper-y_lower);
      soft_lower_bound_+=-2*values_[i]*(
                             a_linear_term_[new_dim_*i+j]
                             +b_linear_term_[new_dim_*i+j]*y_upper);
    }
  }
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
  x_lower_bound_.Destruct();
  x_upper_bound_.Destruct();
  soft_lower_bound_=-DBL_MAX;
} 


void RelaxedNmf::ComputeGradient(Matrix &coordinates, Matrix *gradient) {
  gradient->SetAll(0.0);
  // gradient from the objective
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row;
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
}

void RelaxedNmf::ComputeObjective(Matrix &coordinates, double *objective) {
  *objective=0;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dim_; j++) {
      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    *objective+=convex_part*convex_part;
    for(index_t j=0; j<new_dim_; j++) {
      *objective+=-2*values_[i]*(a_linear_term_[new_dim_*i+j]
                                +b_linear_term_[new_dim_*i]
                                 *(coordinates.get(j, w)+coordinates.get(j, h)));
    }
  } 
}

void RelaxedNmf::ComputeNonRelaxedObjective(Matrix &coordinates, 
                                            double *objective) {
  *objective=0;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row;
    index_t h=col+h_offset_;
    double convex_part=0;
    for(index_t j=0; j<new_dim_; j++) {
      convex_part+=exp(coordinates.get(j, w)+coordinates.get(j, h));
    }
    *objective+=convex_part*(convex_part-2*values_[i]);
  } 
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
  double norm_gradient=la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr());
  if (norm_gradient < grad_tolerance_) {
    return true;
  }
  return false;
}

bool RelaxedNmf::IsIntermediateStepOver(Matrix &coordinates, 
                                        Matrix &gradient, 
                                        double step) {
  double norm_gradient=la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr());
  if (norm_gradient < grad_tolerance_) {
    return true;
  }
  return false;
}

double RelaxedNmf::GetSoftLowerBound() {
  return soft_lower_bound_;
}

//////////////////////////////////////////////////////////////////////
//GlobalNmfEngine/////////////////////////////////////////////////////
void GopNmfEngine::Init(fx_module *module, Matrix &data_points) {
  module_=module;
  l_bfgs_module_=fx_submodule(module_, "l_bfgs");
  PreprocessData(data_points);
  new_dim_ = fx_param_int(module_, "new_dim", 5);
  grad_tolerance_ = fx_param_double(module_, "grad_tolerance", 1e-3);
  desired_global_optimum_gap_= fx_param_double(module_, "opt_gap", 1e-1);
}

void GopNmfEngine::ComputeGlobalOptimum() {
  
  prunes_=0;
  // Solve for  this bounding box
  Matrix lower_bound;
  Matrix upper_bound;
  lower_bound.Copy(x_lower_bound_);
  upper_bound.Copy(x_upper_bound_);
  
  RelaxedNmf opt_fun;
  opt_fun.Init(rows_, columns_, values_,  
               new_dim_, grad_tolerance_,
               lower_bound, upper_bound); 
  Optimizer optimizer;
  optimizer.Init(&opt_fun, l_bfgs_module_);
  Matrix init_data;
  opt_fun.GiveInitMatrix(&init_data);
  optimizer.set_coordinates(init_data);
  optimizer.ComputeLocalOptimumBFGS();
  double new_upper_global_optimum;
  opt_fun.ComputeNonRelaxedObjective(init_data, &new_upper_global_optimum);
  double new_lower_global_optimum;
  opt_fun.ComputeObjective(init_data, &new_lower_global_optimum);
  if (new_lower_global_optimum > upper_solution_.first) {
    return;
  }
  if (new_upper_global_optimum <upper_solution_.first) {
    upper_solution_.first=new_upper_global_optimum;
    upper_solution_.second.Destruct();
    upper_solution_.second.Own(optimizer.coordinates());
  }
  SolutionPack pack;
  pack.solution_.Copy(*optimizer.coordinates());
  pack.box_.first.Own(&upper_bound);
  pack.box_.second.Own(&lower_bound);
  lower_solution_.insert(
      std::make_pair(new_lower_global_optimum,pack));
  
  while (true) {
    Matrix left_upper_bound;
    Matrix left_lower_bound;
    Matrix right_upper_bound;
    Matrix right_lower_bound;
    
    Split(lower_bound, upper_bound, 
          &left_lower_bound, &left_upper_bound,
          &right_lower_bound, &right_upper_bound);

    opt_fun.Destruct();
    optimizer.Destruct();
    // optimize left
    opt_fun.Init(rows_, columns_, values_,  
                 new_dim_, grad_tolerance_,
                 left_lower_bound, left_upper_bound); 
    optimizer.Init(&opt_fun, l_bfgs_module_);
    if (opt_fun.GetSoftLowerBound() > upper_solution_.first) {
      prunes_++;
    } else {
      init_data.Destruct();
      opt_fun.GiveInitMatrix(&init_data);
      optimizer.set_coordinates(init_data);
      optimizer.ComputeLocalOptimumBFGS();
      double new_upper_global_optimum;
      opt_fun.ComputeNonRelaxedObjective(init_data, &new_upper_global_optimum);
      if (new_upper_global_optimum <upper_solution_.first) {
        upper_solution_.first=new_upper_global_optimum;
        upper_solution_.second.Destruct();
        upper_solution_.second.Own(optimizer.coordinates());
      }
      SolutionPack pack;
      pack.solution_.Own(optimizer.coordinates());
      pack.box_.first.Own(&upper_bound);
      pack.box_.second.Own(&lower_bound);
      lower_solution_.insert(
          std::make_pair(new_lower_global_optimum,pack));
    }

    // optimize right
    opt_fun.Destruct();
    optimizer.Destruct();
    init_data.Destruct();
    opt_fun.Init(rows_, columns_, values_,  
                 new_dim_, grad_tolerance_, right_lower_bound, right_upper_bound); 
    optimizer.Init(&opt_fun, l_bfgs_module_);
    opt_fun.GiveInitMatrix(&init_data);
    optimizer.set_coordinates(init_data);
    if (opt_fun.GetSoftLowerBound() > upper_solution_.first) {
      prunes_++;
    } else {
      init_data.Destruct();
      opt_fun.GiveInitMatrix(&init_data);
      optimizer.set_coordinates(init_data);
      optimizer.ComputeLocalOptimumBFGS();
      double new_upper_global_optimum;
      opt_fun.ComputeNonRelaxedObjective(init_data, &new_upper_global_optimum);
      if (new_upper_global_optimum <upper_solution_.first) {
        upper_solution_.first=new_upper_global_optimum;
        upper_solution_.second.Destruct();
        upper_solution_.second.Own(optimizer.coordinates());
      }
      SolutionPack pack;
      pack.solution_.Own(optimizer.coordinates());
      pack.box_.first.Own(&upper_bound);
      pack.box_.second.Own(&lower_bound);
      lower_solution_.insert(
          std::make_pair(new_lower_global_optimum,pack));
    }
     
    // Check for convergence
    DEBUG_ASSERT(upper_solution_.first - lower_solution_.begin()->first); 
    if (upper_solution_.first - lower_solution_.begin()->first) {
      NOTIFY("Algorithm converged global optimum found");
      fx_result_double(module_, "upper_global_optimum", upper_solution_.first);
      fx_result_double(module_, "lower_global_optimum", lower_solution_.begin()->first);
      fx_result_int(module_, "prunes", prunes_);
      return;
    }    
    
    // choose next box to split and work    
    lower_bound.Own(&lower_solution_.begin()->second.box_.first);
    upper_bound.Own(&lower_solution_.begin()->second.box_.second);  
    lower_solution_.erase(lower_solution_.begin()); 
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
  x_lower_bound_.Init(new_dim_, num_of_rows_+num_of_columns_);
  x_lower_bound_.SetAll(0.0);
  x_upper_bound_.Init(new_dim_, num_of_rows_+num_of_columns_);
  x_upper_bound_.SetAll(0.0);
	for(index_t i=0; i<data_mat.n_rows(); i++) {
  	for(index_t j=0; i<data_mat.n_cols(); i++) {
		  values_.PushBackCopy(data_mat.get(i, j));
			rows_.PushBackCopy(i);
			columns_.PushBackCopy(j);
		}
	}
  x_upper_bound_.SetAll(
      *std::max_element(values_.begin(), values_.end()));
  num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
  num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
}

