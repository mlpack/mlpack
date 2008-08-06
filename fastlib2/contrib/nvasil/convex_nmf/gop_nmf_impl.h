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
  num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
  columns_.InitCopy(columns);
  num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
  h_offset_=num_of_rows_;
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
    index_t w=row;
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
  *objective=values_sq_norm_;
  for(index_t i=0; i<values_.size(); i++) {
    index_t row=rows_[i];
    index_t col=columns_[i];
    index_t w=row;
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
}

void RelaxedNmf::ComputeNonRelaxedObjective(Matrix &coordinates, 
                                            double *objective) {
  *objective=values_sq_norm_;
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
  if (norm_gradient*step < grad_tolerance_) {
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
  if (norm_gradient*step < grad_tolerance_) {
    return true;
  }
  return false;
}

double RelaxedNmf::GetSoftLowerBound() {
  return soft_lower_bound_;
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
  fx_set_param_int(l_bfgs_module_, "mem_bfgs", 3);
  fx_set_param_bool(l_bfgs_module_, "use_default_termination", false);
  fx_set_param_bool(l_bfgs_module_, "silent", true);
}

void GopNmfEngine::ComputeGlobalOptimum() {
  
  soft_prunes_=0;
  hard_prunes_=0;
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
  upper_solution_.first=new_upper_global_optimum;
  upper_solution_.second.Own(optimizer.coordinates());
  
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

    optimizer.Destruct();
    opt_fun.Destruct();
    // optimize left
    opt_fun.Init(rows_, columns_, values_,  
                 new_dim_, grad_tolerance_,
                 left_lower_bound, left_upper_bound); 
    optimizer.Init(&opt_fun, l_bfgs_module_);
    if (opt_fun.GetSoftLowerBound() > upper_solution_.first) {
      soft_prunes_++;
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
      double new_lower_global_optimum;
      opt_fun.ComputeObjective(*optimizer.coordinates(), &new_lower_global_optimum);
      if (new_lower_global_optimum < upper_solution_.first) {
        SolutionPack pack;
        optimizer.CopyCoordinates(&pack.solution_);
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
    optimizer.Destruct();
    opt_fun.Destruct();
    init_data.Destruct();
    opt_fun.Init(rows_, columns_, values_,  
                 new_dim_, grad_tolerance_, right_lower_bound, right_upper_bound); 
    optimizer.Init(&opt_fun, l_bfgs_module_);
    opt_fun.GiveInitMatrix(&init_data);
    optimizer.set_coordinates(init_data);
    if (opt_fun.GetSoftLowerBound() > upper_solution_.first) {
      soft_prunes_++;
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
      double new_lower_global_optimum;
      opt_fun.ComputeObjective(*optimizer.coordinates(), &new_lower_global_optimum);
      if (new_lower_global_optimum < upper_solution_.first) {
        SolutionPack pack;
        optimizer.CopyCoordinates(&pack.solution_);
        pack.box_.first.Own(&right_lower_bound);
        pack.box_.second.Own(&right_upper_bound);
        lower_solution_.insert(
            std::make_pair(new_lower_global_optimum,pack));
      } else {
        hard_prunes_++;
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
      NOTIFY("Algorithm converged global optimum found");
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
  x_lower_bound_.SetAll(-0.1);
  x_upper_bound_.Init(new_dim_, num_of_rows_+num_of_columns_);
  x_upper_bound_.SetAll(
     log(*std::max_element(values_.begin(), values_.end())+1e-10));
  x_upper_bound_.SetAll(1.5);
 /* 
  x_lower_bound_.set(0,0, log(1)-15);
  x_upper_bound_.set(0,0, log(1)+15);
  x_lower_bound_.set(1,0, log(2)-15);
  x_upper_bound_.set(1,0, log(2)+15);
  x_lower_bound_.set(0,1, log(3)-15);
  x_upper_bound_.set(0,1, log(3)+15);
  x_lower_bound_.set(1,1, log(4)-15);
  x_upper_bound_.set(1,1, log(4)+15);
  x_lower_bound_.set(0,2, log(1)-15);
  x_upper_bound_.set(0,2, log(1)+15);
  x_lower_bound_.set(1,2, log(2)-15);
  x_upper_bound_.set(1,2, log(2)+15);
  x_lower_bound_.set(0,3, log(2)-15);
  x_upper_bound_.set(0,3, log(2)+15);
  x_lower_bound_.set(1,3, log(1)-15);
  x_upper_bound_.set(1,3, log(1)+15);
*/
  x_lower_bound_.set(0,0, log(math::Random(0.1,1))-5);
  x_upper_bound_.set(0,0, log(math::Random(0.1,1))+5);
  x_lower_bound_.set(1,0, log(math::Random(0.1,1))-5);
  x_upper_bound_.set(1,0, log(math::Random(0.1,1))+5);
  x_lower_bound_.set(0,1, log(math::Random(0.1,1))-5);
  x_upper_bound_.set(0,1, log(math::Random(0.1,1))+5);
  x_lower_bound_.set(1,1, log(math::Random(0.1,1))-5);
  x_upper_bound_.set(1,1, log(math::Random(0.1,1))+5);
  x_lower_bound_.set(0,2, log(math::Random(0.1,1))-5);
  x_upper_bound_.set(0,2, log(math::Random(0.1,1))+5);
  x_lower_bound_.set(1,2, log(math::Random(0.1,1))-5);
  x_upper_bound_.set(1,2, log(math::Random(0.1,1))+5);
  x_lower_bound_.set(0,3, log(math::Random(0.1,1))-5);
  x_upper_bound_.set(0,3, log(math::Random(0.1,1))+5);
  x_lower_bound_.set(1,3, log(math::Random(0.1,1))-5);
  x_upper_bound_.set(1,3, log(math::Random(0.1,1))+5);

}

