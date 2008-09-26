/*
 * =====================================================================================
 * 
 *       Filename:  l_bfgs_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  03/04/2008 10:31:56 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
const fx_entry_doc lbfgs_entries[] = {
  {"num_of_points", FX_PARAM, FX_INT, NULL,
   "  The number of points for the optimization variable.\n"},
  {"sigma", FX_PARAM, FX_DOUBLE, NULL,
   "  The initial penalty parameter on the augmented lagrangian.\n"},
  {"objective_factor", FX_PARAM, FX_DOUBLE, NULL,
   "  obsolete.\n"},
  {"eta", FX_PARAM, FX_DOUBLE, NULL,
   "  wolfe parameter.\n"},
  {"gamma", FX_PARAM, FX_DOUBLE, NULL,
   "  sigma increase rate, after inner loop is done sigma is multiplied by gamma.\n"},
  {"new_dimension", FX_PARAM, FX_INT, NULL,
   "  The dimension of the points\n"},
  {"desired_feasibility", FX_PARAM, FX_DOUBLE, NULL,
   "  Since this is used with augmented lagrangian, we need to know "
     "when the  feasibility is sufficient.\n"},
  {"feasibility_tolerance", FX_PARAM, FX_DOUBLE, NULL,
   "  if the feasibility is not improved by that quantity, then it stops.\n"},
  {"wolfe_sigma1", FX_PARAM, FX_DOUBLE, NULL,
   "  wolfe parameter.\n"},
  {"wolfe_sigma2", FX_PARAM, FX_DOUBLE, NULL,
   "  wolfe parameter.\n"},
  {"min_beta", FX_PARAM, FX_DOUBLE, NULL,
   "  wolfe parameter.\n"},
  {"wolfe_beta", FX_PARAM, FX_DOUBLE, NULL,
   "  wolfe parameter.\n"},
  {"step_size", FX_PARAM, FX_DOUBLE, NULL,
   "  Initial step size for the wolfe search.\n"},
  {"silent", FX_PARAM, FX_BOOL, NULL,
   "  if true then it doesn't emmit updates.\n"},
  {"use_default_termination", FX_PARAM, FX_BOOL, NULL,
   "  let this module decide where to terminate. If false then"
   " the objective function decides .\n"},
  {"norm_grad_tolerance", FX_PARAM, FX_DOUBLE, NULL,
   "  If the norm of the gradient doesn't change more than "
     "this quantity between two iterations and the use_default_termination "
     "is set, the algorithm terminates.\n"},
  {"max_iterations", FX_PARAM, FX_INT, NULL,
   "  maximum number of iterations required.\n"},
  {"mem_bfgs", FX_PARAM, FX_INT, NULL,
   "  the limited memory of BFGS.\n"},
  {"log_file", FX_PARAM, FX_STR, NULL,
   " file to log the output.\n"},
  {"iterations", FX_RESULT, FX_INT, NULL,
   "  iterations until convergence.\n"},
  {"feasibility_error", FX_RESULT, FX_DOUBLE, NULL,
   "  the fesibility error achived by termination.\n"},
  {"final_sigma", FX_RESULT, FX_DOUBLE, NULL,
   "  the last penalty parameter used\n"},
  {"objective", FX_RESULT, FX_DOUBLE, NULL,
   "  the objective achieved by termination.\n"},
  {"wolfe_step", FX_TIMER, FX_CUSTOM, NULL,
   "  Time spent computing the wolfe step.\n"},
  {"bfgs_step", FX_TIMER, FX_CUSTOM, NULL,
   "  Time spent computing the bfgs step.\n"},
  {"update_bfgs", FX_TIMER, FX_CUSTOM, NULL,
   "  Time spent computing the bfgs updating.\n"},

  FX_ENTRY_DOC_DONE
};

const fx_module_doc lbfgs_doc = {
  lbfgs_entries, NULL,
  "The LBFGS module for optimization.\n"
};

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::Init(OptimizedFunction *optimized_function, 
    datanode* module) {
  optimized_function_ = optimized_function;
  module_=module;
  num_of_points_ = fx_param_int(module_, "num_of_points", 1000);
  sigma_ = fx_param_double(module_, "sigma", 10);
  objective_factor_ = fx_param_double(module_, "objective_factor", 1.0);
  eta_ = fx_param_double(module_, "eta", 0.99);
  gamma_ = fx_param_double(module_, "gamma", 5);
  new_dimension_ = fx_param_int(module_, "new_dimension", 2);
  feasibility_tolerance_ = fx_param_double(module_, "feasibility_tolerance", 0.01);
  desired_feasibility_ = fx_param_double(module_, "desired_feasibility", 100); 
  wolfe_sigma1_ = fx_param_double(module_, "wolfe_sigma1", 0.1);
  wolfe_sigma2_ = fx_param_double(module_, "wolfe_sigma2", 0.9);
  step_size_=fx_param_double(module_, "step_size", 3.0);
  silent_=fx_param_bool(module_, "silent", false);
  use_default_termination_=fx_param_bool(module_, "use_default_termination", true);
  if (unlikely(wolfe_sigma1_>=wolfe_sigma2_)) {
    FATAL("Wolfe sigma1 %lg should be less than sigma2 %lg", 
        wolfe_sigma1_, wolfe_sigma2_);
  }
  DEBUG_ASSERT(wolfe_sigma1_>0);
  DEBUG_ASSERT(wolfe_sigma1_<1);
  DEBUG_ASSERT(wolfe_sigma2_>wolfe_sigma1_);
  DEBUG_ASSERT(wolfe_sigma2_<1);
  if (unlikely(wolfe_sigma1_>=wolfe_sigma2_)) {
    FATAL("Wolfe sigma1 %lg should be less than sigma2 %lg", 
        wolfe_sigma1_, wolfe_sigma2_);
  }
  DEBUG_ASSERT(wolfe_sigma1_>0);
  DEBUG_ASSERT(wolfe_sigma1_<1);
  DEBUG_ASSERT(wolfe_sigma2_>wolfe_sigma1_);
  DEBUG_ASSERT(wolfe_sigma2_<1);
  norm_grad_tolerance_ = fx_param_double(module_, "norm_grad_tolerance", 0.1); 
  wolfe_beta_   = fx_param_double(module_, "wolfe_beta", 0.8);
  min_beta_ = fx_param_double(module_, "min_beta", 1e-40);
  max_iterations_ = fx_param_int(module_, "max_iterations", 10000);
  // the memory of bfgs 
  mem_bfgs_ = fx_param_int(module_, "mem_bfgs", 20);
  std::string log_file=fx_param_str(module_, "log_file", "opt_log");
  fp_log_=fopen(log_file.c_str(), "w");
  optimized_function_->set_sigma(sigma_);
  InitOptimization_();
}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::Destruct() {
 fclose(fp_log_);
  double objective;
  double feasibility_error;
  optimized_function_->ComputeFeasibilityError(coordinates_,
      &feasibility_error); 
  optimized_function_->ComputeObjective(coordinates_, &objective);
  fx_result_int(module_, "iterations", num_of_iterations_);
  fx_result_double(module_, "feasibility_error", feasibility_error);
  fx_result_double(module_, "final_sigma", sigma_);
  fx_result_double(module_, "objective", objective);
  s_bfgs_.Renew();
  y_bfgs_.Renew();
  ro_bfgs_.Destruct();
  coordinates_.Destruct();
  previous_coordinates_.Destruct();
  gradient_.Destruct();
  previous_gradient_.Destruct();

}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::ComputeLocalOptimumBFGS() {
  double feasibility_error;
  if (silent_==false) {
    NOTIFY("Starting optimization ...\n");
    // Run a few iterations with gradient descend to fill the memory of BFGS
    NOTIFY("Initializing BFGS");
  }
   index_bfgs_=0;
  // You have to compute also the previous_gradient_ and previous_coordinates_
  // tha are needed only by BFGS
  optimized_function_->ComputeGradient(coordinates_, &gradient_);
  previous_gradient_.CopyValues(gradient_);
  previous_coordinates_.CopyValues(coordinates_);
  ComputeWolfeStep_(&step_, gradient_);
  optimized_function_->ComputeGradient(coordinates_, &gradient_);
  la::SubOverwrite(previous_coordinates_, coordinates_, &s_bfgs_[0]);
  la::SubOverwrite(previous_gradient_, gradient_, &y_bfgs_[0]);
  ro_bfgs_[0] = la::Dot(s_bfgs_[0].n_elements(), 
      s_bfgs_[0].ptr(), y_bfgs_[0].ptr());
  double old_feasibility_error = feasibility_error;
  for(index_t i=0; i<mem_bfgs_; i++) {
    success_t success=ComputeBFGS_(&step_, gradient_, i);
    if (success==SUCCESS_FAIL) {
      if (silent_==false) {
        NOTIFY("LBFGS failed to find a direction, continuing with gradient descent\n");
      }
      ComputeWolfeStep_(&step_, gradient_);
    }
    optimized_function_->ComputeGradient(coordinates_, &gradient_);
    UpdateBFGS_();
    previous_gradient_.CopyValues(gradient_);
    previous_coordinates_.CopyValues(coordinates_);
    num_of_iterations_++;
    if (silent_==false) {
        ReportProgressFile_();
    }
/*    if (use_default_termination_== true) {
       optimized_function_->ComputeFeasibilityError(coordinates_,
          &feasibility_error); 
      if (feasibility_error < desired_feasibility_) {
        NOTIFY("feasibility error %lg less than desired feasibility %lg", 
            feasibility_error, desired_feasibility_);
        return;
      }
    } else {
      if (optimized_function_->IsOptimizationOver(
            coordinates_, gradient_, step_)==true) {
        return;
      }
    }
*/
  }
  if (silent_==false) {
    NOTIFY("Now starting optimizing with BFGS...\n");
  }
  for(index_t it1=0; it1<max_iterations_; it1++) {  
    for(index_t it2=0; it2<max_iterations_; it2++) {
      success_t success_bfgs = ComputeBFGS_(&step_, gradient_, mem_bfgs_); 
      optimized_function_->ComputeGradient(coordinates_, &gradient_);
      optimized_function_->ComputeFeasibilityError(coordinates_, 
          &feasibility_error);
      double norm_grad = la::Dot(gradient_.n_elements(), 
          gradient_.ptr(), gradient_.ptr());
      num_of_iterations_++;
      if (success_bfgs==SUCCESS_FAIL){
        // NOTIFY("LBFGS failed to find a direction, continuing with gradient descent\n");
        // if  (ComputeWolfeStep_(&step_, gradient_)==SUCCESS_FAIL) {
        //   NONFATAL("Gradient descent failed too");
        // }
        //break;
      }
      if (silent_==false) {
        ReportProgressFile_();
      }
      // NOTIFY("feasibility_error:%lg desired_feasibility:%lg", feasibility_error, desired_feasibility_);
      if (use_default_termination_==true) {
        if (feasibility_error < desired_feasibility_) {
          break;
        }
        if (step_*norm_grad/sigma_ < norm_grad_tolerance_) {
          break;
        }
      } else {
        if (optimized_function_->IsIntermediateStepOver(
              coordinates_, gradient_, step_)==true) {
          break;
        }
      }
      UpdateBFGS_();
      previous_coordinates_.CopyValues(coordinates_);
      previous_gradient_.CopyValues(gradient_);
      // Do this check to make sure the method has not started diverging
      double objective;
      optimized_function_->ComputeObjective(coordinates_, &objective);
      if (optimized_function_->IsDiverging(objective)) {
        sigma_*=gamma_;
        optimized_function_->set_sigma(sigma_);
        optimized_function_->ComputeGradient(coordinates_, &gradient_);
       // break;
      }
    }
    
    if (silent_==false) {
      NOTIFY("Inner loop done, increasing sigma...");
    }

    if (use_default_termination_==true) {
      if (fabs(old_feasibility_error - feasibility_error)
          /(old_feasibility_error+1e-20) < feasibility_tolerance_ ||
          feasibility_error < desired_feasibility_) {
        break;
      }
    } else {
      if (optimized_function_->IsOptimizationOver(
            coordinates_, gradient_, step_)==true) {
        break;
      }
    }
    old_feasibility_error = feasibility_error;
    UpdateLagrangeMult_();
    optimized_function_->ComputeGradient(coordinates_, &gradient_);
  }

}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::CopyCoordinates(Matrix *result) {
  result->Copy(coordinates_);
}

template<typename OptimizedFunction>
void  LBfgs<OptimizedFunction>::set_coordinates(Matrix &coordinates) {
  coordinates_.CopyValues(coordinates);
}
template<typename OptimizedFunction>
Matrix *LBfgs<OptimizedFunction>::coordinates() {
  return &coordinates_;
}

template<typename OptimizedFunction>
double LBfgs<OptimizedFunction>::sigma() {
  return sigma_;
}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::set_sigma(double sigma) {
  sigma_=sigma;
  optimized_function_->set_sigma(sigma_);
}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::Reset() {
  sigma_ = fx_param_double(module_, "sigma", 10);
  optimized_function_->set_sigma(sigma_);
}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::set_max_iterations(index_t max_iterations) {
  max_iterations_=max_iterations;
}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::InitOptimization_() {
  if (unlikely(new_dimension_<0)) {
    FATAL("You forgot to set the new dimension\n");
  }
  if (silent_==false) {
    NOTIFY("Initializing optimization ...\n");
  }
  optimized_function_->GiveInitMatrix(&coordinates_);
  num_of_points_=coordinates_.n_cols();
  previous_coordinates_.Init(new_dimension_, num_of_points_);
  gradient_.Init(new_dimension_, num_of_points_);
  previous_gradient_.Init(new_dimension_, num_of_points_);
  for(index_t i=0; i< coordinates_.n_rows(); i++) {
    for(index_t j=0; j<coordinates_.n_cols(); j++) {
      coordinates_.set(i, j, math::Random(0.1, 1.0));
    }
  }
  optimized_function_->Project(&coordinates_);
  if (unlikely(mem_bfgs_<0)) {
    FATAL("You forgot to initialize the memory for BFGS\n");
  }
  // Init the memory for BFGS
  s_bfgs_.Init(mem_bfgs_);
  y_bfgs_.Init(mem_bfgs_);
  ro_bfgs_.Init(mem_bfgs_);
  ro_bfgs_.SetAll(0.0);
  for(index_t i=0; i<mem_bfgs_; i++) {
    s_bfgs_[i].Init(new_dimension_, num_of_points_);
    y_bfgs_[i].Init(new_dimension_, num_of_points_);
  } 
  num_of_iterations_=0;
}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::UpdateLagrangeMult_() {
  optimized_function_->UpdateLagrangeMult(coordinates_);
  sigma_*=gamma_;
  optimized_function_->set_sigma(sigma_);
}

// for optimization purposes the direction  is always the negative of what it is supposed 
// in the wolfe form. so for example if the direction is the negative gradient the direction
// should be the gradient and not the -gradient
template<typename OptimizedFunction>
success_t LBfgs<OptimizedFunction>::ComputeWolfeStep_(double *step, Matrix &direction) {
  fx_timer_start(module_, "wolfe_step");
  success_t success=SUCCESS_PASS;
  Matrix temp_coordinates;
  Matrix temp_gradient;
  temp_gradient.Init(new_dimension_, num_of_points_);
  temp_coordinates.Init(coordinates_.n_rows(), coordinates_.n_cols()); 
  double lagrangian1 = optimized_function_->ComputeLagrangian(coordinates_);
  double lagrangian2 = 0;
  double beta=wolfe_beta_;
  double dot_product = -la::Dot(direction.n_elements(),
                               gradient_.ptr(),
                               direction.ptr());
  double wolfe_factor =  dot_product * wolfe_sigma1_ * wolfe_beta_ * step_size_;
  for(index_t i=0; beta>min_beta_/(1.0+sigma_); i++) { 
    temp_coordinates.CopyValues(coordinates_);
    la::AddExpert(-step_size_*beta, direction, &temp_coordinates);
    optimized_function_->Project(&temp_coordinates); 
    lagrangian2 = optimized_function_->ComputeLagrangian(temp_coordinates);
    // NOTIFY("direction:%lg", la::Dot(direction.n_elements(), direction.ptr(), direction.ptr()) );
    // NOTIFY("step_size:%lg beta:%lg min_beta:%lg", step_size_, beta, min_beta_);
    // NOTIFY("********lagrangian2:%lg lagrangian1:%lg wolfe_factor:%lg", lagrangian2, lagrangian1, wolfe_factor);
    if (lagrangian2 <= lagrangian1 + wolfe_factor)  {
      optimized_function_->ComputeGradient(temp_coordinates, &temp_gradient);
      double dot_product_new = -la::Dot(temp_gradient.n_elements(), 
          temp_gradient.ptr(), direction.ptr());
    //  NOTIFY("dot_product_new:%lg wolfe_sigma2:%lg dot_product:%lg", dot_product_new,wolfe_sigma2_, dot_product);
      if (dot_product_new >= wolfe_sigma2_*dot_product) {
        success = SUCCESS_PASS;
      } else {
        success = SUCCESS_FAIL;
      }
      break;    
    }
    beta *=wolfe_beta_;
    wolfe_factor *=wolfe_beta_;
  }
 // optimized_function_->Project(&temp_coordinates); 

  if (beta<=min_beta_/(1.0+sigma_)) {
    *step=0;
    fx_timer_stop(module_, "wolfe_step");
    return SUCCESS_FAIL;
  } else {
    *step=step_size_*beta;
    coordinates_.CopyValues(temp_coordinates);   
    fx_timer_stop(module_, "wolfe_step");
    if (success==SUCCESS_FAIL) {
      return SUCCESS_FAIL;
    } else {
      return SUCCESS_PASS;
    }
  }
}

template<typename OptimizedFunction>
success_t LBfgs<OptimizedFunction>::ComputeBFGS_(double *step, Matrix &grad, index_t memory) {
  fx_timer_start(module_, "bfgs_step");
  Vector alpha;
  alpha.Init(mem_bfgs_);
  Matrix scaled_y;
  scaled_y.Init(new_dimension_, num_of_points_);
  index_t num=0;
  Matrix temp_direction(grad);
  for(index_t i=index_bfgs_, num=0; num<memory; i=(i+1+mem_bfgs_)%mem_bfgs_, num++) {
   // printf("i:%i  index_bfgs_:%i\n", i, index_bfgs_);
    alpha[i] = la::Dot(new_dimension_ * num_of_points_,
                       s_bfgs_[i].ptr(), 
                       temp_direction.ptr());
    alpha[i] *= ro_bfgs_[i];
    scaled_y.CopyValues(y_bfgs_[i]);
    la::Scale(alpha[i], &scaled_y);
    la::SubFrom(scaled_y, &temp_direction);
  }
  // We need to scale the gradient here
  double s_y = la::Dot(num_of_points_* 
                    new_dimension_, y_bfgs_[index_bfgs_].ptr(),
                    s_bfgs_[index_bfgs_].ptr());
  double y_y = la::Dot(num_of_points_* 
                   new_dimension_, y_bfgs_[index_bfgs_].ptr(),
                   y_bfgs_[index_bfgs_].ptr());
  if (unlikely(y_y<1e-40)){
    NONFATAL("Gradient differences close to singular...norm=%lg\n", y_y);
  } 
  double norm_scale=s_y/(y_y+1e-40);
  la::Scale(norm_scale, &temp_direction);
  Matrix scaled_s;
  double beta;
  scaled_s.Init(new_dimension_, num_of_points_);
  num=0;
  for(index_t j=(index_bfgs_+memory-1)%mem_bfgs_, num=0; num<memory; 
      num++, j=(j-1+mem_bfgs_)%mem_bfgs_) {
   // printf("j:%i  index_bfgs_:%i\n", j, index_bfgs_);

    beta = la::Dot(new_dimension_ * num_of_points_, 
                   y_bfgs_[j].ptr(),
                   temp_direction.ptr());
    beta *= ro_bfgs_[j];
    scaled_s.CopyValues(s_bfgs_[j]);
    la::Scale(alpha[j]-beta, &scaled_s);
    la::AddTo(scaled_s, &temp_direction);
  }
  success_t success=ComputeWolfeStep_(step, temp_direction);
/*  if (step==0) {
    NONFATAL("BFGS Failed looking in the other direction...\n");
    la::Scale(-1.0, &temp_direction);
    ComputeWolfeStep_(step, temp_direction);
    *step=-*step;
    fx_timer_stop(module_, "bfgs_step");
    return SUCCESS_FAIL;  
  }
*/  
  fx_timer_stop(module_, "bfgs_step");
  return success;
}

template<typename OptimizedFunction>
success_t LBfgs<OptimizedFunction>::UpdateBFGS_() {
  index_t try_index_bfgs = (index_bfgs_ - 1 + mem_bfgs_ ) % mem_bfgs_;
  if (UpdateBFGS_(try_index_bfgs)==SUCCESS_FAIL) {
    return SUCCESS_FAIL;
  } else {
    index_bfgs_=try_index_bfgs;
    return SUCCESS_PASS;
  }
}

template<typename OptimizedFunction>
success_t LBfgs<OptimizedFunction>::UpdateBFGS_(index_t index_bfgs) {
  fx_timer_start(module_, "update_bfgs");
  // shift all values
  Matrix temp_s_bfgs;
  Matrix temp_y_bfgs;
  la::SubInit(previous_coordinates_, coordinates_, &temp_s_bfgs); 
  la::SubInit(previous_gradient_, gradient_, &temp_y_bfgs); 
  double temp_ro=la::Dot(new_dimension_ * num_of_points_, 
                                  temp_s_bfgs.ptr(),
                                  temp_y_bfgs.ptr());
  double y_norm=la::Dot(temp_y_bfgs.n_elements(), 
      temp_y_bfgs.ptr(), temp_y_bfgs.ptr());
  if (temp_ro<1e-70*y_norm) {
    fx_timer_stop(module_, "update_bfgs");
    NONFATAL("Rejecting s, y they don't satisfy curvature condition "
        "s*y=%lg < 1e-70 *||y||^2=%lg", temp_ro,1e-70*y_norm);
    return SUCCESS_FAIL;
  } 
  s_bfgs_[index_bfgs].CopyValues(temp_s_bfgs);
  y_bfgs_[index_bfgs].CopyValues(temp_y_bfgs); 
  ro_bfgs_[index_bfgs] = 1.0/temp_ro;

  fx_timer_stop(module_, "update_bfgs");
  return SUCCESS_PASS;
}

/*
template<typename OptimizedFunction>
void ComputeCauchyPoint(Matrix &coordinates, Matrix &gradient, 
    Matrix *cauchy_point) {
  Matrix t_mat;
  Matrix d_mat;
  t_mat.Init(coordinates.n_rows(), coordinates.n_cols());
  d_mat.Init(coordinates.n_rows(), coordinates.n_cols());
  ArrayList<std::pair<double, index_t> > f_list;
  f_list.Init();

  for(index_t i=0; i<t_mat.n_rows(); i++) {
    for(index_t j=0; j<t_mat.n_cols(); j++) {
      double value;
      if (gradient.get(i, j)<0) {
       val=(coordinates.get(i,j)-get_upper_bound(i,j))/gradient.get(i,j);
      } else {
        if (gradient.get(i, j)>0) {
          val=(coordinates.get(i,j)-get_lower_bound(i,j))/gradient.get(i,j);
        } else {
          val=DBL_MAX;
        }
      }
      t_mat.set(i, j, val);
      if (unlikely(val==0.0)) {
        d_mat.set(i, j, 0.0);
      } else {
        d_mat.set(i, j, -gradient.get(i, j));
      }
      if (val>0) {
        f_list.PushBack(std::make_pair(val, i*new_dimension_+j));
      }
    }
  }
  //  Initializations
  Vector c_vec;
  c_vec.Init(mem_bfgs_);
  c_vec.SetAll(0.0);
  Vector p_vec;
  p_vec.Init(mem_bfgs_);
  index_t  n_elements=coordinates_.n_elements();
  double theta=ro_bfgs_[index_bfgs_];
  for(index_t i=0; i<mem_bfgs_; i++) {
    index_t bfgs_ind= (i+index_bfgs_) % mem_bfgs_;
    p_vec[i]=la::Dot(n_elements, y_bfgs_[bfgs_ind].ptr(), d_mat.ptr());
    p_vec[i+mem_bfgs_]=theta*la::Dot(n_elements, s_bfgs_[bfgs_ind].ptr(), d_mat.ptr());
  }
  Matrix m_mat;
  ComputeMMatrix_(&m_mat);
  double f_prime=-la::Dot(n_elements, d_mat.ptr(), d_mat.ptr());
  Vector temp_p;
  la::MulInit(m_mat, p_vec, &temp_p);
  double f_dprime=-f_prime-la::Dot(temp_p, p_vec);
  double delta_tau_min = -f_prime/(f_dprime+1e-20);
  double t_old=0;
  std::sort(f_list.begin(), f_list.end(), 
      std::greater<std::pair<double, index_t> >);
  double t_min=f_list.back().first;
  index_t b=f_list.back().second;
  f_list.PopBack();
  double delta_tau = t_min;
  while (delta_tau_min >= delta_tau) {
    index_t i= b%new_dimension_;
    index_t j= b/new_dimension_;
    if (d_mat.get(i, j)>0) {
      cauchy_point.set(i, j, get_upper_bound(i, j));
    } else {
       cauchy_point.set(i, j, get_lower_bound(i, j));    
    }
    double z_b=cauchy_point.get(i, j)-coordinates.get(i, j);
    la::AddExpert(delta_tau, p_vec, &c_vec);
    Vector temp_prod_vec;
    Vector wb_vec;
    wb_vec.Init(2*mem_bfgs_);
    // form the wb_vec;
    for(index_t i=0; i< mem_bfgs_; i++) {
      index_t bfgs_ind= (i+index_bfgs_) % mem_bfgs_;
      wb_vec[i]=y_bfgs_[bfgs_ind];
      wb_vec[i+mem_bfgs_]=y_bfgs_[bfgs_ind]*theta;
    }
    la::MulInit(m_mat, c_vec, &temp_prod_vec);
    f_prime=f_prime+delta_tau*f_dprime+gradient.get(i,j)*(gradient.get(i, j)
       +theta*z_b-la::Dot(temp_prod_vec, wb_vec));
    la::MulOverwrite(m_mat, p_vec, &temp_p);
    la::MulOverwrite(m_mat, wb_vec, &temp_prod_vec);
    f_dprime=f_dprime-gradient.get(i, j)*(theta*gradient.get(i, j)
        +2*gradient.get(i,  j)*la::Dot(wb_vec, temp_p)
        +gradient.get(i, j)*la::Dot(wb_vec, temp_prod_vec));
    la::AddExpert(gradient.get(i, j), wb_vec, p_vec);
    d_mat.set(i, j, 0.0);
    delta_tau_min=-f_prime/(1e-20+f_dprime);
    t_old=t_min;
    double t_min=f_list.back().first;
    index_t b=f_list.back().second;
    f_list.PopBack();
    delta_tau=t_min-t_old;
  }
  delta_tau_min=std::max(delta_tau_min, 0.0);
  t_old=t_old+delta_tau_min;
  for(index_t k=0; k<f_list.size(); k++) {
    index_t b=f_list[k].second;
    index_t i= b%new_dimension_;
    index_t j= b/new_dimension_;
    double val=coordinates.get(i, j)+t_old*d_mat.get(i, j);
    cauchy_point->set(i, j, val);
  }
  la::AddExpert(delta_tau_min, p_vec, &c_vec);
  //f_list is an  ArrayList with all the free variables
  
  // Computation of r_c
  Matrix rc_mat;
  rc_mat.Copy(gradient);
  rc_mat.AddExpert(*cacuchy_point, theta, &rc_mat);
  rc_mat.AddExpert(coordinates, -theta, &rc_mat);
  Vector temp_prod0;
  la::MulInit(m_mat, c_vec, &temp_prod0);
  for(index_t i=0; i< mem_bfgs_; i++) {
    index_t bfgs_ind = (i+index_bfgs_) % mem_bfgs_;
    la::AddExpert(-c_vec[i], y_bfgs_[bfgs_ind], &rc_mat);
    la::AddExpert(-c_vec[i]*theta, s_bfgs_[bfgs_ind], &rc_mat);
  }  
  for(index_t k=0; k<f_list.size(); k++) {
    index_t b=f_list[k].second;
    index_t i= b%new_dimension_;
    index_t j= b/new_dimension_;
    rc_mat.set(i, j, 0.0);  
  } 
  // the conjugate gradient method for the subspace minimization

}
*/
template<typename OptimizedFunction>
std::string LBfgs<OptimizedFunction>::ComputeProgress_() {
  double lagrangian=optimized_function_->ComputeLagrangian(coordinates_);
  double objective;
  optimized_function_->ComputeObjective(coordinates_, &objective);
  double feasibility_error;
  optimized_function_->ComputeFeasibilityError(
      coordinates_, &feasibility_error);
  double norm_grad=math::Pow<1,2>(la::Dot(gradient_.n_elements(), 
      gradient_.ptr(), gradient_.ptr()));
  char buffer[1024];
  sprintf(buffer, "iteration:%i sigma:%lg lagrangian:%lg objective:%lg error:%lg "
      "grad_norm:%lg step:%lg",
      num_of_iterations_, sigma_, lagrangian, objective, 
      feasibility_error, norm_grad, step_);
  return std::string(buffer);
}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::ReportProgressFile_() {
  std::string progress = ComputeProgress_();
  NOTIFY("%s\n", progress.c_str());
  fprintf(fp_log_, "%s\n", progress.c_str());
}
