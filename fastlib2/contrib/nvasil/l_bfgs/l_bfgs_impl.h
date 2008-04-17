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
  new_dimension_ = fx_param_int(module, "new_dimension", 2);
  feasibility_tolerance_ = fx_param_double(module_, "feasibility_tolerance", 0.01);
  desired_feasibility_ = fx_param_double(module_, "desired_feasibility", 100); 
  wolfe_sigma1_ = fx_param_double(module_, "wolfe_sigma1", 0.1);
  wolfe_sigma2_ = fx_param_double(module_, "wolfe_sigma2", 0.9);
  step_size_=fx_param_double(module_, "step_size", 3.0);
  silent_=fx_param_bool(module_, "silent", false);
  if (unlikely(wolfe_sigma1_>=wolfe_sigma2_)) {
    FATAL("Wolfe sigma1 %lg should be less than sigma2 %lg", 
        wolfe_sigma1_, wolfe_sigma2_);
  }
  DEBUG_ASSERT(wolfe_sigma1_>0);
  DEBUG_ASSERT(wolfe_sigma1_<1);
  DEBUG_ASSERT(wolfe_sigma2_>0);
  DEBUG_ASSERT(wolfe_sigma2_<1);
  if (unlikely(wolfe_sigma1_>=wolfe_sigma2_)) {
    FATAL("Wolfe sigma1 %lg should be less than sigma2 %lg", 
        wolfe_sigma1_, wolfe_sigma2_);
  }
  DEBUG_ASSERT(wolfe_sigma1_>0);
  DEBUG_ASSERT(wolfe_sigma1_<1);
  DEBUG_ASSERT(wolfe_sigma2_>0);
  DEBUG_ASSERT(wolfe_sigma2_<1);
  norm_grad_tolerance_ = fx_param_double(module_, "norm_grad_tolerance", 0.1); 
  wolfe_beta_   = fx_param_double(module_, "wolfe_beta", 0.8);
  min_beta_ = fx_param_double(module_, "min_beta", 1e-20);
  max_iterations_ = fx_param_int(module_, "max_iterations", 10000);
  // the memory of bfgs 
  mem_bfgs_ = fx_param_int(module_, "mem_bfgs", 120);
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
  fx_format_result(module_, "iterations", "%i", num_of_iterations_);
  fx_format_result(module_, "feasibility_error", "%lg", feasibility_error);
  fx_format_result(module_, "final_sigma", "%lg", sigma_);
  fx_format_result(module_, "objective","%lg", objective);
}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::ComputeLocalOptimumBFGS() {
  double feasibility_error;

  NOTIFY("Starting optimization ...\n");
  // Run a few iterations with gradient descend to fill the memory of BFGS
  NOTIFY("Initializing BFGS");
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
    ComputeBFGS_(&step_, gradient_, i);
    if (step_==0) {
      NOTIFY("LBFGS failed to find a direction, continuing with gradient descent\n");
      ComputeWolfeStep_(&step_, gradient_);
    }
//    if (success=SUCCESS_FAIL) {
//     NOTIFY("LBFGS failed to find a direction, continuing with gradient descent\n");
//      ComputeWolfeStep_(&step_, gradient_);
//      UpdateBFGS_();
//    }
    optimized_function_->ComputeGradient(coordinates_, &gradient_);
    UpdateBFGS_();
    previous_gradient_.CopyValues(gradient_);
    previous_coordinates_.CopyValues(coordinates_);
    optimized_function_->ComputeFeasibilityError(coordinates_,
        &feasibility_error); 
    num_of_iterations_++;
    if (silent_==false) {
        ReportProgressFile_();
    }
    if (feasibility_error < desired_feasibility_) {
      return;
    }
  }
 
  NOTIFY("Now starting optimizing with BFGS...\n");
  for(index_t it1=0; it1<max_iterations_; it1++) {  
    for(index_t it2=0; it2<max_iterations_; it2++) {
      success_t success_bfgs = ComputeBFGS_(&step_, gradient_, mem_bfgs_); 
      optimized_function_->ComputeGradient(coordinates_, &gradient_);
      optimized_function_->ComputeFeasibilityError(coordinates_, 
          &feasibility_error);
      double norm_grad = la::Dot(gradient_.n_elements(), 
          gradient_.ptr(), gradient_.ptr());
      num_of_iterations_++;
      if (silent_==false) {
        ReportProgressFile_();
      }
      if (success_bfgs==SUCCESS_FAIL){
        break;
      }
      if (step_*norm_grad/sigma_ < norm_grad_tolerance_) {
        break;
      }
     // NOTIFY("feasibility_error:%lg desired_feasibility:%lg", feasibility_error, desired_feasibility_);
      if (feasibility_error < desired_feasibility_) {
        break;
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
   // NOTIFY("%lg %lg\n", old_feasibility_error, feasibility_error);
    if (fabs(old_feasibility_error - feasibility_error)
        /(old_feasibility_error+1e-20) < feasibility_tolerance_ ||
        feasibility_error < desired_feasibility_) {
      break;
    }
    old_feasibility_error = feasibility_error;
    UpdateLagrangeMult_();
    optimized_function_->ComputeGradient(coordinates_, &gradient_);
  }

}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::GetResults(Matrix *result) {
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
void LBfgs<OptimizedFunction>::Reset() {
  sigma_ = fx_param_double(module_, "sigma", 10);
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
  NOTIFY("Initializing optimization ...\n");
  coordinates_.Init(new_dimension_, num_of_points_);
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

template<typename OptimizedFunction>
success_t LBfgs<OptimizedFunction>::ComputeWolfeStep_(double *step, Matrix &direction) {
  fx_timer_start(module_, "wolfe_step");
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
  for(index_t i=0; beta>min_beta_/sigma_; i++) { 
    temp_coordinates.CopyValues(coordinates_);
    la::AddExpert(-step_size_*beta, direction, &temp_coordinates);
    optimized_function_->Project(&coordinates_); 
    lagrangian2 = optimized_function_->ComputeLagrangian(temp_coordinates);
    if (lagrangian2 <= lagrangian1 + wolfe_factor)  {
      optimized_function_->ComputeGradient(temp_coordinates, &temp_gradient);
      double dot_product_new = -la::Dot(temp_gradient.n_elements(), 
          temp_gradient.ptr(), direction.ptr());
      if (dot_product_new >= wolfe_sigma2_*dot_product) {
        break;
      }     
    }
    beta *=wolfe_beta_;
    wolfe_factor *=wolfe_beta_;
  }
  if(beta<=min_beta_/sigma_) {
    *step=0;
    fx_timer_stop(module_, "wolfe_step");
    return SUCCESS_FAIL;
  } else {
    *step=step_size_*beta;
    coordinates_.CopyValues(temp_coordinates);   
    fx_timer_stop(module_, "wolfe_step");
    return SUCCESS_PASS;
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
  if (unlikely(y_y<1e-10)){
    NONFATAL("Gradient differences close to singular...norm=%lg\n", y_y);
  } 
  double norm_scale=s_y/(y_y+1e-10);
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
  ComputeWolfeStep_(step, temp_direction);
  if (step==0) {
    NONFATAL("BFGS Failed looking in the other direction...\n");
    la::Scale(-1.0, &temp_direction);
    ComputeWolfeStep_(step, temp_direction);
    *step=-*step;
    fx_timer_stop(module_, "bfgs_step");
    return SUCCESS_FAIL;  
  }
  fx_timer_stop(module_, "bfgs_step");
  return SUCCESS_PASS;
  fx_timer_stop(module_, "bfgs_step");
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
    NONFATAL("Rejecting s, y they don't satisfy curvature condition");
    return SUCCESS_FAIL;
  } 
  s_bfgs_[index_bfgs].CopyValues(temp_s_bfgs);
  y_bfgs_[index_bfgs].CopyValues(temp_y_bfgs); 
  ro_bfgs_[index_bfgs] = 1.0/temp_ro;

  fx_timer_stop(module_, "update_bfgs");
  return SUCCESS_PASS;
}

template<typename OptimizedFunction>
std::string LBfgs<OptimizedFunction>::ComputeProgress_() {
  double lagrangian=optimized_function_->ComputeLagrangian(coordinates_);
  double objective;
  optimized_function_->ComputeObjective(coordinates_, &objective);
  double feasibility_error;
  optimized_function_->ComputeFeasibilityError(
      coordinates_, &feasibility_error);
  double norm_grad=la::Dot(gradient_.n_elements(), 
      gradient_.ptr(), gradient_.ptr());
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
