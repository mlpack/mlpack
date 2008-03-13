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
  wolfe_sigma1_ = fx_param_double(module_, "wolfe_sigma1", 0.1);
  wolfe_sigma2_ = fx_param_double(module_, "wolfe_sigma2", 0.9);
  step_size_=fx_param_double(module_, "step_size", 1.0);
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
  max_iterations_ = fx_param_int(module_, "max_iterations", 10000);
  // the memory of bfgs 
  mem_bfgs_ = fx_param_int(module_, "mem_bfgs_", 50);
  std::string log_file=fx_param_str(module_, "log_file", "opt_log");
  fp_log_=fopen(log_file.c_str(), "w");
  optimized_function_->set_sigma(sigma_);
}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::Destruct() {
 fclose(fp_log_);

}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::ComputeLocalOptimumBFGS() {
  double feasibility_error;
  double step; 
  if (unlikely(mem_bfgs_<0)) {
    FATAL("You forgot to initialize the memory for BFGS\n");
  }
  InitOptimization_();
  // Init the memory for BFGS
  s_bfgs_.Init(mem_bfgs_);
  y_bfgs_.Init(mem_bfgs_);
  ro_bfgs_.Init(mem_bfgs_);
  ro_bfgs_.SetAll(0.0);
  for(index_t i=0; i<mem_bfgs_; i++) {
    s_bfgs_[i].Init(new_dimension_, num_of_points_);
    y_bfgs_[i].Init(new_dimension_, num_of_points_);
  } 
  NOTIFY("Starting optimization ...\n");
  //datanode_write(module_, stdout);
  //datanode_write(module_, fp_log_);
  // Run a few iterations with gradient descend to fill the memory of BFGS
  NOTIFY("Running a few iterations with gradient descent to fill "
         "the memory of BFGS...\n");
   index_bfgs_=0;
  // You have to compute also the previous_gradient_ and previous_coordinates_
  // tha are needed only by BFGS
  optimized_function_->ComputeGradient(coordinates_, &gradient_);
  previous_gradient_.Copy(gradient_);
  previous_coordinates_.Copy(coordinates_);
  ComputeWolfeStep_(&step, gradient_);
  optimized_function_->ComputeGradient(coordinates_, &gradient_);
  la::SubOverwrite(previous_coordinates_, coordinates_, &s_bfgs_[0]);
  la::SubOverwrite(previous_gradient_, gradient_, &y_bfgs_[0]);
  ro_bfgs_[0] = la::Dot(s_bfgs_[0].n_elements(), 
      s_bfgs_[0].ptr(), y_bfgs_[0].ptr());
  for(index_t i=0; i<mem_bfgs_; i++) {
    ComputeBFGS_(&step, gradient_, i);
    optimized_function_->ComputeGradient(coordinates_, &gradient_);
    UpdateBFGS_();
    previous_gradient_.CopyValues(gradient_);
    previous_coordinates_.CopyValues(coordinates_);
    optimized_function_->ComputeFeasibilityError(coordinates_,
        &feasibility_error); 
  } 
  NOTIFY("Now starting optimizing with BFGS...\n");
  double old_feasibility_error = feasibility_error;
  for(index_t it1=0; it1<max_iterations_; it1++) {  
    for(index_t it2=0; it2<max_iterations_; it2++) {
      ComputeBFGS_(&step, gradient_, mem_bfgs_);
      optimized_function_->ComputeGradient(coordinates_, &gradient_);
      optimized_function_->ComputeFeasibilityError(coordinates_, 
          &feasibility_error);
      double norm_grad = la::Dot(gradient_.n_elements(), 
          gradient_.ptr(), gradient_.ptr()); 
      if (step*norm_grad < norm_grad_tolerance_) {
        break;
      }
      ReportProgressFile_();
      UpdateBFGS_();
      previous_coordinates_.CopyValues(coordinates_);
      previous_gradient_.CopyValues(gradient_);
      num_of_iterations_++;
    }
    NOTIFY("%lg %lg\n", fabs(old_feasibility_error - feasibility_error)
        /old_feasibility_error, 
       feasibility_tolerance_);
    if (fabs(old_feasibility_error - feasibility_error)
        /old_feasibility_error < feasibility_tolerance_) {
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
void LBfgs<OptimizedFunction>::InitOptimization_() {
  if (unlikely(new_dimension_<0)) {
    FATAL("You forgot to set the new dimension\n");
  }
  NOTIFY("Initializing optimization ...\n");
  coordinates_.Init(new_dimension_, num_of_points_);
  gradient_.Init(new_dimension_, num_of_points_);
  for(index_t i=0; i< coordinates_.n_rows(); i++) {
    for(index_t j=0; j<coordinates_.n_cols(); j++) {
      coordinates_.set(i, j, math::Random(0.1, 1.0));
    }
  }
  optimized_function_->Project(&coordinates_);
}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::UpdateLagrangeMult_() {
  optimized_function_->UpdateLagrangeMult(coordinates_);
  sigma_*=gamma_;
  optimized_function_->set_sigma(sigma_);
}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::ComputeWolfeStep_(double *step, Matrix &direction) {
  fx_timer_start(module_, "wolfe_step");
  Matrix temp_coordinates;
  Matrix temp_gradient;
  temp_gradient.Init(new_dimension_, num_of_points_);
  temp_coordinates.Init(coordinates_.n_rows(), coordinates_.n_cols()); 
  double lagrangian1 = optimized_function_->ComputeLagrangian(coordinates_);
  double lagrangian2 = 0;
  double beta=wolfe_beta_;
  double dot_product = la::Dot(direction.n_elements(),
                               gradient_.ptr(),
                               direction.ptr());
  double wolfe_factor =  dot_product * wolfe_sigma1_ * wolfe_beta_ * step_size_;
  for(index_t i=0; beta>1e-200; i++) { 
    temp_coordinates.CopyValues(coordinates_);
    la::AddExpert(-step_size_*beta, direction, &temp_coordinates);
    lagrangian2 = optimized_function_->ComputeLagrangian(temp_coordinates);
    if (lagrangian1-lagrangian2 >= wolfe_factor)  {
      optimized_function_->ComputeGradient(temp_coordinates, &temp_gradient);
      double dot_product_new = la::Dot(temp_gradient.n_elements(), 
          temp_gradient.ptr(), direction.ptr());
      if (dot_product_new <= wolfe_sigma2_*dot_product) {
        break;
      }     
    }
    beta *=wolfe_beta_;
    wolfe_factor *=wolfe_beta_;
  }
  if(beta<=1e-100) {
    *step=0;
  } else {
    *step=step_size_*beta;
  }
  coordinates_.CopyValues(temp_coordinates);   
  optimized_function_->Project(&coordinates_); 
  fx_timer_stop(module_, "wolfe_step");
}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::ComputeBFGS_(double *step, Matrix &grad, index_t memory) {
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
    NONFATAL("Gradient differences close to singular...\n");
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
    la::Scale(-1.0, &temp_direction);
    ComputeWolfeStep_(step, temp_direction);
  }
  (*step)*= la::Dot(num_of_points_*new_dimension_, 
          temp_direction.ptr(), temp_direction.ptr());
  fx_timer_stop(module_, "bfgs_step");
}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::UpdateBFGS_() {
  index_bfgs_ = (index_bfgs_ - 1 + mem_bfgs_ ) % mem_bfgs_;
  UpdateBFGS_(index_bfgs_);
}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::UpdateBFGS_(index_t index_bfgs) {
  fx_timer_start(module_, "update_bfgs");
  // shift all values
  la::SubOverwrite(previous_coordinates_, coordinates_, &s_bfgs_[index_bfgs]);
  la::SubOverwrite(previous_gradient_, gradient_, &y_bfgs_[index_bfgs]);
  ro_bfgs_[index_bfgs_] = la::Dot(new_dimension_ * num_of_points_, 
                                  s_bfgs_[index_bfgs].ptr(),
                                  y_bfgs_[index_bfgs].ptr());
  if unlikely(fabs(ro_bfgs_[index_bfgs]) <=1e-20) {
    int sign =int(2*((ro_bfgs_[index_bfgs]>0.0)-0.5));
    ro_bfgs_[index_bfgs_] =sign/1e-20;
    NONFATAL("Ro values close to singular ...\n");
  } else {
    ro_bfgs_[index_bfgs] = 1.0/ro_bfgs_[index_bfgs_];
  }
  fx_timer_stop(module_, "update_bfgs");
}

template<typename OptimizedFunction>
std::string LBfgs<OptimizedFunction>::ComputeProgress_() {
  double objective;
  optimized_function_->ComputeObjective(coordinates_, &objective);
  double feasibility_error;
  optimized_function_->ComputeFeasibilityError(
      coordinates_, &feasibility_error);
  double norm_grad=la::Dot(gradient_.n_elements(), 
      gradient_.ptr(), gradient_.ptr());
  char buffer[1024];
  sprintf(buffer, "iteration:%i sigma:%lg objective:%lg error:%lg "
      "grad_norm:%lg",
      num_of_iterations_, sigma_, objective, feasibility_error, norm_grad);
  return std::string(buffer);
}

template<typename OptimizedFunction>
void LBfgs<OptimizedFunction>::ReportProgressFile_() {
  std::string progress = ComputeProgress_();
  NOTIFY("%s\n", progress.c_str());
  fprintf(fp_log_, "%s\n", progress.c_str());
}
