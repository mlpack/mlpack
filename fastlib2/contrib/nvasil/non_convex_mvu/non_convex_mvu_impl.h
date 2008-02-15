/*
 * =====================================================================================
 * 
 *       Filename:  non_convex_mvu_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  02/03/2008 11:45:39 AM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

NonConvexMVU::NonConvexMVU() {
 eta_ = 0.25;
 gamma_ = 1.92;
 sigma_ = 1e2;
 step_size_ = 1; 
 max_iterations_ = 20000;
 tolerance_ = 1e-5;
 armijo_sigma_=1e-1;
 armijo_beta_=0.5;
 new_dimension_ = -1;
 mem_bfgs_ = -1;
}
void NonConvexMVU::Init(std::string data_file, index_t knns) {
  Init(data_file, knns, 20);
}

void NonConvexMVU::Init(std::string data_file, index_t knns, index_t leaf_size) {
  knns_=knns;
  leaf_size_=leaf_size;
  NOTIFY("Loading data ...\n");
  data::Load(data_file.c_str(), &data_);
  num_of_points_ = data_.n_cols();
  NOTIFY("Data loaded ...\n");
  NOTIFY("Building tree with data ...\n");
  allknn_.Init(data_, data_, leaf_size_, knns_);
  NOTIFY("Tree built ...\n");
  NOTIFY("Computing neighborhoods ...\n");
  allknn_.ComputeNeighbors(&neighbors_,
                            &distances_);
  NOTIFY("Neighborhoods computes ...\n");
  previous_feasibility_error_ = DBL_MAX;
}

void NonConvexMVU::ComputeLocalOptimum() {
  double distance_constraint;
  double centering_constraint;
  double sum_of_dist_square = la::LengthEuclidean(distances_.size(), &distances_[0]);
  InitOptimization_();
  NOTIFY("Starting optimization ...\n");
  ComputeFeasibilityError_(&distance_constraint, &centering_constraint);
  previous_feasibility_error_= distance_constraint + centering_constraint; 
  double step; 
  for(index_t it1=0; it1<max_iterations_; it1++) {  
    for(index_t it2=0; it2<max_iterations_; it2++) {
      ComputeGradient_();
      LocalSearch_(&step);
      ComputeFeasibilityError_(&distance_constraint, &centering_constraint);
      NOTIFY("Iteration: %"LI"d : %"LI"d, feasibility error (dist)): %lg\n"
             "feasibility error (center): %lg \n", it1, it2, 
               distance_constraint , centering_constraint);
      if (step < tolerance_){
        break;
      }
    }
    if (distance_constraint/sum_of_dist_square < tolerance_) {
      NOTIFY("Converged !!\n");
      NOTIFY("Objective function: %lg\n", ComputeObjective_(coordinates_));
      NOTIFY("Distances constraints: %lg, Centering constraint: %lg\n", 
              distance_constraint/sum_of_dist_square, centering_constraint);
      return;
    }
   // UpdateLagrangeMult_();
   UpdateLagrangeMultStochastic_();
  }
    NOTIFY("Didn't converge, maximum number of iterations reached !!\n");
    NOTIFY("Objective function: %lg\n", ComputeObjective_(coordinates_));
    NOTIFY("Distances constraints: %lg, Centering constraint: %lg\n", 
              distance_constraint, centering_constraint);
 
}

void NonConvexMVU::ComputeLocalOptimumBFGS() {
  double distance_constraint;
  double centering_constraint;
  double sum_of_dist_square = la::LengthEuclidean(distances_.size(), &distances_[0]);
  if (unlikely(mem_bfgs_<0)) {
    FATAL("You forgot to initialize the memory for BFGS\n");
  }
  InitOptimization_();
  // You have to compute also the previous_gradient_ and previous_coordinates_
  // tha are needed only by BFGS
  ComputeGradient_();
  previous_gradient_.Copy(gradient_);
  previous_coordinates_.Copy(coordinates_);
  // Init the memory for BFGS
  s_bfgs_.Init(mem_bfgs_);
  y_bfgs_.Init(mem_bfgs_);
  ro_bfgs_.Init(mem_bfgs_);
  for(index_t i=0; i<mem_bfgs_; i++) {
    s_bfgs_[i].Init(new_dimension_, num_of_points_);
    y_bfgs_[i].Init(new_dimension_, num_of_points_);
  } 
  NOTIFY("Starting optimization ...\n");
  ComputeFeasibilityError_(&distance_constraint, &centering_constraint);
  previous_feasibility_error_= distance_constraint + centering_constraint; 
  double step; 
  // Run a few iterations with gradient descend to fill the memory of BFGS
  NOTIFY("Running a few iterations with gradient descent to fill "
         "the memory of BFGS...\n");
  index_bfgs_=0;
  for(index_t i=0; i<mem_bfgs_; i++) {
    LocalSearch_(&step);
    ComputeGradient_();
    la::SubOverwrite(coordinates_, previous_coordinates_, &s_bfgs_[i]);
    la::SubOverwrite(gradient_, previous_gradient_, &y_bfgs_[i]);
    ro_bfgs_[i] = la::Dot(num_of_points_ * new_dimension_, s_bfgs_[i].ptr(),
                          y_bfgs_[i].ptr());
    previous_gradient_.CopyValues(gradient_);
    previous_coordinates_.CopyValues(coordinates_);
  } 
  NOTIFY("Now starting optimizing with BFGS...\n");
  for(index_t it1=0; it1<max_iterations_; it1++) {  
    for(index_t it2=0; it2<max_iterations_; it2++) {
      ComputeBFGS_();
      ComputeGradient_();
      la::SubFrom(gradient_, &coordinates_);
      UpdateBFGS_();
      ComputeFeasibilityError_(&distance_constraint, &centering_constraint);
      NOTIFY("Iteration: %"LI"d : %"LI"d, feasibility error (dist)): %lg\n"
             "feasibility error (center): %lg \n", it1, it2, 
               distance_constraint , centering_constraint);
      step=la::DistanceSqEuclidean(new_dimension_ * num_of_points_,
          previous_coordinates_.ptr(), coordinates_.ptr());
      if (step  < tolerance_){
        break;
      }
      previous_coordinates_.CopyValues(coordinates_);
      previous_gradient_.CopyValues(gradient_);
    }
    if (distance_constraint/sum_of_dist_square < tolerance_) {
      NOTIFY("Converged !!\n");
      NOTIFY("Objective function: %lg\n", ComputeObjective_(coordinates_));
      NOTIFY("Distances constraints: %lg, Centering constraint: %lg\n", 
              distance_constraint/sum_of_dist_square, centering_constraint);
      return;
    }
   // UpdateLagrangeMult_();
   UpdateLagrangeMultStochastic_();
  }
    NOTIFY("Didn't converge, maximum number of iterations reached !!\n");
    NOTIFY("Objective function: %lg\n", ComputeObjective_(coordinates_));
    NOTIFY("Distances constraints: %lg, Centering constraint: %lg\n", 
              distance_constraint, centering_constraint);
  
}

void NonConvexMVU::set_eta(double eta) {
  eta_ = eta;
}

void NonConvexMVU::set_gamma(double gamma) {
  gamma_ = gamma;
}

void NonConvexMVU::set_step_size(double step_size) {
  step_size_ = step_size;
}

void NonConvexMVU::set_max_iterations(index_t max_iterations) {
  max_iterations_ = max_iterations;
}

void NonConvexMVU::set_new_dimension(index_t new_dimension) {
  new_dimension_ = new_dimension;
}

void NonConvexMVU::set_tolerance(double tolerance) {
  tolerance_ = tolerance;
}

void NonConvexMVU::set_armijo_sigma(double armijo_sigma) {
  armijo_sigma_ = armijo_sigma;
}

void NonConvexMVU::set_armijo_beta(double armijo_beta) {
  armijo_beta_ = armijo_beta;
}

void NonConvexMVU::set_mem_bfgs(index_t mem_bfgs) {
  mem_bfgs_ = mem_bfgs;
}

void NonConvexMVU::InitOptimization_() {
  if (unlikely(new_dimension_<0)) {
    FATAL("You forgot to set the new dimension\n");
  }
  NOTIFY("Initializing optimization ...\n");
  coordinates_.Init(new_dimension_, num_of_points_);
  gradient_.Init(new_dimension_, num_of_points_);
  for(index_t i=0; i< coordinates_.n_rows(); i++) {
    for(index_t j=0; j<coordinates_.n_cols(); j++) {
      coordinates_.set(i, j, math::Random(0.1, 1));
    }
  }
  lagrange_mult_.Init(knns_ * num_of_points_);
  for(index_t i=0; i<lagrange_mult_.length(); i++) {
    lagrange_mult_[i]=math::Random(0.1, 1.0);
  }
  centering_lagrange_mult_.Init(new_dimension_);
  for(index_t i=0; i<new_dimension_; i++) {
    centering_lagrange_mult_[i]=math::Random(0.1, 1.0);
  }
 
}

void NonConvexMVU::UpdateLagrangeMult_() {
  double distance_constraint;
  double centering_constraint;
  ComputeFeasibilityError_(&distance_constraint, &centering_constraint);
  double feasibility_error = distance_constraint + centering_constraint;
  if (feasibility_error< eta_ * previous_feasibility_error_) {
    for(index_t i=0; i<num_of_points_; i++) {
      for(index_t j=0; j<new_dimension_; j++) {
        // Update the Lagrange multiplier for the centering constraint
        centering_lagrange_mult_[j] -= sigma_ * coordinates_.get(j, i);
      }
      for(index_t k=0; k<knns_; k++) {
        double *point1 = coordinates_.GetColumnPtr(i);
        double *point2 = coordinates_.GetColumnPtr(neighbors_[i*knns_+k]);
        double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                           -distances_[i*knns_+k];
        lagrange_mult_[i*knns_+k]-=sigma_*dist_diff;
      }
    }  
    // I know this looks redundant but we just follow the algorithm
    // sigma_ = sigma_; 
  } else {
    // langange multipliers unchanged
    sigma_*=gamma_;
    // previous feasibility error unchanged
  }
  previous_feasibility_error_ = feasibility_error;
}

// This one choses stocastically what to update, sigma or lagrange multipliers
// This guarantees that both happening
void NonConvexMVU::UpdateLagrangeMultStochastic_() {
  if (math::Random(0, 1) < 0.5) {
    // Update lagrange multipliers
    for(index_t i=0; i<num_of_points_; i++) {
      for(index_t j=0; j<new_dimension_; j++) {
        // Update the Lagrange multiplier for the centering constraint
        centering_lagrange_mult_[j] -= sigma_ * coordinates_.get(j, i);
      }
      for(index_t k=0; k<knns_; k++) {
        double *point1 = coordinates_.GetColumnPtr(i);
        double *point2 = coordinates_.GetColumnPtr(neighbors_[i*knns_+k]);
        double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                           -distances_[i*knns_+k];
        lagrange_mult_[i*knns_+k]-=sigma_*dist_diff;
      }
    }  
      
  } else {
    // increase sigma
    sigma_ =std::min(sigma_ *gamma_, 1e7)    ;
  }
}

void NonConvexMVU::LocalSearch_(double *step) {
  Matrix temp_coordinates;
  temp_coordinates.Init(coordinates_.n_rows(), coordinates_.n_cols()); 
  double lagrangian1 = ComputeLagrangian_(coordinates_);
  double lagrangian2 = 0;
  double beta=armijo_beta_;
  double gradient_norm = la::LengthEuclidean(gradient_.n_rows()
                                             * gradient_.n_cols(),
                                             gradient_.ptr());
  double armijo_factor =  gradient_norm * armijo_sigma_ * armijo_beta_ * step_size_;
  for(index_t i=0; ; i++) {
    temp_coordinates.CopyValues(coordinates_);
    la::AddExpert(-step_size_*beta/gradient_norm, gradient_, &temp_coordinates);
    lagrangian2 =  ComputeLagrangian_(temp_coordinates);
    if (lagrangian1-lagrangian2 >= armijo_factor) {
      break;
    } else {
      beta *=armijo_beta_;
      armijo_factor *=armijo_beta_;
    }
  }
  *step=step_size_*beta;
/*  
  temp_coordinates.CopyValues(coordinates_);
  la::AddExpert(-0.01/gradient_norm, gradient_, &temp_coordinates);
  lagrangian2 =  ComputeLagrangian_(temp_coordinates);
*/
  NOTIFY("step_size: %lg, sigma: %lg\n", beta * step_size_, sigma_);
  NOTIFY("lagrangian1 - lagrangian2 = %lg\n", lagrangian1-lagrangian2);
  NOTIFY("lagrangian2: %lg, Objective: %lg\n", lagrangian2,
                                               ComputeObjective_(temp_coordinates));
  coordinates_.CopyValues(temp_coordinates);   
}

void NonConvexMVU::ComputeBFGS_() {
  Vector alpha;
  alpha.Init(mem_bfgs_);
  Matrix scaled_y;
  scaled_y.Init(new_dimension_, num_of_points_);
  index_t num=0;
  for(index_t i=index_bfgs_, num=0; num<mem_bfgs_; i=(i+1)%mem_bfgs_, num++) {
    alpha[i] = la::Dot(new_dimension_ * num_of_points_,
                       s_bfgs_[i].ptr(), 
                       gradient_.ptr());
    alpha[i] *= ro_bfgs_[i];
    scaled_y.CopyValues(y_bfgs_[i]);
    la::Scale(alpha[i], &scaled_y);
    la::SubFrom(scaled_y, &gradient_);
  }
  // We need to scale the gradient here
  double norm_scale=1/ro_bfgs_[index_bfgs_]/la::Dot(num_of_points_ *
                    new_dimension_, y_bfgs_[index_bfgs_].ptr(),
                    y_bfgs_[index_bfgs_].ptr());
  la::Scale(norm_scale, &gradient_);
  
  Matrix scaled_s;
  double beta;
  scaled_s.Init(new_dimension_, num_of_points_);
  num=0;
  for(index_t j=(index_bfgs_-1+mem_bfgs_)%mem_bfgs_, num=0; num<mem_bfgs_; 
      num++, j=(j-1+mem_bfgs_)%mem_bfgs_) {
    beta = la::Dot(new_dimension_ * num_of_points_, 
                   y_bfgs_[j].ptr(),
                   gradient_.ptr());
    beta *= ro_bfgs_[j];
    scaled_s.CopyValues(s_bfgs_[j]);
    la::Scale(alpha[j]-beta, &scaled_s);
    la::AddTo(scaled_s, &gradient_);
  }
}

void NonConvexMVU::UpdateBFGS_() {
  // shift all values
  index_bfgs_ = (index_bfgs_ - 1 + mem_bfgs_ ) % mem_bfgs_;
  la::SubOverwrite(coordinates_, previous_coordinates_, &s_bfgs_[index_bfgs_]);
  la::SubOverwrite(gradient_, previous_gradient_, &y_bfgs_[index_bfgs_]);
  ro_bfgs_[index_bfgs_] = 1.0/la::Dot(new_dimension_ * num_of_points_, 
                                      s_bfgs_[index_bfgs_].ptr(),
                                      y_bfgs_[index_bfgs_].ptr());
}

double NonConvexMVU::ComputeLagrangian_(Matrix &coord) {
  double lagrangian=0;
  Vector deviations;
  deviations.Init(new_dimension_);
  deviations.SetAll(0.0);
  for(index_t i=0; i<coord.n_cols(); i++) {
    // we are maximizing the trace or minimize the -trace
    lagrangian -= la::Dot(new_dimension_, 
                          coord.GetColumnPtr(i),
                          coord.GetColumnPtr(i));
    for(index_t k=0; k<knns_; k++) {
      double *point1 = coord.GetColumnPtr(i);
      double *point2 = coord.GetColumnPtr(neighbors_[i*knns_+k]);
      double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                          -distances_[i*knns_+k];
      lagrangian += -lagrange_mult_[i]*dist_diff +  0.5*sigma_*dist_diff*dist_diff;
    }
    for(index_t k=0; k<new_dimension_; k++) {
      deviations[k] += coord.get(k, i);
    }
  }
  // Update the centering conditions  
  for(index_t k=0; k<new_dimension_; k++) {
    lagrangian += -deviations[k]*centering_lagrange_mult_[k] +
        0.5 * sigma_ * deviations[k] * deviations[k];
  }
  
  return 0.5*lagrangian; 
}



void NonConvexMVU::ComputeFeasibilityError_(double *distance_constraint,
                                            double *centering_constraint) {
  Vector deviations;
  deviations.Init(new_dimension_);
  deviations.SetAll(0.0);
  *distance_constraint=0;
  *centering_constraint=0;
  for(index_t i=0; i<coordinates_.n_cols(); i++) {
    for(index_t k=0; k<knns_; k++) {
      double *point1 = coordinates_.GetColumnPtr(i);
      double *point2 = coordinates_.GetColumnPtr(neighbors_[i*knns_+k]);
      *distance_constraint+= math::Sqr((la::DistanceSqEuclidean(new_dimension_, 
                                                                point1, point2) 
                             -distances_[i*knns_+k]));
    }
    for(index_t k=0; k<new_dimension_; k++) {
      deviations[k] += coordinates_.get(k, i);
    }
  }
  for(index_t k=0; k<new_dimension_; k++) {
    *centering_constraint += deviations[k] * deviations[k];
  }
}

double NonConvexMVU::ComputeFeasibilityError_() {
  double distance_constraint;
  double centering_constraint;
  ComputeFeasibilityError_(&distance_constraint, &centering_constraint);
  return distance_constraint+centering_constraint;
}

void NonConvexMVU::ComputeGradient_() {
  gradient_.CopyValues(coordinates_);
  // we need to use -CRR^T because we want to maximize CRR^T
  la::Scale(-1.0, &gradient_);
  Vector dimension_sums;
  dimension_sums.Init(new_dimension_);
  dimension_sums.SetAll(0.0);
  for(index_t i=0; i<gradient_.n_cols(); i++) {
    for(index_t k=0; k<knns_; k++) {
      double a_i_r[new_dimension_];
      double *point1 = coordinates_.GetColumnPtr(i);
      double *point2 = coordinates_.GetColumnPtr(neighbors_[i*knns_+k]);
      la::SubOverwrite(new_dimension_, point2, point1, a_i_r);
      double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                          -distances_[i*knns_+k] ;
      la::AddExpert(new_dimension_,
           (-lagrange_mult_[i*knns_+k]+dist_diff*sigma_), 
          a_i_r, 
          gradient_.GetColumnPtr(i));
      la::AddExpert(new_dimension_,
           (lagrange_mult_[i*knns_+k]-dist_diff*sigma_), 
          a_i_r, 
          gradient_.GetColumnPtr(neighbors_[i*knns_+k]));
    }
    
    for(index_t k=0; k<new_dimension_; k++) {
      gradient_.set(k, i, gradient_.get(k, i) - centering_lagrange_mult_[k]);
      dimension_sums[k] += coordinates_.get(k, i);
    }   
  }
  
  for(index_t i=0; i<gradient_.n_cols(); i++)  {
    la::AddExpert(new_dimension_, sigma_, 
        dimension_sums.ptr(), 
        gradient_.GetColumnPtr(i));
  }  
}

double NonConvexMVU::ComputeObjective_(Matrix &coord) {
  double variance=0;
  for(index_t i=0; i< coord.n_cols(); i++) {
    variance-=la::Dot(new_dimension_, 
                      coord.GetColumnPtr(i),
                      coord.GetColumnPtr(i));

  }
  return variance;
}
