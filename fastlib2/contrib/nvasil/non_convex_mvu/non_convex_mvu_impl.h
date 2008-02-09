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
 eta_ = 0.9;
 gamma_ = 0.9;
 step_size_ = 1; 
 max_iterations_ = 1000;
 tolerance_ = 1e-4;
 armijo_sigma_=1e-1;
 armijo_beta_=0.5;
 new_dimension_ = -1;
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
  double new_feasibility_error = DBL_MAX;
  double old_feasibility_error = DBL_MAX; 
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
  sigma_ = 1.0;
  
  previous_feasibility_error_= ComputeFeasibilityError_();
  NOTIFY("Starting optimization ...\n");
  for(index_t it1=0; it1<max_iterations_; it1++) {  
    for(index_t it2=0; it2<max_iterations_; it2++) {
      ComputeGradient_();
      LocalSearch_();
      new_feasibility_error = ComputeFeasibilityError_();
      NOTIFY("Iteration: %"LI"d : %"LI"d, feasibility error: %lg \n", it1, it2, 
          new_feasibility_error);
      if (fabs(new_feasibility_error-old_feasibility_error)<tolerance_){
        break;
      }
      old_feasibility_error = new_feasibility_error;
    }
    if (new_feasibility_error < tolerance_) {
      break;
    }
    UpdateLagrangeMult_();
  }
  NOTIFY("Converged !!\n");
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
 
void NonConvexMVU::UpdateLagrangeMult_() {
  double feasibility_error = ComputeFeasibilityError_();
  if (feasibility_error< eta_ * previous_feasibility_error_) {
    for(index_t i=0; i<gradient_.n_cols(); i++) {
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
    previous_feasibility_error_ = feasibility_error;
  } else {
    // langange multipliers unchanged
    sigma_*=gamma_;
    // previous feasibility error unchanged
  }

}

void NonConvexMVU::LocalSearch_() {
  Matrix temp_coordinates;
  temp_coordinates.Init(coordinates_.n_rows(), coordinates_.n_cols()); 
  double lagrangian1 = ComputeLagrangian_(coordinates_);
  double lagrangian2 = 0;
  double beta=armijo_beta_;
  double gradient_norm = la::LengthEuclidean(gradient_.n_rows()
                                             *gradient_.n_cols(),
                                             gradient_.ptr());
  double armijo_factor = gradient_norm *armijo_sigma_ * armijo_beta_ * step_size_;
  for(index_t i=0; ; i++) {
    temp_coordinates.CopyValues(coordinates_);
    la::AddExpert(-step_size_*beta, gradient_, &temp_coordinates);
    lagrangian2 =  ComputeLagrangian_(temp_coordinates);
    if (lagrangian1-lagrangian2 >= armijo_factor) {
      break;
    } else {
      beta *=armijo_beta_;
      armijo_factor *=armijo_beta_;
    }
  }
  NOTIFY("lagrangian1 - lagrangian2 = %lg\n", lagrangian1-lagrangian2);
  coordinates_.CopyValues(temp_coordinates);   
}

double NonConvexMVU::ComputeLagrangian_(Matrix &coordinates) {
  double lagrangian=0;
  for(index_t i=0; i<coordinates.n_cols(); i++) {
    lagrangian += la::LengthEuclidean(new_dimension_, coordinates.GetColumnPtr(i));
    for(index_t k=0; k<knns_; k++) {
      double *point1 = coordinates_.GetColumnPtr(i);
      double *point2 = coordinates_.GetColumnPtr(neighbors_[i*knns_+k]);
      double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                          -distances_[i*knns_+k];
      lagrangian += -lagrange_mult_[i]*dist_diff +  0.5*sigma_*dist_diff*dist_diff;
    }
  }  
  return 0.5*lagrangian; 
}



double NonConvexMVU::ComputeFeasibilityError_() {
  double error=0;
  for(index_t i=0; i<coordinates_.n_cols(); i++) {
    for(index_t k=0; k<knns_; k++) {
      double *point1 = coordinates_.GetColumnPtr(i);
      double *point2 = coordinates_.GetColumnPtr(neighbors_[i*knns_+k]);
      error+= math::Sqr((la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                          -distances_[i*knns_+k]));
    }
  }
  return error;  
}

void NonConvexMVU::ComputeGradient_() {
  gradient_.CopyValues(coordinates_);
//  la::Scale(-1.0, &gradient_);
  for(index_t i=0; i<gradient_.n_cols(); i++) {
    for(index_t k=0; k<knns_; k++) {
      double a_i_r[new_dimension_];
      double *point1 = coordinates_.GetColumnPtr(i);
      double *point2 = coordinates_.GetColumnPtr(neighbors_[i*knns_+k]);
      la::SubOverwrite(new_dimension_, point1, point2, a_i_r);
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
  }   
}


