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
 knns_=5;
 kfns_=0;
 leaf_size_=20;
 eta_ = 0.9;
 gamma_ =1.3;
 sigma_ = 1e3;
 step_size_ = 2; 
 max_iterations_ = 10000000;
 distance_tolerance_ = 1e-1;
 wolfe_sigma1_=1e-1;
 wolfe_sigma2_=0.9;
 wolfe_beta_=0.8;
 new_dimension_ = -1;
 mem_bfgs_ = -1;
 max_violation_of_distance_constraint_ =4*1e6;
}

template<GradientEnum gradient_mode>
void NonConvexMVU::Init(std::string data_file) {
  Matrix data;
  NOTIFY("Loading data ...\n");
  data::Load(data_file.c_str(), &data);
  if (unlikely(data_.n_cols()<=0)) {
    FATAL("Failed to load data, 0 entries found, probably empty file\n");
  }
  Init(data);
}

template<GradientEnum gradient_mode>
void NonConvexMVU::Init(Matrix &data) {
  data_.Own(&data);
  RemoveMean_(data_);
  num_of_points_ = data_.n_cols();
  NOTIFY("Data loaded ...\n");
  NOTIFY("Nearest neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  allknn_.Init(data_, leaf_size_, knns_);
  NOTIFY("Tree built ...\n");
  NOTIFY("Computing neighborhoods ...\n");
  ArrayList<index_t> from_tree_neighbors;
  ArrayList<double>  from_tree_distances;
  allknn_.ComputeNeighbors(&from_tree_neighbors,
                           &from_tree_distances);
  NOTIFY("Neighborhoods computed...\n");

  if (gradient_mode==DeterministicGrad) { 
    NOTIFY("Consolidating neighbors...\n");
    ConsolidateNeighbors_(from_tree_neighbors,
                          from_tree_distances,
                          knns_,
                          &nearest_neighbor_pairs_,
                          &nearest_distances_,
                          &num_of_nearest_pairs_);
  } 
  if (gradient_mode == StochasticGrad) {
    NOTIFY("Initializing for stochastic grad\n");
    nearest_neighbors_.Steal(&from_tree_neighbors);
    nearest_distances_.Steal(&from_tree_distances);
  } else {
    // dummy Init
    nearest_neighbors_.Init();
  }
  if (kfns_!=0) {
    NOTIFY("Furthest neighbor factors ...\n");
    allkfn_.Init(data_, leaf_size_, kfns_);
    NOTIFY("Tree built ...\n");
    NOTIFY("Computing neighborhoods ...\n");
    from_tree_neighbors.Destruct();
    from_tree_distances.Destruct();
    allkfn_.ComputeNeighbors(&from_tree_neighbors,
                             &from_tree_distances);
    NOTIFY("Neighborhoods computed...\n");
    ConsolidateNeighbors_(from_tree_neighbors,
                        from_tree_distances,
                        kfns_,
                        &furthest_neighbor_pairs_,
                        &furthest_distances_,
                        &num_of_furthest_pairs_);
    if (gradient_mode==StochasticGrad) {
      furthest_neighbors_.Steal(&from_tree_neighbors);
      furthest_distances_.Destruct();
      furthest_distances_.Steal(&from_tree_distances);
    }     
  } else {
    //the annoying fastlib initialization thing
    furthest_neighbor_pairs_.Init();
    furthest_distances_.Init();
  }
  previous_feasibility_error_ = DBL_MAX;
}

TEMPLATE_TAG_
void NonConvexMVU::ComputeLocalOptimumSGD() {
  if (unlikely(gradient_mode==DeterministicGrad)) {
    FATAL("You have coonflicting flags, you are trying to optimize with "
          "stochastic gradient descent with a deterministic gradient option\n");
  }
  InitOptimization_<OPT_PARAM_>();
  Vector gradient1;
  Vector gradient2;
  gradient1.Init(new_dimension_);
  gradient2.Init(new_dimension_);
  index_t it1;
  double step0=1.0e0;
  double step;
  double total_distances=0.0;
  double objective=0;
  double feasibility_error=0;
  for(index_t i=0; i<nearest_distances_.size(); i++) {
    total_distances+=nearest_distances_[i]*nearest_distances_[i];
  }
    
  NOTIFY("Optimization with Stochastic Gradient Descent Started\n");
  double prev_feas=DBL_MAX;
  index_t exterior_points=0;
  for(index_t kk=0; kk<3; kk++) {
  for(it1=0; it1<100; it1++) {  
    step=1.0/std::pow(2,step0/(it1+1));
    for(index_t it2=0; it2<max_iterations_; it2++) {
      for(index_t i=0; i<num_of_points_; i++) {
        for(index_t k=0; k<kfns_; k++) {
          index_t current_index = i;
          index_t furthest_index = furthest_neighbors_[i*kfns_+k];
          //printf("%i  %i\n", current_index, nearest_index);
          double *point1=coordinates_.GetColumnPtr(current_index);
          double *point2=coordinates_.GetColumnPtr(furthest_index);
          double dist1=la::DistanceSqEuclidean(new_dimension_, point1, point2); 
          //printf("%lg  %lg\n", norm_grad1, norm_grad2);
          double scale_factor=sigma_*(furthest_distances_[i*kfns_+k]-dist1)/(dist1+1e-10);
          
          if (dist1<furthest_distances_[i*kfns_+k]) {
            la::SubOverwrite(new_dimension_, point2, point1, gradient1.ptr());
            la::SubOverwrite(new_dimension_, point1, point2, gradient2.ptr());
            exterior_points++;

            la::AddExpert(new_dimension_, step*scale_factor, 
                          gradient1.ptr(),
                          point1);
            la::AddExpert(new_dimension_, step*scale_factor, 
                          gradient2.ptr(),
                          point2);
          }
        }
        for(index_t k=0; k<knns_; k++) {
          index_t current_index = i;
          index_t nearest_index = nearest_neighbors_[i*knns_ + k];
          //printf("%i  %i\n", current_index, nearest_index);
          double *point1=coordinates_.GetColumnPtr(current_index);
          double *point2=coordinates_.GetColumnPtr(nearest_index);
          double dist1=la::DistanceSqEuclidean(new_dimension_, point1, point2); 
          ComputePairGradient_<OPT_PARAM_>(current_index, k, 
              coordinates_, &gradient1, &gradient2);
          //printf("%lg  %lg\n", norm_grad1, norm_grad2);
          double scale_factor=1.0/(dist1+1e-10);
          la::AddExpert(new_dimension_, -step*scale_factor, 
                        gradient1.ptr(),
                        point1);
          la::AddExpert(new_dimension_, -step*scale_factor, 
                        gradient2.ptr(),
                        point2);
          //NOTIFY("%lg\n", ComputeFeasibilityError_<OPT_PARAM_>());
        }
      }
    }
    RemoveMean_(coordinates_);
    double augmented_lagrangian=ComputeLagrangian_<OPT_PARAM_>(coordinates_);
    objective =ComputeObjective_<OPT_PARAM_>(coordinates_);
    feasibility_error =ComputeFeasibilityError_<OPT_PARAM_>();
    if (fabs(prev_feas-feasibility_error)<distance_tolerance_) {
      break;
    } else{
      prev_feas=feasibility_error;
    }
    NOTIFY("Iteration: %"LI"d : %"LI"d, feasibility error: %lg\n"
           "    stress     : %lg\n"
           "    exteriors  : %i\n"
           "    step       : %lg\n"
           "    lang_mult  : %lg\n"
           "    sigma      : %lg\n"
           "    lagrangian : %lg\n", it1, it1,
           feasibility_error, 
           feasibility_error/total_distances,
           exterior_points,
           step,
           la::LengthEuclidean(lagrange_mult_),
           sigma_,
           augmented_lagrangian);
   // UpdateLagrangeMult_<OPT_PARAM_>();
   // current_index=math::RandInt(0, num_of_points_);
    if (feasibility_error/total_distances<1e-6) {
      break;
    }
    exterior_points=0;
  }
  }
  char buffer[1024];
  sprintf(buffer, "Converged_"
      "Objective_function_%lg_"
      "stress_%lg_objective_%lg",
       ComputeObjective_<OPT_PARAM_>(coordinates_),
       feasibility_error/total_distances, objective);
  result_summary_=buffer;
  printf("%s\n", result_summary_.c_str());
 
}

// PROBLEM_TYPE:
TEMPLATE_TAG_
void NonConvexMVU::ComputeLocalOptimumBFGS() {
  double distance_constraint;
  double centering_constraint;
  double step; 
  if (unlikely(mem_bfgs_<0)) {
    FATAL("You forgot to initialize the memory for BFGS\n");
  }
  InitOptimization_<OPT_PARAM_>();
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
  // Run a few iterations with gradient descend to fill the memory of BFGS
  NOTIFY("Running a few iterations with gradient descent to fill "
         "the memory of BFGS...\n");
   index_bfgs_=0;
  // You have to compute also the previous_gradient_ and previous_coordinates_
  // tha are needed only by BFGS
  ComputeGradient_<OPT_PARAM_>(coordinates_, &gradient_);
  previous_gradient_.Copy(gradient_);
  previous_coordinates_.Copy(coordinates_);
  LocalSearch_<OPT_PARAM_>(&step, gradient_);
  ComputeGradient_<OPT_PARAM_>(coordinates_, &gradient_);
  la::SubOverwrite(previous_coordinates_, coordinates_, &s_bfgs_[0]);
  la::SubOverwrite(previous_gradient_, gradient_, &y_bfgs_[0]);
  ro_bfgs_[0] = la::Dot(s_bfgs_[0].n_elements(), 
      s_bfgs_[0].ptr(), y_bfgs_[0].ptr());
  for(index_t i=0; i<mem_bfgs_; i++) {
    ComputeBFGS_<OPT_PARAM_>(&step, gradient_, i);
    ComputeGradient_<OPT_PARAM_>(coordinates_, &gradient_);
    UpdateBFGS_();
    previous_gradient_.CopyValues(gradient_);
    previous_coordinates_.CopyValues(coordinates_);
    ComputeFeasibilityError_<OPT_PARAM_>(&distance_constraint, &centering_constraint);
    NOTIFY("%i Feasibility error: %lg + %lg\n", i, distance_constraint, 
        centering_constraint);
  } 
  max_violation_of_distance_constraint_= DBL_MAX*ComputeFeasibilityError_<OPT_PARAM_>();
  NOTIFY("Now starting optimizing with BFGS...\n");
  ComputeFeasibilityError_<OPT_PARAM_>(&distance_constraint, &centering_constraint);
  double old_distance_constraint = distance_constraint;
  for(index_t it1=0; it1<max_iterations_; it1++) {  
    for(index_t it2=0; it2<max_iterations_; it2++) {
      ComputeBFGS_<OPT_PARAM_>(&step, gradient_, mem_bfgs_);
      ComputeGradient_<OPT_PARAM_>(coordinates_, &gradient_);
      ComputeFeasibilityError_<OPT_PARAM_>(&distance_constraint, 
                                             &centering_constraint);
      double norm_gradient = la::LengthEuclidean(gradient_.n_elements(),
                                                 gradient_.ptr());
      double augmented_lagrangian=ComputeLagrangian_<OPT_PARAM_>(coordinates_);
      double objective =ComputeObjective_<OPT_PARAM_>(coordinates_);
      NOTIFY("Iteration: %"LI"d : %"LI"d, feasibility error (dist)): %lg\n"
             "    feasibility error (center): %lg \n"
             "    objective  : %lg\n"
             "    grad       : %lg\n"
             "    step       : %lg\n"
             "    lang_mult  : %lg\n"
             "    sigma      : %lg\n"
             "    lagrangian : %lg\n", it1, it2, 
               distance_constraint, 
               centering_constraint,
               objective,
               norm_gradient,
               step,
               la::LengthEuclidean(lagrange_mult_),
               sigma_,
               augmented_lagrangian);
     
      if (fabs(old_distance_constraint - distance_constraint) < distance_tolerance_
          || sigma_>=1e50 || 
          distance_constraint > max_violation_of_distance_constraint_) {
        break;
      }
      UpdateBFGS_();
      previous_coordinates_.CopyValues(coordinates_);
      previous_gradient_.CopyValues(gradient_);
      old_distance_constraint = distance_constraint;
    }
    if (sigma_>1e50) {
      char buffer[1024];
      sprintf(buffer, "Converged_"
          "Objective_function_%lg_"
          "Distances_constraints_%lg_Centering_constraint_%lg",
           ComputeObjective_<OPT_PARAM_>(coordinates_),
           distance_constraint, centering_constraint);
      result_summary_=buffer;
      printf("%s\n", result_summary_.c_str());
      NOTIFY("Converged !!\n");
      NOTIFY("Objective function: %lg\n", ComputeObjective_<OPT_PARAM_>(coordinates_));
      NOTIFY("Distances constraints: %lg, Centering constraint: %lg\n", 
              distance_constraint, centering_constraint);
      Vector var;
      Variance_(coordinates_, &var);
      printf("Variance: ");
      for(index_t i=0; i<new_dimension_; i++) {
        printf("%lg ", var[i]);
      }
      printf("\n");
      return;
    }
    if (sigma_>1e120) {
      break;
    }
    //UpdateLagrangeMultStochastic_();
    UpdateLagrangeMult_<OPT_PARAM_>();
    ComputeGradient_<OPT_PARAM_>(coordinates_, &gradient_);
  }
    NOTIFY("Didn't converge, maximum number of iterations reached !!\n");
    NOTIFY("Objective function: %lg\n", ComputeObjective_<OPT_PARAM_>(coordinates_));
    NOTIFY("Distances constraints: %lg, Centering constraint: %lg\n", 
              distance_constraint, centering_constraint);
    printf("Variance: ");
    Vector var;
    Variance_(coordinates_, &var);

    for(index_t i=0; i<new_dimension_; i++) {
      printf("%lg ", var[i]);
    }
    printf("\n");
}

void NonConvexMVU::set_knns(index_t knns) {
 knns_=knns;
}

void NonConvexMVU::set_kfns(index_t kfns) {
  kfns_=kfns;
}

void NonConvexMVU::set_leaf_size(index_t leaf_size) {
  leaf_size_ =leaf_size;
}
 
void NonConvexMVU::set_eta(double eta) {
  eta_ = eta;
}

void NonConvexMVU::set_gamma(double gamma) {
  gamma_ = gamma;
}

void NonConvexMVU::set_sigma(double sigma) {
  sigma_=sigma;
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

void NonConvexMVU::set_distance_tolerance(double tolerance) {
  distance_tolerance_ = tolerance;
}

void NonConvexMVU::set_wolfe_sigma(double wolfe_sigma1, double wolfe_sigma2) {
  if (unlikely(wolfe_sigma1>=wolfe_sigma2_)) {
    FATAL("Wolfe sigma1 %lg should be less than sigma2 %lg", 
        wolfe_sigma1, wolfe_sigma2);
  }
  DEBUG_ASSERT(wolfe_sigma1>0);
  DEBUG_ASSERT(wolfe_sigma1<1);
  DEBUG_ASSERT(wolfe_sigma2>0);
  DEBUG_ASSERT(wolfe_sigma2<1);
  wolfe_sigma1_ = wolfe_sigma1;
  wolfe_sigma2_ = wolfe_sigma2;
}

void NonConvexMVU::set_wolfe_beta(double wolfe_beta) {
  wolfe_beta_ = wolfe_beta;
}

void NonConvexMVU::set_mem_bfgs(index_t mem_bfgs) {
  mem_bfgs_ = mem_bfgs;
}

Matrix &NonConvexMVU::coordinates() {
  return coordinates_;
}

std::string NonConvexMVU::result_summary() {
  return result_summary_;
}

TEMPLATE_TAG_
void NonConvexMVU::InitOptimization_() {
  if (unlikely(new_dimension_<0)) {
    FATAL("You forgot to set the new dimension\n");
  }
  NOTIFY("Initializing optimization ...\n");
  coordinates_.Init(new_dimension_, num_of_points_);
  if (gradient_mode==DeterministicGrad) {
    gradient_.Init(new_dimension_, num_of_points_);
  }
  for(index_t i=0; i< coordinates_.n_rows(); i++) {
    for(index_t j=0; j<coordinates_.n_cols(); j++) {
      coordinates_.set(i, j, data_.get(i,j));
    }
  }
  RemoveMean_(coordinates_);
  if (gradient_mode==DeterministicGrad) {
    lagrange_mult_.Init(num_of_nearest_pairs_);
  } 
  if (gradient_mode == StochasticGrad) {
    lagrange_mult_.Init(knns_ * num_of_points_);
  }
  if (constraints_mode==EqualityOnNearest) {
    for(index_t i=0; i<lagrange_mult_.length(); i++) {
      lagrange_mult_[i]=math::Random(0.1, 1.0)-0.5; 
    }
  } 
  if (constraints_mode == InequalityOnNearest) {
    for(index_t i=0; i<lagrange_mult_.length(); i++) {
      lagrange_mult_[i]=math::Random(0.1, 1.0);
    }
  } 
  if (objective_mode==Feasibility) {
    lagrange_mult_.SetAll(0.0);
  }
}

TEMPLATE_TAG_
void NonConvexMVU::UpdateLagrangeMult_() {
  double distance_constraint;
  double centering_constraint;
  if (objective_mode == Feasibility) {
    return;
  }
  ComputeFeasibilityError_<OPT_PARAM_>(&distance_constraint, &centering_constraint);
  double feasibility_error = distance_constraint + centering_constraint;
  if (feasibility_error< eta_ * previous_feasibility_error_) {
    if (gradient_mode==DeterministicGrad) {
      for(index_t i=0; i<num_of_nearest_pairs_; i++) {
        index_t n1=nearest_neighbor_pairs_[i].first;
        index_t n2=nearest_neighbor_pairs_[i].second;
        double *point1 = coordinates_.GetColumnPtr(n1);
        double *point2 = coordinates_.GetColumnPtr(n2);
        double dist_diff =la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                              -nearest_distances_[i];
        // this is for equality constraints
        if (constraints_mode == EqualityOnNearest) {
          lagrange_mult_[i]-=sigma_*dist_diff;
        }
        // this is for inequality constraints
        if (constraints_mode == InequalityOnNearest) {
          lagrange_mult_[i]= max(
              lagrange_mult_[i]-sigma_*dist_diff, 0.0);
        }
      }
    } 
    if (gradient_mode==StochasticGrad) {
      for(index_t i=0; i<num_of_points_; i++) {
        for(index_t k=0; k<knns_; k++) {
          index_t n1=i;
          index_t n2=nearest_neighbors_[i*knns_+k];
          double *point1 = coordinates_.GetColumnPtr(n1);
          double *point2 = coordinates_.GetColumnPtr(n2);
          double dist_diff =la::DistanceSqEuclidean(new_dimension_, 
              point1, point2) 
                                -nearest_distances_[i*knns_+k];
          // this is for equality constraints
          if (constraints_mode == EqualityOnNearest) {
            lagrange_mult_[i*knns_+k]-=sigma_*dist_diff;
          }
          // this is for inequality constraints
          if (constraints_mode == InequalityOnNearest) {
            lagrange_mult_[i*knns_+k]= max(
                lagrange_mult_[i*knns_+k]-sigma_*dist_diff, 0.0);
          }
        }
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

TEMPLATE_TAG_
void NonConvexMVU::LocalSearch_(double *step, Matrix &direction) {
  Matrix temp_coordinates;
  Matrix temp_gradient;
  temp_gradient.Init(new_dimension_, num_of_points_);
  temp_coordinates.Init(coordinates_.n_rows(), coordinates_.n_cols()); 
  double lagrangian1 = ComputeLagrangian_<OPT_PARAM_>(coordinates_);
  double lagrangian2 = 0;
  double beta=wolfe_beta_;
  double dot_product = la::Dot(direction.n_elements(),
                               gradient_.ptr(),
                               direction.ptr());
  double wolfe_factor =  dot_product * wolfe_sigma1_ * wolfe_beta_ * step_size_;
  for(index_t i=0; beta>1e-100; i++) { 
    temp_coordinates.CopyValues(coordinates_);
    la::AddExpert(-step_size_*beta, direction, &temp_coordinates);
    lagrangian2 =  ComputeLagrangian_<OPT_PARAM_>(temp_coordinates);
    if (lagrangian1-lagrangian2 >= wolfe_factor)  {
      ComputeGradient_<OPT_PARAM_>(temp_coordinates, &temp_gradient);
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
  RemoveMean_(coordinates_); 
}

TEMPLATE_TAG_
void NonConvexMVU::ComputeBFGS_(double *step, Matrix &grad, index_t memory) {
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
 LocalSearch_<OPT_PARAM_>(step, temp_direction);
 if (step==0) {
   la::Scale(-1.0, &temp_direction);
   LocalSearch_<OPT_PARAM_>(step, temp_direction);
 }
/*  NOTIFY("gradient_norm = %lg norm_scalar %lg\n", 
         la::Dot(num_of_points_*new_dimension_, 
         temp_gradient.ptr(), temp_gradient.ptr()),
         norm_scale);
*/         
  (*step)*= la::Dot(num_of_points_*new_dimension_, 
         temp_direction.ptr(), temp_direction.ptr());
}

void NonConvexMVU::UpdateBFGS_() {
  index_bfgs_ = (index_bfgs_ - 1 + mem_bfgs_ ) % mem_bfgs_;
  UpdateBFGS_(index_bfgs_);
}

void NonConvexMVU::UpdateBFGS_(index_t index_bfgs) {
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
}

// PROBLEM_TYPE:
// 0 = feasibility problem
// 1 = equality constraints
// 2 = inequality constraints
// 3 = furthest neighbor objective
TEMPLATE_TAG_
double NonConvexMVU::ComputeLagrangian_(Matrix &coord) {
  double lagrangian=0;

  if (objective_mode == MaxVariance) {
    for(index_t i=0; i<coord.n_cols(); i++) {
      // we are maximizing the trace or minimize the -trace(RR^T)
      // we dont need it for the feasibility problem
      lagrangian -= la::Dot(new_dimension_, 
                            coord.GetColumnPtr(i),
                            coord.GetColumnPtr(i));
    }
  }
  if (objective_mode == MinVariance) {
    for(index_t i=0; i<coord.n_cols(); i++) {
      // we are minimizing the trace  -trace(RR^T)
      lagrangian += la::Dot(new_dimension_, 
                            coord.GetColumnPtr(i),
                            coord.GetColumnPtr(i));
    }
  }
  if (objective_mode ==MaxFurthestNeighbors) {
    if (gradient_mode==DeterministicGrad) {
      for(index_t i=0; i<num_of_furthest_pairs_; i++) {
        // we are maximizing the distances of the furthest neighbors
        index_t n1=furthest_neighbor_pairs_[i].first;
        index_t n2=furthest_neighbor_pairs_[i].second;
        lagrangian -= la::DistanceSqEuclidean(new_dimension_,
                          coord.GetColumnPtr(n1),
                          coord.GetColumnPtr(n2)); 
      }
    }
    if (gradient_mode==StochasticGrad) {
      for(index_t i=0; i<num_of_points_; i++) {
        for(index_t k=0; k<kfns_; k++) {
          index_t p1=i;
          index_t p2=furthest_neighbors_[p1*kfns_+k];
          lagrangian -= la::DistanceSqEuclidean(new_dimension_,
                          coord.GetColumnPtr(p1),
                          coord.GetColumnPtr(p2));         
        }
      }
    }
  }
  if (gradient_mode==DeterministicGrad) { 
    for(index_t i=0; i<num_of_nearest_pairs_; i++) {
      index_t n1=nearest_neighbor_pairs_[i].first;
      index_t n2=nearest_neighbor_pairs_[i].second;
      double *point1 = coord.GetColumnPtr(n1);
      double *point2 = coord.GetColumnPtr(n2);
      double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                             -nearest_distances_[i];
      if (objective_mode==Feasibility) {
        lagrangian +=  0.5*sigma_*dist_diff*dist_diff; 
      }
      if (constraints_mode==EqualityOnNearest) {
        lagrangian += -lagrange_mult_[i]*dist_diff +  0.5*sigma_*dist_diff*dist_diff;
      }
      if (constraints_mode == InequalityOnNearest) {
        if (sigma_*dist_diff<=lagrange_mult_[i])  {
          lagrangian += -lagrange_mult_[i]*dist_diff +  0.5*sigma_*dist_diff*dist_diff;
        } else {
          lagrangian-=math::Sqr(lagrange_mult_[i])/(2*sigma_);
        }
      }
    }
  }
  if (gradient_mode==StochasticGrad) { 
    for(index_t i=0; i<num_of_points_; i++) {
      for(index_t k=0; k<knns_; k++) {
        index_t n1=i;
        index_t n2=nearest_neighbors_[i*knns_+k];
        double *point1 = coord.GetColumnPtr(n1);
        double *point2 = coord.GetColumnPtr(n2);
        double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                             -nearest_distances_[n1*knns_+k];
        if (objective_mode==Feasibility) {
          lagrangian +=  0.5*sigma_*dist_diff*dist_diff; 
          continue;
        }
        if (constraints_mode==EqualityOnNearest) {
          lagrangian += -lagrange_mult_[n1*knns_+k]*dist_diff +  
            0.5*sigma_*dist_diff*dist_diff;
        }
        if (constraints_mode == InequalityOnNearest) {
          if (sigma_*dist_diff<=lagrange_mult_[n1*knns_+k])  {
            lagrangian += -lagrange_mult_[i*knns_+k]*dist_diff +  
              0.5*sigma_*dist_diff*dist_diff;
          } else {
            lagrangian-=math::Sqr(lagrange_mult_[i*knns_+k])/(2*sigma_);
          }
        }
      }
    }
  }

  
    return 0.5*lagrangian; 
}

TEMPLATE_TAG_
void NonConvexMVU::ComputeFeasibilityError_(double *distance_constraint,
                                            double *centering_constraint) {
  Vector deviations;
  deviations.Init(new_dimension_);
  deviations.SetAll(0.0);
  *distance_constraint=0;
  *centering_constraint=0;
  if (gradient_mode==DeterministicGrad) {
    for(index_t i=0; i<num_of_nearest_pairs_; i++) {
      index_t n1=nearest_neighbor_pairs_[i].first;
      index_t n2=nearest_neighbor_pairs_[i].second;
      double *point1 = coordinates_.GetColumnPtr(n1);
      double *point2 = coordinates_.GetColumnPtr(n2);
      if (constraints_mode == EqualityOnNearest) {
        *distance_constraint += math::Sqr(la::DistanceSqEuclidean(new_dimension_, 
                                                                  point1, point2) 
                                          -nearest_distances_[i]);
      }
      if (constraints_mode == InequalityOnNearest) {
        double dist =(la::DistanceSqEuclidean(new_dimension_, 
                                              point1, point2) 
                      -nearest_distances_[i]);
        if (dist<0) {
          *distance_constraint += math::Sqr(dist); 
        }
      }
    }
  } 
  if (gradient_mode==StochasticGrad) {
    for(index_t i=0; i<num_of_points_; i++) {
      double *point1=coordinates_.GetColumnPtr(i);
      for(index_t k=0; k<knns_; k++) {
        double *point2 = coordinates_.GetColumnPtr(nearest_neighbors_[i*knns_+k]);
        if (constraints_mode == EqualityOnNearest) {
          *distance_constraint += 
            math::Sqr(la::DistanceSqEuclidean(new_dimension_, 
                                              point1, point2) 
                                              -nearest_distances_[i*knns_+k]);
        }
        if (constraints_mode == InequalityOnNearest) {
          double dist =(la::DistanceSqEuclidean(new_dimension_, 
                                                point1, point2) 
                       -nearest_distances_[i*knns_+k]);
          if (dist<0) {
            *distance_constraint += math::Sqr(dist); 
          }
        }
      }
    }
  }  
  for(index_t i=0; i<num_of_points_; i++) { 
    for(index_t k=0; k<new_dimension_; k++) {
      deviations[k] += coordinates_.get(k, i);
    }
  }
 
  for(index_t k=0; k<new_dimension_; k++) {
    *centering_constraint += deviations[k] * deviations[k];
  }
}

TEMPLATE_TAG_
double NonConvexMVU::ComputeFeasibilityError_() {
  double distance_constraint;
  double centering_constraint;
  ComputeFeasibilityError_<OPT_PARAM_>(&distance_constraint, 
                                         &centering_constraint);
  return distance_constraint+centering_constraint;
}

// PROBLEM_TYPE:
// 0 = feasibility problem
// 1 = equality constraints
// 2 = inequality constraints
// 3 = furthest neighbor objective
TEMPLATE_TAG_
void NonConvexMVU::ComputeGradient_(Matrix &coord, Matrix *grad) {
  if (gradient_mode==StochasticGrad) {
    FATAL("You are trying to compute a stochastic gradient, while it is not "
          "possible\n");
  }
  if (objective_mode==Feasibility) {
    grad->SetAll(0.0);
  }
  if (objective_mode == MaxVariance) {
    grad->CopyValues(coord);
    // we need to use -CRR^T because we want to maximize CRR^T
    la::Scale(-1.0, grad);
  }
  if (objective_mode == MinVariance) {
    grad->CopyValues(coord);
  }
  if (objective_mode == MaxFurthestNeighbors) {
    grad->SetAll(0.0);
    for(index_t i=0; i<num_of_furthest_pairs_; i++) {
      // we are maximizing the distances of the furthest neighbors
      index_t n1=furthest_neighbor_pairs_[i].first;
      index_t n2=furthest_neighbor_pairs_[i].second;
      double diff[new_dimension_];
      la::SubOverwrite(new_dimension_,
                       coord.GetColumnPtr(n2), 
                       coord.GetColumnPtr(n1),
                       diff);
      la::AddExpert(new_dimension_, -1.0, diff, grad->GetColumnPtr(n1));
      la::AddTo(new_dimension_, diff, grad->GetColumnPtr(n2));
    }
  }
 
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    double a_i_r[new_dimension_];
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coord.GetColumnPtr(n1);
    double *point2 = coord.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                           -nearest_distances_[i];
    la::SubOverwrite(new_dimension_, point2, point1, a_i_r);
    // feasibility problem
    if (objective_mode==Feasibility) {
      la::AddExpert(new_dimension_,
                    dist_diff*sigma_, 
          a_i_r, 
          grad->GetColumnPtr(n1));
          la::AddExpert(new_dimension_,
                       -dist_diff*sigma_, 
          a_i_r, 
          grad->GetColumnPtr(n2));
 
    } 
    // equality constraints
    if (constraints_mode == EqualityOnNearest) {     
      la::AddExpert(new_dimension_,
          -lagrange_mult_[i]+dist_diff*sigma_,
          a_i_r, 
          grad->GetColumnPtr(n1));
      la::AddExpert(new_dimension_,
          lagrange_mult_[i]-dist_diff*sigma_,
          a_i_r, 
          grad->GetColumnPtr(n2));
    }
    // inequality constraints
    if (constraints_mode == InequalityOnNearest) {
      if (lagrange_mult_[i]>=dist_diff*sigma_) {
        la::AddExpert(new_dimension_,
            -lagrange_mult_[i]+dist_diff*sigma_,
            a_i_r, 
            grad->GetColumnPtr(n1));
        la::AddExpert(new_dimension_,
            lagrange_mult_[i]-dist_diff*sigma_, 
            a_i_r, 
            grad->GetColumnPtr(n2));
      }
    }
  }
}

TEMPLATE_TAG_
void NonConvexMVU::ComputePairGradient_(index_t p1, index_t chosen_neighbor, 
    Matrix &coord, Vector *gradient1, Vector *gradient2) {
  
  index_t p2=nearest_neighbors_[p1*knns_+chosen_neighbor];
  double p1_p2_dist = nearest_distances_[p1*knns_+chosen_neighbor]; 
  Vector point1, point2;
  coord.MakeColumnVector(p1, &point1);
  coord.MakeColumnVector(p2, &point2);
  
  if (objective_mode==Feasibility) {
    gradient1->SetAll(0.0);
    gradient2->SetAll(0.0);
  }
  if (objective_mode == MaxVariance) {
    gradient1->CopyValues(point1);
    gradient2->CopyValues(point2);
    // we need to use -CRR^T because we want to maximize CRR^T
    la::Scale(-1.0, gradient1);
    la::Scale(-1.0, gradient2);
  }
  if (objective_mode == MinVariance) {
    gradient1->CopyValues(point1);
    gradient2->CopyValues(point2);
  }
  if (objective_mode == MaxFurthestNeighbors) {
    // we are maximizing the distances of the furthest neighbors
    index_t n1=furthest_neighbors_[p1*kfns_];
    index_t n2=furthest_neighbors_[p2*kfns_];
    la::SubOverwrite(new_dimension_,
                     point1.ptr(), 
                     coord.GetColumnPtr(n1),
                     gradient1->ptr());
    la::SubOverwrite(new_dimension_,
                     point2.ptr(), 
                     coord.GetColumnPtr(n2),
                     gradient2->ptr());
  
  }
 
  double a_i_r[new_dimension_];
  double dist_diff = la::DistanceSqEuclidean(point1, point2) 
                         -p1_p2_dist;
  la::SubOverwrite(new_dimension_, point2.ptr(), point1.ptr(), a_i_r);
  // feasibility problem
  if (objective_mode==Feasibility) {
    la::AddExpert(new_dimension_,
                  dist_diff*sigma_, 
        a_i_r, 
        gradient1->ptr());
        la::AddExpert(new_dimension_,
                     -dist_diff*sigma_, 
        a_i_r, 
        gradient2->ptr());
        return;
  } 
  // equality constraints
  if (constraints_mode == EqualityOnNearest) {     
    la::AddExpert(new_dimension_,
        -lagrange_mult_[p1*knns_+chosen_neighbor]+dist_diff*sigma_,
        a_i_r, 
        gradient1->ptr());
    la::AddExpert(new_dimension_,
        lagrange_mult_[p1*knns_+chosen_neighbor]-dist_diff*sigma_,
        a_i_r, 
        gradient2->ptr());
  }
  // inequality constraints
  if (constraints_mode==InequalityOnNearest) {
    if (lagrange_mult_[p1*knns_+chosen_neighbor]>=dist_diff*sigma_) {
      la::AddExpert(new_dimension_,
          -lagrange_mult_[p1*knns_+chosen_neighbor]+dist_diff*sigma_,
          a_i_r, 
          gradient1->ptr());
      la::AddExpert(new_dimension_,
          lagrange_mult_[p1*knns_+chosen_neighbor]-dist_diff*sigma_, 
          a_i_r, 
          gradient2->ptr());
    }
  }
} 

TEMPLATE_TAG_
double NonConvexMVU::ComputeObjective_(Matrix &coord) {
  double variance=0;
  if (objective_mode==MaxVariance) {
    for(index_t i=0; i< coord.n_cols(); i++) {
      variance-=la::Dot(new_dimension_, 
                        coord.GetColumnPtr(i),
                        coord.GetColumnPtr(i));
    }
  }
  if (objective_mode==MinVariance) {
    for(index_t i=0; i< coord.n_cols(); i++) {
      variance+=la::Dot(new_dimension_, 
                        coord.GetColumnPtr(i),
                        coord.GetColumnPtr(i));
    }
  }
  if (objective_mode==MaxFurthestNeighbors) {
    for(index_t i=0; i<num_of_furthest_pairs_; i++) {
      // we are maximizing the distances of the furthest neighbors
      index_t n1=furthest_neighbor_pairs_[i].first;
      index_t n2=furthest_neighbor_pairs_[i].second;
      variance -= la::DistanceSqEuclidean(new_dimension_,
                        coord.GetColumnPtr(n1),
                        coord.GetColumnPtr(n2)); 
    }
  }

  return variance;
}

// Assumes zero mean
// variance should not be initialized
void NonConvexMVU::Variance_(Matrix &coord, Vector *variance) {
  variance->Init(coord.n_rows());
  for(index_t i=0; i<new_dimension_; i++) {
    (*variance)[i]=la::LengthEuclidean(new_dimension_, 
                                       coord.GetColumnPtr(i))/num_of_points_;
  }
}

void NonConvexMVU::RemoveMean_(Matrix &mat) {
  Vector dimension_sums;
  dimension_sums.Init(mat.n_rows());
  dimension_sums.SetAll(0.0);
  for(index_t i=0; i<mat.n_cols(); i++) {
    la::AddTo(mat.n_rows(), mat.GetColumnPtr(i), dimension_sums.ptr());
  }
  for(index_t i=0; i<mat.n_cols(); i++)  {
    la::AddExpert(mat.n_rows(), -1.0/mat.n_cols(), 
        dimension_sums.ptr(), 
        mat.GetColumnPtr(i));
  }
}

void NonConvexMVU::ConsolidateNeighbors_(ArrayList<index_t> &from_tree_ind,
    ArrayList<double>  &from_tree_dist,
    index_t num_of_neighbors,
    ArrayList<std::pair<index_t, index_t> > *neighbor_pairs,
    ArrayList<double> *distances,
    index_t *num_of_pairs) {
  
  *num_of_pairs=0;
  neighbor_pairs->Init();
  distances->Init();
  bool skip=false;
  for(index_t i=0; i<num_of_points_; i++) {
    for(index_t k=0; k<num_of_neighbors; k++) {  
      index_t n1=i;                         //neighbor 1
      index_t n2=from_tree_ind[i*num_of_neighbors+k];  //neighbor 2
      if (n1 > n2) {
        for(index_t n=0; n<num_of_neighbors; n++) {
          if (from_tree_ind[n2*num_of_neighbors+n] == n1) {
            skip=true;
            break;
          }
        }
      }  
      if (skip==false) {
        *num_of_pairs+=1;
        neighbor_pairs->AddBackItem(std::make_pair(n1, n2));
        distances->AddBackItem(from_tree_dist[i*num_of_neighbors+k]);
      }
      skip=false;
    }
  }
}
     
 
