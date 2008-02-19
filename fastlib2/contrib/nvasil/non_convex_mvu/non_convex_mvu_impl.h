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
 gamma_ =1.3;
 sigma_ = 1e3;
 step_size_ = 2; 
 max_iterations_ = 100000;
 distance_tolerance_ = 1e-1;
 gradient_tolerance_ = 1e-20;
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
  RemoveMean_(data_);
  num_of_points_ = data_.n_cols();
  NOTIFY("Data loaded ...\n");
  NOTIFY("Building tree with data ...\n");
  allknn_.Init(data_, leaf_size_, knns_);
  NOTIFY("Tree built ...\n");
  NOTIFY("Computing neighborhoods ...\n");
  ArrayList<index_t> from_tree_neighbors;
  ArrayList<double>  from_tree_distances;
  allknn_.ComputeNeighbors(&from_tree_neighbors,
                           &from_tree_distances);
  NOTIFY("Neighborhoods computed...\n");
  NOTIFY("Consolidating neighbors...\n");
  ConsolidateNeighbors_(from_tree_neighbors,
                        from_tree_distances,
                        &neighbor_pairs_,
                        &distances_,
                        &num_of_pairs_);
  previous_feasibility_error_ = DBL_MAX;
}

// PROBLEM_TYPE:
// 0 Feasibility problem
// 1 Equality Constraints
// 2 Inequality Constraints
template<int PROBLEM_TYPE>
void NonConvexMVU::ComputeLocalOptimumBFGS() {
  double distance_constraint;
  double centering_constraint;
  double step; 
  if (unlikely(mem_bfgs_<0)) {
    FATAL("You forgot to initialize the memory for BFGS\n");
  }
  InitOptimization_<PROBLEM_TYPE>();
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
  ComputeGradient_<PROBLEM_TYPE>();
  previous_gradient_.Copy(gradient_);
  previous_coordinates_.Copy(coordinates_);
  LocalSearch_<PROBLEM_TYPE>(&step, gradient_);
  ComputeGradient_<PROBLEM_TYPE>();
  la::SubOverwrite(previous_coordinates_, coordinates_, &s_bfgs_[0]);
  la::SubOverwrite(previous_gradient_, gradient_, &y_bfgs_[0]);
  ro_bfgs_[0] = la::Dot(s_bfgs_[0].n_elements(), 
      s_bfgs_[0].ptr(), y_bfgs_[0].ptr());
  for(index_t i=0; i<mem_bfgs_; i++) {
    //LocalSearch_<PROBLEM_TYPE>(&step, gradient_);
    ComputeBFGS_<PROBLEM_TYPE>(&step, gradient_, i);
    ComputeGradient_<PROBLEM_TYPE>();
    UpdateBFGS_();
    previous_gradient_.CopyValues(gradient_);
    previous_coordinates_.CopyValues(coordinates_);
    ComputeFeasibilityError_<PROBLEM_TYPE>(&distance_constraint, &centering_constraint);
    NOTIFY("%i Feasibility error: %lg + %lg\n", i, distance_constraint, 
        centering_constraint);
  } 
  NOTIFY("Now starting optimizing with BFGS...\n");
  ComputeFeasibilityError_<PROBLEM_TYPE>(&distance_constraint, &centering_constraint);
  double old_feasibility_error = previous_feasibility_error_;
  for(index_t it1=0; it1<max_iterations_; it1++) {  
    for(index_t it2=0; it2<max_iterations_; it2++) {
      ComputeBFGS_<PROBLEM_TYPE>(&step, gradient_, mem_bfgs_);
      ComputeGradient_<PROBLEM_TYPE>();
      ComputeFeasibilityError_<PROBLEM_TYPE>(&distance_constraint, 
                                             &centering_constraint);
 
      old_feasibility_error = distance_constraint+centering_constraint;
      double norm_gradient = la::LengthEuclidean(gradient_.n_elements(),
                                                 gradient_.ptr());
      double augmented_lagrangian=ComputeLagrangian_<PROBLEM_TYPE>(coordinates_);
      double objective =ComputeObjective_<PROBLEM_TYPE>(coordinates_);
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
     // step=la::DistanceSqEuclidean(new_dimension_ * num_of_points_,
     //     previous_coordinates_.ptr(), coordinates_.ptr());
      if (step < gradient_tolerance_ ||
          distance_constraint < distance_tolerance_){
        break;
      }
      UpdateBFGS_();
      previous_coordinates_.CopyValues(coordinates_);
      previous_gradient_.CopyValues(gradient_);
    }
    if (distance_constraint < distance_tolerance_) {
      NOTIFY("Converged !!\n");
      NOTIFY("Objective function: %lg\n", ComputeObjective_<PROBLEM_TYPE>(coordinates_));
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
    //UpdateLagrangeMultStochastic_();
    UpdateLagrangeMult_<PROBLEM_TYPE>();
    ComputeGradient_<PROBLEM_TYPE>();
  }
    NOTIFY("Didn't converge, maximum number of iterations reached !!\n");
    NOTIFY("Objective function: %lg\n", ComputeObjective_<PROBLEM_TYPE>(coordinates_));
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

void NonConvexMVU::set_gradient_tolerance(double tolerance) {
  gradient_tolerance_ = tolerance;
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

Matrix &NonConvexMVU::coordinates() {
  return coordinates_;
}

template<int PROBLEM_TYPE>
void NonConvexMVU::InitOptimization_() {
  if (unlikely(new_dimension_<0)) {
    FATAL("You forgot to set the new dimension\n");
  }
  NOTIFY("Initializing optimization ...\n");
  coordinates_.Init(new_dimension_, num_of_points_);
  gradient_.Init(new_dimension_, num_of_points_);
  for(index_t i=0; i< coordinates_.n_rows(); i++) {
    for(index_t j=0; j<coordinates_.n_cols(); j++) {
      coordinates_.set(i, j, data_.get(i,j));
    }
  }
  data::Save("init.csv", coordinates_);
  RemoveMean_(coordinates_);
  lagrange_mult_.Init(num_of_pairs_);
  if (PROBLEM_TYPE==0) {
    lagrange_mult_.SetAll(0.0);
  }
  if (PROBLEM_TYPE==1) {
    for(index_t i=0; i<lagrange_mult_.length(); i++) {
      lagrange_mult_[i]=math::Random(0.1, 1.0)-0.5; 
    }
  } 
  if (PROBLEM_TYPE==2) {
    for(index_t i=0; i<lagrange_mult_.length(); i++) {
      lagrange_mult_[i]=math::Random(0.1, 1.0);
    }
  } 
}

template<int PROBLEM_TYPE>
void NonConvexMVU::UpdateLagrangeMult_() {
  double distance_constraint;
  double centering_constraint;
  //  in the case of feasibility problem  we do nothing
  if (PROBLEM_TYPE==0) {
    return;
  }
  ComputeFeasibilityError_<PROBLEM_TYPE>(&distance_constraint, &centering_constraint);
  double feasibility_error = distance_constraint + centering_constraint;
  if (feasibility_error< eta_ * previous_feasibility_error_) {
    for(index_t i=0; i<num_of_pairs_; i++) {
      index_t n1=neighbor_pairs_[i].first;
      index_t n2=neighbor_pairs_[i].second;
      double *point1 = coordinates_.GetColumnPtr(n1);
      double *point2 = coordinates_.GetColumnPtr(n2);
      double dist_diff =la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                           -distances_[i];
      // this is for equality constraints
      if (PROBLEM_TYPE==1) {
        lagrange_mult_[i]-=sigma_*dist_diff;
      }
      // this is for inequality constraints
      if (PROBLEM_TYPE==2) {
        lagrange_mult_[i]= max(
            lagrange_mult_[i]-sigma_*dist_diff, 0.0);
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

template<int PROBLEM_TYPE>
void NonConvexMVU::LocalSearch_(double *step, Matrix &grad) {
  Matrix temp_coordinates;
  temp_coordinates.Init(coordinates_.n_rows(), coordinates_.n_cols()); 
  double lagrangian1 = ComputeLagrangian_<PROBLEM_TYPE>(coordinates_);
  double lagrangian2 = 0;
  double beta=armijo_beta_;
  double gradient_norm = la::LengthEuclidean(grad.n_rows()
                                             * grad.n_cols(),
                                             grad.ptr());
  double armijo_factor =  gradient_norm * armijo_sigma_ * armijo_beta_ * step_size_;
  for(index_t i=0; ; i++) {
    temp_coordinates.CopyValues(coordinates_);
    la::AddExpert(-step_size_*beta/gradient_norm, grad, &temp_coordinates);
    lagrangian2 =  ComputeLagrangian_<PROBLEM_TYPE>(temp_coordinates);
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
  la::AddExpert(-0.00001/gradient_norm, gradient_, &temp_coordinates);
  lagrangian2 =  ComputeLagrangian_<PROBLEM_TYPE>(temp_coordinates);
*/
//  NOTIFY("step_size: %lg, sigma: %lg\n", beta * step_size_, sigma_);
//  NOTIFY("lagrangian1 - lagrangian2 = %lg\n", lagrangian1-lagrangian2);
//  NOTIFY("lagrangian2: %lg, Objective: %lg\n", lagrangian2,
//                                               ComputeObjective_(temp_coordinates));
  coordinates_.CopyValues(temp_coordinates);   
  RemoveMean_(coordinates_); 
}

template<int PROBLEM_TYPE>
void NonConvexMVU::ComputeBFGS_(double *step, Matrix &grad, index_t memory) {
  Vector alpha;
  alpha.Init(mem_bfgs_);
  Matrix scaled_y;
  scaled_y.Init(new_dimension_, num_of_points_);
  index_t num=0;
  Matrix temp_gradient(grad);
  for(index_t i=index_bfgs_, num=0; num<memory; i=(i+1+mem_bfgs_)%mem_bfgs_, num++) {
   // printf("i:%i  index_bfgs_:%i\n", i, index_bfgs_);
    alpha[i] = la::Dot(new_dimension_ * num_of_points_,
                       s_bfgs_[i].ptr(), 
                       temp_gradient.ptr());
    alpha[i] *= ro_bfgs_[i];
    scaled_y.CopyValues(y_bfgs_[i]);
    la::Scale(alpha[i], &scaled_y);
    la::SubFrom(scaled_y, &temp_gradient);
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
  la::Scale(norm_scale, &temp_gradient);
  Matrix scaled_s;
  double beta;
  scaled_s.Init(new_dimension_, num_of_points_);
  num=0;
  for(index_t j=(index_bfgs_+memory-1)%mem_bfgs_, num=0; num<memory; 
      num++, j=(j-1+mem_bfgs_)%mem_bfgs_) {
   // printf("j:%i  index_bfgs_:%i\n", j, index_bfgs_);

    beta = la::Dot(new_dimension_ * num_of_points_, 
                   y_bfgs_[j].ptr(),
                   temp_gradient.ptr());
    beta *= ro_bfgs_[j];
    scaled_s.CopyValues(s_bfgs_[j]);
    la::Scale(alpha[j]-beta, &scaled_s);
    la::AddTo(scaled_s, &temp_gradient);
  }
 LocalSearch_<PROBLEM_TYPE>(step, temp_gradient);
/*  NOTIFY("gradient_norm = %lg norm_scalar %lg\n", 
         la::Dot(num_of_points_*new_dimension_, 
         temp_gradient.ptr(), temp_gradient.ptr()),
         norm_scale);
*/         
  (*step)*= la::Dot(num_of_points_*new_dimension_, 
         temp_gradient.ptr(), temp_gradient.ptr());
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
template<int PROBLEM_TYPE>
double NonConvexMVU::ComputeLagrangian_(Matrix &coord) {
  double lagrangian=0;
    
  if (PROBLEM_TYPE==1) {
    for(index_t i=0; i<coord.n_cols(); i++) {
      // we are maximizing the trace or minimize the -trace(RR^T)
      // we dont need it for the feasibility problem
      lagrangian -= la::Dot(new_dimension_, 
                            coord.GetColumnPtr(i),
                            coord.GetColumnPtr(i));
    }
  }
  if (PROBLEM_TYPE==2) {
    for(index_t i=0; i<coord.n_cols(); i++) {
      // we are maximizing the trace or minimize the -trace(RR^T)
      // we dont need it for the feasibility problem
      lagrangian += la::Dot(new_dimension_, 
                            coord.GetColumnPtr(i),
                            coord.GetColumnPtr(i));
    }
  }
  for(index_t i=0; i<num_of_pairs_; i++) {
    index_t n1=neighbor_pairs_[i].first;
    index_t n2=neighbor_pairs_[i].second;
    double *point1 = coord.GetColumnPtr(n1);
    double *point2 = coord.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                          -distances_[i];
    if (PROBLEM_TYPE==0) {
      lagrangian +=  0.5*sigma_*dist_diff*dist_diff; 
    }
    if (PROBLEM_TYPE==1) {
      lagrangian += -lagrange_mult_[i]*dist_diff +  0.5*sigma_*dist_diff*dist_diff;
    }
    if (PROBLEM_TYPE==2) {
      if (sigma_*dist_diff<=lagrange_mult_[i])  {
          lagrangian += -lagrange_mult_[i]*dist_diff +  0.5*sigma_*dist_diff*dist_diff;
      } else {
        lagrangian-=math::Sqr(lagrange_mult_[i])/(2*sigma_);
      }
    }
  }
  return 0.5*lagrangian; 
}

template<int PROBLEM_TYPE>
void NonConvexMVU::ComputeFeasibilityError_(double *distance_constraint,
                                            double *centering_constraint) {
  Vector deviations;
  deviations.Init(new_dimension_);
  deviations.SetAll(0.0);
  *distance_constraint=0;
  *centering_constraint=0;
  for(index_t i=0; i<num_of_pairs_; i++) {
    index_t n1=neighbor_pairs_[i].first;
    index_t n2=neighbor_pairs_[i].second;
    double *point1 = coordinates_.GetColumnPtr(n1);
    double *point2 = coordinates_.GetColumnPtr(n2);
    if (PROBLEM_TYPE !=2) {
      *distance_constraint += math::Sqr(la::DistanceSqEuclidean(new_dimension_, 
                                                                point1, point2) 
                                        -distances_[i]);
    }
    if (PROBLEM_TYPE == 2) {
      double dist =(la::DistanceSqEuclidean(new_dimension_, 
                                            point1, point2) 
                    -distances_[i]);
      if (dist<0) {
        *distance_constraint += math::Sqr(dist); 
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

template<int PROBLEM_TYPE>
double NonConvexMVU::ComputeFeasibilityError_() {
  double distance_constraint;
  double centering_constraint;
  ComputeFeasibilityError_<PROBLEM_TYPE>(&distance_constraint, 
                                         &centering_constraint);
  return distance_constraint+centering_constraint;
}

// PROBLEM_TYPE:
// 0 = feasibility problem
// 1 = equality constraints
// 2 = inequality constraints
template<int PROBLEM_TYPE>
void NonConvexMVU::ComputeGradient_() {
  if (PROBLEM_TYPE==0) {
    gradient_.SetAll(0.0);
  }
  if (PROBLEM_TYPE==1) {
    gradient_.CopyValues(coordinates_);
    // we need to use -CRR^T because we want to maximize CRR^T
    la::Scale(-1.0, &gradient_);
  }
  if (PROBLEM_TYPE==2){
    gradient_.CopyValues(coordinates_);
  }
 for(index_t i=0; i<num_of_pairs_; i++) {
   double a_i_r[new_dimension_];
   index_t n1=neighbor_pairs_[i].first;
   index_t n2=neighbor_pairs_[i].second;
   double *point1 = coordinates_.GetColumnPtr(n1);
   double *point2 = coordinates_.GetColumnPtr(n2);
   double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                          -distances_[i];
   la::SubOverwrite(new_dimension_, point2, point1, a_i_r);
   // feasibility problem
   if (PROBLEM_TYPE==0) {
     la::AddExpert(new_dimension_,
                   dist_diff*sigma_, 
         a_i_r, 
         gradient_.GetColumnPtr(n1));
         la::AddExpert(new_dimension_,
                      -dist_diff*sigma_, 
          a_i_r, 
          gradient_.GetColumnPtr(n2));
 
    } 
    // equality constraints
    if (PROBLEM_TYPE==1) {     
      la::AddExpert(new_dimension_,
          -lagrange_mult_[i]+dist_diff*sigma_,
          a_i_r, 
          gradient_.GetColumnPtr(n1));
      la::AddExpert(new_dimension_,
          lagrange_mult_[i]-dist_diff*sigma_,
          a_i_r, 
          gradient_.GetColumnPtr(n2));
    }
    // inequality constraints
    if (PROBLEM_TYPE==2) {
      if (lagrange_mult_[i]>=dist_diff*sigma_) {
        la::AddExpert(new_dimension_,
            -lagrange_mult_[i]+dist_diff*sigma_,
            a_i_r, 
            gradient_.GetColumnPtr(n1));
        la::AddExpert(new_dimension_,
            lagrange_mult_[i]-dist_diff*sigma_, //***
            a_i_r, 
            gradient_.GetColumnPtr(n2));
      }
    }
  }
}
template<int PROBLEM_TYPE>
double NonConvexMVU::ComputeObjective_(Matrix &coord) {
  double variance=0;
  if (PROBLEM_TYPE==1) {
    for(index_t i=0; i< coord.n_cols(); i++) {
      variance-=la::Dot(new_dimension_, 
                        coord.GetColumnPtr(i),
                        coord.GetColumnPtr(i));
    }
  }
  if (PROBLEM_TYPE==2) {
    for(index_t i=0; i< coord.n_cols(); i++) {
      variance+=la::Dot(new_dimension_, 
                        coord.GetColumnPtr(i),
                        coord.GetColumnPtr(i));
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
    ArrayList<std::pair<index_t, index_t> > *neighbor_pairs,
    ArrayList<double> *distances,
    index_t *num_of_pairs) {
  
  *num_of_pairs=0;
  neighbor_pairs->Init();
  distances->Init();
  bool skip=false;
  for(index_t i=0; i<num_of_points_; i++) {
    for(index_t k=0; k<knns_; k++) {  
      index_t n1=i;                         //neighbor 1
      index_t n2=from_tree_ind[i*knns_+k];  //neighbor 2
      if (n1 > n2) {
        for(index_t n=0; n<knns_; n++) {
          if (from_tree_ind[n2*knns_+n] == n1) {
            skip=true;
            break;
          }
        }
      }  
      if (skip==false) {
        *num_of_pairs+=1;
        neighbor_pairs->AddBackItem(std::make_pair(n1, n2));
        distances->AddBackItem(from_tree_dist[i*knns_+k]);
      }
      skip=false;
    }
  }
}
     
 
