/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/*
 * =====================================================================================
 * 
 *       Filename:  mvu_objectives_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  03/05/2008 04:20:27 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#include <cerrno>
void MaxVariance::Init(datanode *module, Matrix &data) {
  module_=module;
  knns_ = fx_param_int(module_, "knns", 5);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  new_dimension_ = fx_param_int_req(module_, "new_dimension");
  num_of_points_ = data.n_cols();
  NOTIFY("Data loaded ...\n");
  NOTIFY("Nearest neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  if (knns_==0) {
    allknn_.Init(data, leaf_size_, MAX_KNNS); 
  } else {
    allknn_.Init(data, leaf_size_, knns_); 
  }
  NOTIFY("Tree built ...\n");
  NOTIFY("Computing neighborhoods ...\n");
  ArrayList<index_t> from_tree_neighbors;
  ArrayList<double>  from_tree_distances;
  allknn_.ComputeNeighbors(&from_tree_neighbors,
                           &from_tree_distances);
  NOTIFY("Neighborhoods computed...\n");
  if (knns_==0) {
    NOTIFY("Auto-tuning the knn...\n" );
    MaxVarianceUtils::EstimateKnns(from_tree_neighbors,
        from_tree_distances,
        MAX_KNNS, 
        data.n_cols(),
        data.n_rows(),
        &knns_); 
    NOTIFY("Optimum knns is %i", knns_);
    fx_format_result(module_, "optimum_knns", "%i",knns_);
    NOTIFY("Consolidating neighbors...\n");
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
        from_tree_distances,
        MAX_KNNS,
        knns_,
        &nearest_neighbor_pairs_,
        &nearest_distances_,
        &num_of_nearest_pairs_);
  } else { 
    NOTIFY("Consolidating neighbors...\n");
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
        from_tree_distances,
        knns_,
        knns_,
        &nearest_neighbor_pairs_,
        &nearest_distances_,
        &num_of_nearest_pairs_);
  }
 
  eq_lagrange_mult_.Init(num_of_nearest_pairs_);
  eq_lagrange_mult_.SetAll(1.0);
  double max_nearest_distance=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    max_nearest_distance=std::max(nearest_distances_[i], max_nearest_distance);
  }
  sum_of_furthest_distances_=-max_nearest_distance*
      data.n_cols()*data.n_cols();
 
  NOTIFY("Lower bound for optimization %lg", sum_of_furthest_distances_);
  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  fx_format_result(module_, "lower_optimal_bound", "%lg", sum_of_furthest_distances_);
}

void MaxVariance::Init(fx_module *module) {
  module_=module;
  new_dimension_ = fx_param_int_req(module_, "new_dimension");

  std::string nearest_neighbor_file=fx_param_str(module, 
      "nearest_neighbor_file", "nearest.txt");
  std::string furthest_neighbor_file=fx_param_str(module, 
      "furthest_neighbor_file", "futherst.txt");
  FILE *fp=fopen(nearest_neighbor_file.c_str(), "r");
  if (fp==NULL) {
    FATAL("Error while opening %s...%s", nearest_neighbor_file.c_str(),
        strerror(errno));
  }
  nearest_neighbor_pairs_.Init();
  nearest_distances_.Init();
  num_of_points_=0;
  while(!feof(fp)) {
    index_t n1, n2;
    double distance;
    fscanf(fp,"%i %i %lg", &n1, &n2, &distance);
    nearest_neighbor_pairs_.PushBackCopy(std::make_pair(n1, n2));
    nearest_distances_.PushBackCopy(distance);
    if (n1>num_of_points_) {
      num_of_points_=n1;
    }
    if (n2>num_of_points_) {
      num_of_points_=n2;
    }
  }
  num_of_points_++;
  fclose(fp);   
  num_of_nearest_pairs_=nearest_neighbor_pairs_.size(); 
  eq_lagrange_mult_.Init(num_of_nearest_pairs_);
  eq_lagrange_mult_.SetAll(1.0);
  double max_nearest_distance=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    max_nearest_distance=std::max(nearest_distances_[i], max_nearest_distance);
  }
  sum_of_furthest_distances_=-max_nearest_distance*
      num_of_points_*num_of_points_;
  NOTIFY("Lower bound for optimization %lg", sum_of_furthest_distances_);
  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  fx_format_result(module_, "lower_optimal_bound", "%lg", sum_of_furthest_distances_);
}


void MaxVariance::Destruct() {
  allknn_.Destruct();
  nearest_neighbor_pairs_.Renew();
  nearest_distances_.Renew();
  eq_lagrange_mult_.Destruct();

}

void MaxVariance::ComputeGradient(Matrix &coordinates, Matrix *gradient) {
  gradient->CopyValues(coordinates);
  // we need to use -CRR^T because we want to maximize CRR^T
  la::Scale(-1.0, gradient);
  index_t dimension=coordinates.n_rows();
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    double a_i_r[dimension];
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(dimension, point1, point2) 
                           -nearest_distances_[i];
    la::SubOverwrite(dimension, point2, point1, a_i_r);

   // equality constraints
   la::AddExpert(dimension,
       -eq_lagrange_mult_[i]+dist_diff*sigma_,
        a_i_r, 
        gradient->GetColumnPtr(n1));
   la::AddExpert(dimension,
        eq_lagrange_mult_[i]-dist_diff*sigma_,
        a_i_r, 
        gradient->GetColumnPtr(n2));
  }
}

void MaxVariance::ComputeObjective(Matrix &coordinates, double *objective) {
  *objective=0;
  index_t dimension = coordinates.n_rows();
  for(index_t i=0; i< coordinates.n_cols(); i++) {
     *objective-=la::Dot(dimension, 
                        coordinates.GetColumnPtr(i),
                        coordinates.GetColumnPtr(i));
  }
}

void MaxVariance::ComputeFeasibilityError(Matrix &coordinates, double *error) {
  index_t dimension=coordinates.n_rows();
  *error=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    *error += math::Sqr(la::DistanceSqEuclidean(dimension, 
                                               point1, point2) 
                                          -nearest_distances_[i]);
  }
}

double MaxVariance::ComputeLagrangian(Matrix &coordinates) {
  index_t dimension=coordinates.n_rows();
  double lagrangian=0;
  ComputeObjective(coordinates, &lagrangian);
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(dimension, point1, point2) 
                           -nearest_distances_[i];
    lagrangian+=dist_diff*dist_diff*sigma_
        -eq_lagrange_mult_[i]*dist_diff;
  }
  return lagrangian;
}

void MaxVariance::UpdateLagrangeMult(Matrix &coordinates) {
  index_t dimension=coordinates.n_rows();
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff =la::DistanceSqEuclidean(dimension, point1, point2) 
                            -nearest_distances_[i];
    eq_lagrange_mult_[i]-=sigma_*dist_diff;
  }
}

void MaxVariance::set_sigma(double sigma) {
  sigma_=sigma;
}

bool MaxVariance::IsDiverging(double objective) {
  if (objective < sum_of_furthest_distances_) {
    NOTIFY("objective(%lg) < sum_of_furthest_distances (%lg)", objective,
        sum_of_furthest_distances_);
    return true;
  } else {
    return false;
  }
}

void MaxVariance::Project(Matrix *coordinates) {
  OptUtils::RemoveMean(coordinates);
}

index_t  MaxVariance::num_of_points() {
  return num_of_points_;
}

void MaxVariance::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(new_dimension_, num_of_points_);
  for(index_t i=0; i<num_of_points_; i++) {
    for(index_t j=0; j<new_dimension_ ; j++) {
      init_data->set(j, i,math::Random(0, 1) );
    }
  }
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
void MaxFurthestNeighbors::Init(datanode *module, Matrix &data) {
  module_=module;
  new_dimension_ = fx_param_int_req(module_, "new_dimension");
  num_of_points_=data.n_cols();
  infeasibility1_=DBL_MAX;
  previous_infeasibility1_=DBL_MAX;
  desired_feasibility_error_ = fx_param_double(module_, "desired_feasibility_error", 1);
  grad_tolerance_ = fx_param_double(module_, "grad_tolerance", 0.1);
  infeasibility_tolerance_=  fx_param_double(module_, "infeasibility_tolerance", 0.01);
  knns_ = fx_param_int(module_, "knns", 5);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  NOTIFY("Data loaded ...\n");
  NOTIFY("Nearest neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  if (knns_==0) {
     allknn_.Init(data, leaf_size_, MAX_KNNS); 
  } else {
    allknn_.Init(data, leaf_size_, knns_); 
  }
  NOTIFY("Tree built ...\n");
  NOTIFY("Computing neighborhoods ...\n");
  ArrayList<index_t> from_tree_neighbors;
  ArrayList<double>  from_tree_distances;
  allknn_.ComputeNeighbors(&from_tree_neighbors,
                           &from_tree_distances);

  NOTIFY("Neighborhoods computed...\n");
  if (knns_==0) {
    NOTIFY("Auto-tuning the knn...\n" );
    MaxVarianceUtils::EstimateKnns(from_tree_neighbors,
        from_tree_distances,
        MAX_KNNS, 
        data.n_cols(),
        data.n_rows(),
        &knns_); 
    NOTIFY("Optimum knns is %i", knns_);
    fx_format_result(module_, "optimum_knns", "%i",knns_);
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
        from_tree_distances,
        MAX_KNNS,
        knns_,
        &nearest_neighbor_pairs_,
        &nearest_distances_,
        &num_of_nearest_pairs_);
  } else { 
    NOTIFY("Consolidating neighbors...\n");
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
        from_tree_distances,
        knns_,
        knns_,
        &nearest_neighbor_pairs_,
        &nearest_distances_,
        &num_of_nearest_pairs_);
  }
  sum_of_nearest_distances_=math::Pow<1,2>(la::Dot(nearest_distances_.size(), 
      &nearest_distances_[0], &nearest_distances_[0]));
  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  eq_lagrange_mult_.Init(num_of_nearest_pairs_);
  eq_lagrange_mult_.SetAll(1.0);
  NOTIFY("Furtherst neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  allkfn_.Init(data, leaf_size_, 1); 
  NOTIFY("Tree built ...\n");
  NOTIFY("Computing furthest neighborhoods ...\n");
  from_tree_neighbors.Renew();
  from_tree_distances.Renew();
  allkfn_.ComputeNeighbors(&from_tree_neighbors,
                           &from_tree_distances);
  NOTIFY("Furthest Neighbors computed...\n");
  NOTIFY("Consolidating neighbors...\n");
  MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
      from_tree_distances,
      1,
      1,
      &furthest_neighbor_pairs_,
      &furthest_distances_,
      &num_of_furthest_pairs_);
  double max_nearest_distance=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    max_nearest_distance=std::max(nearest_distances_[i], max_nearest_distance);
  }
  sum_of_furthest_distances_=-max_nearest_distance*
      data.n_cols()*num_of_furthest_pairs_;
 
  NOTIFY("Lower bound for optimization %lg", sum_of_furthest_distances_);
  fx_format_result(module_, "lower_optimal_bound", "%lg", sum_of_furthest_distances_);
}

void MaxFurthestNeighbors::Init(fx_module *module) {
  module_=module;
  new_dimension_ = fx_param_int_req(module_, "new_dimension");
  infeasibility1_=DBL_MAX;
  previous_infeasibility1_=DBL_MAX;
  desired_feasibility_error_ = fx_param_double(module_, "desired_feasibility_error", 1);
  grad_tolerance_ = fx_param_double(module_, "grad_tolerance", 0.1);
  infeasibility_tolerance_=  fx_param_double(module_, "infeasibility_tolerance", 0.01);
  std::string nearest_neighbor_file=fx_param_str_req(module, 
      "nearest_neighbor_file");
  std::string furthest_neighbor_file=fx_param_str_req(module, 
      "furthest_neighbor_file");
  FILE *fp=fopen(nearest_neighbor_file.c_str(), "r");
  if (fp==NULL) {
    FATAL("Error while opening %s...%s", nearest_neighbor_file.c_str(),
        strerror(errno));
  }
  nearest_neighbor_pairs_.Init();
  nearest_distances_.Init();
  num_of_points_=0;
  while(!feof(fp)) {
    index_t n1, n2;
    double distance;
    fscanf(fp,"%i %i %lg", &n1, &n2, &distance);
    nearest_neighbor_pairs_.PushBackCopy(std::make_pair(n1, n2));
    nearest_distances_.PushBackCopy(distance);
    if (n1>num_of_points_) {
      num_of_points_=n1;
    }
    if (n2>num_of_points_) {
      num_of_points_=n2;
    }
  }
  num_of_points_++;
  num_of_nearest_pairs_=nearest_neighbor_pairs_.size();
  sum_of_nearest_distances_=math::Pow<1,2>(la::Dot(nearest_distances_.size(), 
      &nearest_distances_[0], &nearest_distances_[0]));

  fclose(fp);
  fp=fopen(furthest_neighbor_file.c_str(), "r");
  if (fp==NULL) {
    FATAL("Error while opening %s...%s", furthest_neighbor_file.c_str(),
        strerror(errno));
  }
  furthest_neighbor_pairs_.Init();
  furthest_distances_.Init();
  while(!feof(fp)) {
    index_t n1, n2;
    double distance;
    fscanf(fp,"%i %i %lg", &n1, &n2, &distance);
    furthest_neighbor_pairs_.PushBackCopy(std::make_pair(n1, n2));
    furthest_distances_.PushBackCopy(distance);
  }
  fclose(fp);
  num_of_furthest_pairs_=furthest_neighbor_pairs_.size();
  eq_lagrange_mult_.Init(num_of_nearest_pairs_);
  eq_lagrange_mult_.SetAll(1.0);
  double max_nearest_distance=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    max_nearest_distance=std::max(nearest_distances_[i], max_nearest_distance);
  }
  sum_of_furthest_distances_=-max_nearest_distance*
      num_of_points_*num_of_points_;
 
  NOTIFY("Lower bound for optimization %lg", sum_of_furthest_distances_);
  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  fx_format_result(module_, "lower_optimal_bound", "%lg", sum_of_furthest_distances_);
}

void MaxFurthestNeighbors::Destruct() {
  allknn_.Destruct();
  allkfn_.Destruct();
  nearest_neighbor_pairs_.Renew();
  nearest_distances_.Renew();
  eq_lagrange_mult_.Destruct();
  furthest_neighbor_pairs_.Renew();
  furthest_distances_.Renew();
}

void MaxFurthestNeighbors::ComputeGradient(Matrix &coordinates, Matrix *gradient) {
  index_t dimension=coordinates.n_rows();
  gradient->SetAll(0.0);
  // objective
  for(index_t i=0; i<num_of_furthest_pairs_; i++) {
    double a_i_r[dimension];
    index_t n1=furthest_neighbor_pairs_[i].first;
    index_t n2=furthest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    la::SubOverwrite(dimension, point2, point1, a_i_r);

    la::AddExpert(dimension, -1.0, a_i_r,
        gradient->GetColumnPtr(n1));
    la::AddTo(dimension, a_i_r,
        gradient->GetColumnPtr(n2));
  }
  // equality constraints
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    double a_i_r[dimension];
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(dimension, point1, point2) 
                           -nearest_distances_[i];
    la::SubOverwrite(dimension, point2, point1, a_i_r);

    la::AddExpert(dimension,
        -eq_lagrange_mult_[i]+dist_diff*sigma_,
        a_i_r, 
        gradient->GetColumnPtr(n1));
    la::AddExpert(dimension,
        eq_lagrange_mult_[i]-dist_diff*sigma_,
        a_i_r, 
        gradient->GetColumnPtr(n2));
  }
}

void MaxFurthestNeighbors::ComputeObjective(Matrix &coordinates, 
    double *objective) {
  *objective=0;
  index_t dimension = coordinates.n_rows();
  for(index_t i=0; i<num_of_furthest_pairs_; i++) {
    index_t n1=furthest_neighbor_pairs_[i].first;
    index_t n2=furthest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double diff = la::DistanceSqEuclidean(dimension, point1, point2);
    *objective-=diff;   
  }
}

void MaxFurthestNeighbors::ComputeFeasibilityError(Matrix &coordinates, double *error) {
  index_t dimension=coordinates.n_rows();
  *error=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(dimension, 
                                               point1, point2) 
                           -nearest_distances_[i];
    *error+=dist_diff*dist_diff;
  }
  *error= 100 * math::Pow<1,2>(*error)/sum_of_nearest_distances_;
}

double MaxFurthestNeighbors::ComputeLagrangian(Matrix &coordinates) {
  index_t dimension=coordinates.n_rows();
  double lagrangian=0;
  ComputeObjective(coordinates, &lagrangian);
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(dimension, point1, point2) 
                           -nearest_distances_[i];
    lagrangian+=dist_diff*dist_diff*sigma_/2
        -eq_lagrange_mult_[i]*dist_diff;
  }
  return lagrangian;
}

void MaxFurthestNeighbors::UpdateLagrangeMult(Matrix &coordinates) {
  index_t dimension=coordinates.n_rows();
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff =la::DistanceSqEuclidean(dimension, point1, point2) 
                            -nearest_distances_[i];
    eq_lagrange_mult_[i]-=sigma_*dist_diff;
  }
}

void MaxFurthestNeighbors::set_sigma(double sigma) {
  sigma_=sigma;
}

void MaxFurthestNeighbors::set_lagrange_mult(double val) {
  eq_lagrange_mult_.SetAll(val);
}
bool MaxFurthestNeighbors::IsDiverging(double objective) {
  if (objective < sum_of_furthest_distances_) {
    NOTIFY("objective(%lg) < sum_of_furthest_distances (%lg)", objective,
        sum_of_furthest_distances_);
    return true;
  } else {
    return false;
  }
}

void MaxFurthestNeighbors::Project(Matrix *coordinates) {
  OptUtils::RemoveMean(coordinates);
}

index_t MaxFurthestNeighbors::num_of_points() {
  return num_of_points_;
}

void MaxFurthestNeighbors::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(new_dimension_, num_of_points_);
  for(index_t i=0; i<num_of_points_; i++) {
    for(index_t j=0; j<new_dimension_ ; j++) {
      init_data->set(j, i,math::Random(0, 1) );
    }
  }
}
bool MaxFurthestNeighbors::IsOptimizationOver(Matrix &coordinates, 
      Matrix &gradient, double step) { 
  ComputeFeasibilityError(coordinates, &infeasibility1_);
  if (infeasibility1_<desired_feasibility_error_ || 
      fabs(infeasibility1_-previous_infeasibility1_)<infeasibility_tolerance_)  {
    NOTIFY("Optimization is over");
    return true;
  } else {
    previous_infeasibility1_=infeasibility1_;
    return false; 
  }

}

bool MaxFurthestNeighbors::IsIntermediateStepOver(Matrix &coordinates, 
      Matrix &gradient, double step) {
   double norm_gradient=math::Pow<1,2>(la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr()));
  double feasibility_error;
  ComputeFeasibilityError(coordinates, &feasibility_error);
  if (norm_gradient*step < grad_tolerance_ 
      ||  feasibility_error<desired_feasibility_error_) {
    return true;
  }
  return false;


} 


///////////////////////////////////////////////////////////////
void MaxVarianceUtils::ConsolidateNeighbors(ArrayList<index_t> &from_tree_ind,
   ArrayList<double>  &from_tree_dist,
    index_t num_of_neighbors,
    index_t chosen_neighbors,
    ArrayList<std::pair<index_t, index_t> > *neighbor_pairs,
    ArrayList<double> *distances,
    index_t *num_of_pairs) {
  
  *num_of_pairs=0;
  index_t num_of_points=from_tree_ind.size()/num_of_neighbors;
  neighbor_pairs->Init();
  distances->Init();
  bool skip=false;
  for(index_t i=0; i<num_of_points; i++) {
    for(index_t k=0; k<chosen_neighbors; k++) {  
      index_t n1=i;                         //neighbor 1
      index_t n2=from_tree_ind[i*num_of_neighbors+k];  //neighbor 2
      if (n1 > n2) {
        for(index_t n=0; n<chosen_neighbors; n++) {
          if (from_tree_ind[n2*num_of_neighbors+n] == n1) {
            skip=true;
            break;
          }
        }
      }  
      if (skip==false) {
        *num_of_pairs+=1;
        neighbor_pairs->PushBackCopy(std::make_pair(n1, n2));
        distances->PushBackCopy(from_tree_dist[i*num_of_neighbors+k]);
      }
      skip=false;
    }
  }
}

void MaxVarianceUtils::EstimateKnns(ArrayList<index_t> &neares_neighbors,
                                    ArrayList<double> &nearest_distances,
                                    index_t maximum_knns, 
                                    index_t num_of_points,
                                    index_t dimension,
                                    index_t *optimum_knns) {
  double max_loocv_score=-DBL_MAX;
  double loocv_score=0;
  //double unit_sphere_volume=math::SphereVolume(1.0, dimension);
  *optimum_knns=0;
  for(index_t k=2; k<maximum_knns; k++) {
    loocv_score=0.0;
    double mean_band=0.0;
    for(index_t i=0; i<num_of_points; i++){
      double scale_factor=pow(nearest_distances[i*maximum_knns+k], dimension/2);
      double probability=0;
      for(index_t j=0; j<k; j++) {
        probability+=exp(-nearest_distances[i*maximum_knns+j]
            /(2*math::Pow<1,2>(nearest_distances[i*maximum_knns+k])))/scale_factor;
      }
      loocv_score+=log(probability);
      mean_band+=nearest_distances[i*maximum_knns+k];
    }
    NOTIFY("Knn=%i mean_band=%lg score=%lg, dimension=%i", k, mean_band/num_of_points, loocv_score, dimension);
    if (loocv_score > max_loocv_score) {
      max_loocv_score=loocv_score;
      *optimum_knns=k;
    }
  }
}

