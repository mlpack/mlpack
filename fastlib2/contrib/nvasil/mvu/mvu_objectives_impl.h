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

void MaxVariance::Init(datanode *module, Matrix &data) {
  module_=module;
  knns_ = fx_param_int(module_, "knns", 5);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  NOTIFY("Data loaded ...\n");
  NOTIFY("Nearest neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  allknn_.Init(data, leaf_size_, knns_); 
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
      knns_,
      &nearest_neighbor_pairs_,
      &nearest_distances_,
      &num_of_nearest_pairs_);
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

void MaxVariance::ConsolidateNeighbors_(ArrayList<index_t> &from_tree_ind,
    ArrayList<double>  &from_tree_dist,
    index_t num_of_neighbors,
    ArrayList<std::pair<index_t, index_t> > *neighbor_pairs,
    ArrayList<double> *distances,
    index_t *num_of_pairs) {
  
  *num_of_pairs=0;
  index_t num_of_points=from_tree_ind.size()/num_of_neighbors;
  neighbor_pairs->Init();
  distances->Init();
  bool skip=false;
  for(index_t i=0; i<num_of_points; i++) {
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

void MaxVariance::Project(Matrix *coordinates) {
  OptUtils::RemoveMean(coordinates);
}
////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
void MaxVarianceInequalityOnFurthest::Init(datanode *module, Matrix &data) {
  module_=module;
  knns_ = fx_param_int(module_, "knns", 5);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  NOTIFY("Data loaded ...\n");
  NOTIFY("Nearest neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  allknn_.Init(data, leaf_size_, knns_); 
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
      knns_,
      &nearest_neighbor_pairs_,
      &nearest_distances_,
      &num_of_nearest_pairs_);
  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  eq_lagrange_mult_.Init(num_of_nearest_pairs_);
  eq_lagrange_mult_.SetAll(1.0);
  NOTIFY("Furtherst neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  allkfn_.Init(data, leaf_size_, 1); 
  NOTIFY("Tree built ...\n");
  NOTIFY("Computing furthest neighborhoods ...\n");
  from_tree_neighbors.Destruct();
  from_tree_distances.Destruct();
  allkfn_.ComputeNeighbors(&from_tree_neighbors,
                           &from_tree_distances);
  NOTIFY("Furthest Neighbors computed...\n");
  NOTIFY("Consolidating neighbors...\n");
  ConsolidateNeighbors_(from_tree_neighbors,
      from_tree_distances,
      1,
      &furthest_neighbor_pairs_,
      &furthest_distances_,
      &num_of_furthest_pairs_);
  ineq_lagrange_mult_.Init(num_of_furthest_pairs_);
  ineq_lagrange_mult_.SetAll(1.0);
  double max_nearest_distance=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    max_nearest_distance=std::max(nearest_distances_[i], max_nearest_distance);
  }
  sum_of_furthest_distances_=-max_nearest_distance*
      data.n_cols()*data.n_cols()*data.n_cols();
 
  NOTIFY("Lower bound for optimization %lg", sum_of_furthest_distances_);
  fx_format_result(module_, "lower_optimal_bound", "%lg", sum_of_furthest_distances_);
}

void MaxVarianceInequalityOnFurthest::ComputeGradient(Matrix &coordinates, 
    Matrix *gradient) {
  gradient->CopyValues(coordinates);
    // we need to use -CRR^T because we want to maximize CRR^T
    la::Scale(-1.0, gradient);
  index_t dimension=coordinates.n_rows();

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

  // inequality constraints
  for(index_t i=0; i<num_of_furthest_pairs_; i++) {
    double a_i_r[dimension];
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(dimension, point1, point2) 
                           -nearest_distances_[i];
    if (sigma_*dist_diff <= ineq_lagrange_mult_[i]) {
      la::SubOverwrite(dimension, point2, point1, a_i_r);
      la::AddExpert(dimension,
          -ineq_lagrange_mult_[i]+dist_diff*sigma_,
           a_i_r, 
           gradient->GetColumnPtr(n1));
      la::AddExpert(dimension,
          ineq_lagrange_mult_[i]-dist_diff*sigma_,
          a_i_r, 
          gradient->GetColumnPtr(n2));
    } 
  }
}

void MaxVarianceInequalityOnFurthest::ComputeObjective(Matrix &coordinates, 
    double *objective) {
  *objective=0;
  index_t dimension = coordinates.n_rows();
  for(index_t i=0; i< coordinates.n_cols(); i++) {
     *objective-=la::Dot(dimension, 
                        coordinates.GetColumnPtr(i),
                        coordinates.GetColumnPtr(i));
  }
}

void MaxVarianceInequalityOnFurthest::ComputeFeasibilityError(Matrix &coordinates, 
    double *error) {
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
  for(index_t i=0; i<num_of_furthest_pairs_; i++) {
    index_t n1=furthest_neighbor_pairs_[i].first;
    index_t n2=furthest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff=math::Sqr(la::DistanceSqEuclidean(dimension, 
                         point1, point2) 
                         -furthest_distances_[i]);
    if (dist_diff<=0) {
      *error+=dist_diff*dist_diff;
    }
  }
}

double MaxVarianceInequalityOnFurthest::ComputeLagrangian(Matrix &coordinates) {
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
  for(index_t i=0; i<num_of_furthest_pairs_; i++) {
    index_t n1=furthest_neighbor_pairs_[i].first;
    index_t n2=furthest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff=math::Sqr(la::DistanceSqEuclidean(dimension, 
                         point1, point2) 
                         -furthest_distances_[i]);
    if (dist_diff*sigma_<=ineq_lagrange_mult_[i]) {
      lagrangian+=(-ineq_lagrange_mult_[i]+sigma_/2*dist_diff)*dist_diff;
    } else {
      lagrangian-=math::Sqr(ineq_lagrange_mult_[i])/(2*sigma_);
    }
  }
  return lagrangian;
}


void MaxVarianceInequalityOnFurthest::UpdateLagrangeMult(Matrix &coordinates) {
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
  for(index_t i=0; i<num_of_furthest_pairs_; i++) {
    index_t n1=nearest_neighbor_pairs_[i].first;
    index_t n2=nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff =la::DistanceSqEuclidean(dimension, point1, point2) 
                            -nearest_distances_[i];
    ineq_lagrange_mult_[i]=std::max(
        ineq_lagrange_mult_[i]-sigma_*dist_diff, 0.0);
  }
}

void MaxVarianceInequalityOnFurthest::set_sigma(double sigma) {
  sigma_=sigma;
}

bool MaxVarianceInequalityOnFurthest::IsDiverging(double objective) {
  if (objective < sum_of_furthest_distances_) {
    NOTIFY("objective(%lg) < sum_of_furthest_distances (%lg)", objective,
        sum_of_furthest_distances_);
    return true;
  } else {
    return false;
  }
}

void MaxVarianceInequalityOnFurthest::ConsolidateNeighbors_(ArrayList<index_t> &from_tree_ind,
   ArrayList<double>  &from_tree_dist,
    index_t num_of_neighbors,
    ArrayList<std::pair<index_t, index_t> > *neighbor_pairs,
    ArrayList<double> *distances,
    index_t *num_of_pairs) {
  
  *num_of_pairs=0;
  index_t num_of_points=from_tree_ind.size()/num_of_neighbors;
  neighbor_pairs->Init();
  distances->Init();
  bool skip=false;
  for(index_t i=0; i<num_of_points; i++) {
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
 
void MaxVarianceInequalityOnFurthest::Project(Matrix *coordinates) {
  OptUtils::RemoveMean(coordinates);
}

////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
void MaxFurthestNeighbors::Init(datanode *module, Matrix &data) {
  module_=module;
  knns_ = fx_param_int(module_, "knns", 5);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  NOTIFY("Data loaded ...\n");
  NOTIFY("Nearest neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  allknn_.Init(data, leaf_size_, knns_); 
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
      knns_,
      &nearest_neighbor_pairs_,
      &nearest_distances_,
      &num_of_nearest_pairs_);

  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  eq_lagrange_mult_.Init(num_of_nearest_pairs_);
  eq_lagrange_mult_.SetAll(1.0);
  NOTIFY("Furtherst neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  allkfn_.Init(data, leaf_size_, 1); 
  NOTIFY("Tree built ...\n");
  NOTIFY("Computing furthest neighborhoods ...\n");
  from_tree_neighbors.Destruct();
  from_tree_distances.Destruct();
  allkfn_.ComputeNeighbors(&from_tree_neighbors,
                           &from_tree_distances);
  NOTIFY("Furthest Neighbors computed...\n");
  NOTIFY("Consolidating neighbors...\n");
  ConsolidateNeighbors_(from_tree_neighbors,
      from_tree_distances,
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

bool MaxFurthestNeighbors::IsDiverging(double objective) {
  if (objective < sum_of_furthest_distances_) {
    NOTIFY("objective(%lg) < sum_of_furthest_distances (%lg)", objective,
        sum_of_furthest_distances_);
    return true;
  } else {
    return false;
  }
}

void MaxFurthestNeighbors::ConsolidateNeighbors_(ArrayList<index_t> &from_tree_ind,
   ArrayList<double>  &from_tree_dist,
    index_t num_of_neighbors,
    ArrayList<std::pair<index_t, index_t> > *neighbor_pairs,
    ArrayList<double> *distances,
    index_t *num_of_pairs) {
  
  *num_of_pairs=0;
  index_t num_of_points=from_tree_ind.size()/num_of_neighbors;
  neighbor_pairs->Init();
  distances->Init();
  bool skip=false;
  for(index_t i=0; i<num_of_points; i++) {
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

void MaxFurthestNeighbors::Project(Matrix *coordinates) {
  OptUtils::RemoveMean(coordinates);
}

