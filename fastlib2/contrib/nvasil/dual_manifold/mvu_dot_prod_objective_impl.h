/*
 * =====================================================================================
 * 
 *       Filename:  mvu_dot_prod_objective_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  04/09/2008 06:24:01 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

void MVUDotProdObjective::Init(datanode *module,
    Matrix *coordinates, 
    ArrayList<std::pair<index_t, index_t> > &pairs_to_consider, 
    // The values of the (row, column) values, also known as the dot products
    ArrayList<double> &dot_prod_values) {
  
  module_=module;
  auxiliary_mat_=coordinates;
  pairs_to_consider_.InitCopy(pairs_to_consider);
  dot_prod_values_.InitCopy(dot_prod_values);
  eq_lagrange_mult_.Init(dot_prod_values.size());
  eq_lagrange_mult_.SetAll(0.0);
  num_of_constraints_=dot_prod_values_.size();
}

void MVUDotProdObjective::ComputeGradient(Matrix &coordinates, Matrix *gradient) {
  gradient->CopyValues(coordinates);
  // we need to use -CRR^T because we want to maximize CRR^T
  la::Scale(-1.0, gradient);
  index_t dimension=auxiliary_mat_->n_rows();
  Vector constant;
  for (index_t i=0; i<num_of_constraints_; i++) {
    index_t ind1=pairs_to_consider_[i].first; 
    index_t ind2=pairs_to_consider_[i].second;
    double *p1=auxiliary_mat_->GetColumnPtr(ind1);
    double *p2=coordinates.GetColumnPtr(ind2);
    double dot_prod =la::Dot(dimension, p1, p2);
    double diff=dot_prod-dot_prod_values_[i];
    la::AddExpert(dimension,  
        -eq_lagrange_mult_[i]+sigma_*diff,
        p1,
        gradient->GetColumnPtr(ind1));
  } 
}

void MVUDotProdObjective::ComputeObjective(Matrix &coordinates, double *objective) {
  *objective=0;
  index_t dimension = coordinates.n_rows();
  for(index_t i=0; i< coordinates.n_cols(); i++) {
     *objective-=la::Dot(dimension, 
                        coordinates.GetColumnPtr(i),
                        coordinates.GetColumnPtr(i));
  } 
}

void MVUDotProdObjective::ComputeFeasibilityError(Matrix &coordinates, double *error) {
  index_t dimension = coordinates.n_rows();
  *error=0;
  for(index_t i=0; i<num_of_constraints_; i++) {
    index_t ind1=pairs_to_consider_[i].first; 
    index_t ind2=pairs_to_consider_[i].second;
    double *p1=auxiliary_mat_->GetColumnPtr(ind1);
    double *p2=coordinates.GetColumnPtr(ind2);
    double dot_prod =la::Dot(dimension, p1, p2);
    double diff=dot_prod-dot_prod_values_[i];
    *error +=diff*diff;
  }
}

double MVUDotProdObjective::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian=0;
  index_t dimension = coordinates.n_rows();
  ComputeObjective(coordinates, &lagrangian);
  for(index_t i=0; i<num_of_constraints_; i++) {
    index_t ind1=pairs_to_consider_[i].first; 
    index_t ind2=pairs_to_consider_[i].second;
    double *p1=auxiliary_mat_->GetColumnPtr(ind1);
    double *p2=coordinates.GetColumnPtr(ind2);
    double dot_prod =la::Dot(dimension, p1, p2);
    double diff=dot_prod-dot_prod_values_[i];
    lagrangian+= -eq_lagrange_mult_[i]*diff + sigma_*diff*diff;
  }
  return lagrangian;
}

void MVUDotProdObjective::UpdateLagrangeMult(Matrix &coordinates) {
  index_t dimension=coordinates.n_rows();
  for(index_t i=0; i<num_of_constraints_; i++) {
    index_t ind1=pairs_to_consider_[i].first; 
    index_t ind2=pairs_to_consider_[i].second;
    double *p1=auxiliary_mat_->GetColumnPtr(ind1);
    double *p2=coordinates.GetColumnPtr(ind2);
    double dot_prod =la::Dot(dimension, p1, p2);
    double diff=dot_prod-dot_prod_values_[i];
    eq_lagrange_mult_[i]-=sigma_*diff;
  }
}

void MVUDotProdObjective::Project(Matrix *coordinates) {
  OptUtils::RemoveMean(coordinates);
}

void MVUDotProdObjective::set_sigma(double sigma) {
   sigma_=sigma;
} 

bool MVUDotProdObjective::IsDiverging(double objective) {
  return false;
}

