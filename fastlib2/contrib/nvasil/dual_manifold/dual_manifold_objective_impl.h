/*
 * =====================================================================================
 * 
 *       Filename:  dual_manifold_objective_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  03/18/2008 08:09:51 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

void DualMaxVarianc::eInit(datanode *module, Matrix *other_part, 
      ArrayList<std::pair<index_t, index_t> > pairs_to_consider,
      ArrayList<double> *dot_prod_values) {
  other_part_=other_part;
  module_=module;
  pairs_to_consider_=pairs_to_consider;
  dot_prod_values_=dot_prod_values;
  eq_lagrange_mult_.Init(pairs_to_consider_.size());
  eq_lagrange_mult_.SetAll(1.0);
}
void DualMaxVariance::ComputeGradient(Matrix &coordinates, 
    Matrix *gradient) {
  gradient->CopyValues(coordinates);
  la::Scale(-2.0, &gradient);
  for(index_t i=0; i<pairs_to_concider_.size(); i++) {
    index_t n1=pairs_to_consider_[i].first;
    index_t n2=pairs_to_consider_[i].second;
    double *p1=coordinates.GetColumnPtr(n1);
    double *p2=other_part_->GetColumnPtr(n2);
    double diff=la::Dot(dimension, p1, p2)-(*dot_prod_values)[i];
    la::AddExpert(dimension, -eq_lagrange_mult_[i]+sigma_*diff,
        gradient->GetColumnPtr(n1));
  }
}
void DualMaxVariance::ComputeObjective(Matrix &coordinates, 
    double *objective) {
  index_t dimension=coordinates.n_rows();
  *objective=0;
  for(index_t i=0; i<coordinates.n_cols(); i++) {
    *objective-=la::Dot(dimension, coordinates.GetColumnPtr(i),
        coordinates.GetColumnPtr(i));
  }
}

void DualMaxVariance::ComputeFeasibilityError(Matrix &coordinates, 
    double *error) {
  DEBUG_ASSERT(coordinates.n_rows()==other_part->n_rows());
  *error=0;
  index_t dimension=coordinates.n_rows();
  for(index_t i=0; i<pairs_to_concider_.size(); i++) {
    index_t n1=pairs_to_consider_[i].first;
    index_t n2=pairs_to_consider_[i].second;
    double *p1=coordinates.GetColumnPtr(n1);
    double *p2=other_part_->GetColumnPtr(n2);
    *error+=math::Sqr(la::Dot(dimension, p1, p2)-(*dot_prod_values)[i]);
  }
}
double DualMaxVariance::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian=0;
  ComputeObjective(coordinates, &lagrangian);
  for(index_t i=0; i<pairs_to_concider_.size(); i++) {
    index_t n1=pairs_to_consider_[i].first;
    index_t n2=pairs_to_consider_[i].second;
    double *p1=coordinates.GetColumnPtr(n1);
    double *p2=other_part_->GetColumnPtr(n2);
    double diff=la::Dot(dimension, p1, p2)-(*dot_prod_values)[i];
    *error+=(-eq_lagrange_mult_[i]+sigma_/2*diff)*diff;
  }
}

void DualMaxVariance::UpdateLagrangeMult(Matrix &coordinates) {
  index_t dimension=coordinates.n_rows();
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1=pairs_to_consider_[i].first;
    index_t n2=pairs_to_consider_[i].second;
    double *p1=coordinates.GetColumnPtr(n1);
    double *p2=other_part_->GetColumnPtr(n2);
    double diff=la::Dot(dimension, p1, p2)-(*dot_prod_values)[i];
   eq_lagrange_mult_[i]-=sigma_*diff;
  }
}

void DualMaxVariance::Project(Matrix *coordinates) {
  OptUtils::RemoveMean(coordinates);
}

void DualMaxVariance::set_sigma(double sigma) {
  sigma_=sigma;
} 
 
