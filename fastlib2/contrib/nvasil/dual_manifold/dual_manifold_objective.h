/*
 * =====================================================================================
 * 
 *       Filename:  dual_manifold_objective.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  03/18/2008 07:45:50 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef DUAL_MANIFOLD_OBJECTIVES_H_
#define DUAL_MANIFOLD_OBJECTIVES_H_

#include "fastlib/fastlib.h"
#include "../l_bfgs/optimization_utils.h"

class DualMaxVariance {
 public:
  void Init(datanode *module, Matrix *other_part, 
      ArrayList<std::pair<index_t, index_t> > *pairs_to_consider,
      ArrayList<double> *dot_prod_values);
  void Destruct();
  void ComputeGradient(Matrix &coordinates, Matrix *gradient);
  void ComputeObjective(Matrix &coordinates, double *objective);
  void ComputeFeasibilityError(Matrix &coordinates, double *error);
  double ComputeLagrangian(Matrix &coordinates);
  void UpdateLagrangeMult(Matrix &coordinates);
  void Project(Matrix *coordinates);
  void set_sigma(double sigma); 
  
 private:
  datanode *module_;
  ArrayList<std::pair<index_t, index_t> > pairs_to_consider_;
  ArrayList<double> *dot_prod_values_;
  Matrix *other_part_;
  Vector eq_lagrange_mult_;
  double sigma_;
};

#include "dual_manifold_objective_impl.h"
#endif // DUAL_MANIFOLD_OBJECTIVES_H_

