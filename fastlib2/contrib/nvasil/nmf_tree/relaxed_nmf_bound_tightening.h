/*
 * =====================================================================================
 * 
 *       Filename:  relaxed_nmf.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  08/16/2008 12:11:24 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
#ifndef RELAXED_NMF_BOUND_TIGHTENER_H_
#define RELAXED_NMF_BOUND_TIGHTENER_H_

class RelaxedNmfBoundTightener {
 public:
  void Init(fx_module *module,
            ArrayList<index_t> &rows,
            ArrayList<index_t> &columns,
            ArrayList<double> &values,
            Matrix *x_lower_bound, // the initial lower bound for x (optimization variable)
            Matrix *x_upper_bound  // the initial upper bound for x (optimization variable)
            index_t opt_var_row,
            index_t opt_var_column);
  void Destruct();
  // The following are required by LBFGS
  void ComputeGradient(Matrix &coordinates, Matrix *gradient);
  void ComputeObjective(Matrix &coordinates, double *objective);
  // This class implements a convex relaxation of the nmf objective
  // At some point we need to compute the original objective the non relaxed
  void ComputeNonRelaxedObjective(Matrix &coordinates, double *objective);
  void ComputeFeasibilityError(Matrix &coordinates, double *error);
  double ComputeLagrangian(Matrix &coordinates);
  void UpdateLagrangeMult(Matrix &coordinates);
  void Project(Matrix *coordinates);
  void set_sigma(double sigma); 
  void GiveInitMatrix(Matrix *init_data);
	bool IsDiverging(double objective); 
  bool IsOptimizationOver(Matrix &coordinates, Matrix &gradient, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);
  
  // The following are required by the branch and bound
  double GetSoftLowerBound();
    
 private:
  // holds all the info
  fx_module *module_;
  // number of rows of the original matrix
  index_t num_of_rows_;
  // number of columns of the original matrix
  index_t num_of_columns_;
  // offset of the H matrix on the coordinate variable
  index_t h_offset_;
  index_t w_offset_;
  double values_sq_norm_;
  index_t new_dimension_;
  // constant term for the LP relaxation part
  Vector a_linear_term_;
  // linear term for the LP relaxation part
  Vector b_linear_term_;
  ArrayList<index_t> rows_;
  ArrayList<index_t> columns_;
  ArrayList<double> values_;
  double function_upper_bound_;
  // lower bound for the optimization variable
  Matrix x_lower_bound_;
  // upper bound for the optimization variable
  Matrix x_upper_bound_;
  // soft lower bound of the relaxation
  double soft_lower_bound_;
  // tolerance for the gradient norm
  double grad_tolerance_;
  double desired_duality_gap_;
};

#include "relaxed_nmf_impl.h"
#endif
