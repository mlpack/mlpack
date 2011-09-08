/*
 * =====================================================================================
 * 
 *       Filename:  geometric_nmf.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  07/02/2008 02:37:01 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef GEOMETRIC_NMF_H_
#define GEOMETRIC_NMF_H_
#include "fastlib/fastlib.h"
#include "../l_bfgs/optimization_utils.h"
#include "mlpack/allknn/allknn.h"
#include "../l_bfgs/optimization_utils.h"
#include "../mvu/mvu_objectives.h"

class GeometricNmf {
 public:
  GeometricNmf();
  void Init(fx_module *module, 
			ArrayList<size_t> &rows,
			ArrayList<size_t> &columns,
      ArrayList<double>  &values);
  void Init(fx_module *module, 
			ArrayList<size_t> &rows,
			ArrayList<size_t> &columns,
      ArrayList<double>  &values,
      Matrix &lower_bound,
      Matrix &upper_bound);
  void Destruct();
  void ComputeGradient(Matrix &coordinates, Matrix *gradient);
  void ComputeObjective(Matrix &coordinates, double *objective);
  void ComputeFeasibilityError(Matrix &coordinates, double *error);
  double ComputeLagrangian(Matrix &coordinates);
  void UpdateLagrangeMult(Matrix &coordinates);
  void Project(Matrix *coordinates);
  void set_sigma(double sigma); 
  // returns false if the problem is infeasible
  bool GiveInitMatrix(Matrix *init_data);
	bool IsDiverging(double objective); 
  bool IsOptimizationOver(Matrix &coordinates, Matrix &gradient, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);
  
 private:
  fx_module *module_;
  double sigma_;
  //AllkNN allknn_;
  size_t leaf_size_;
 // ArrayList<double> nearest_dot_products_;
 // ArrayList<double> nearest_distances_;
 //  ArrayList<std::pair<size_t, size_t> >  nearest_neighbor_pairs_;
 // size_t num_of_nearest_pairs_;
  size_t knns_;
	size_t num_of_columns_;
	size_t num_of_rows_;
	size_t new_dim_;
	size_t offset_h_;
  size_t num_of_logs_;
  // epsilon are the auxiliary variables that bring the data
  // in the feasible domain. We add it in the inequalities
  size_t offset_epsilon_; 
	ArrayList<size_t> rows_;
	ArrayList<size_t> columns_;
  ArrayList<double>  values_;
  Matrix *lower_bound_;
  Matrix *upper_bound_;
  size_t number_of_constraints_;
  double desired_duality_gap_;
  double gradient_tolerance_;
  double v_accuracy_;
};

#include "geometric_nmf_impl.h"
#endif
