/*
 * =====================================================================================
 * 
 *       Filename:  sdp_objectives.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  05/23/2008 12:55:44 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
#ifndef SDP_OBJECTIVES_H_
#define SDP_OBJECTIVES_H_
#include "fastlib/fastlib.h"
#include "../l_bfgs/optimization_utils.h"

class SmallSdpNmf {
 public:
  void Init(fx_module *module, 
			ArrayList<size_t> &rows,
			ArrayList<size_t> &columns,
      ArrayList<double>  &values);
  void Destruct();
  void ComputeGradient(Matrix &coordinates, Matrix *gradient);
  void ComputeObjective(Matrix &coordinates, double *objective);
  void ComputeFeasibilityError(Matrix &coordinates, double *error);
  double ComputeLagrangian(Matrix &coordinates);
  void UpdateLagrangeMult(Matrix &coordinates);
  void Project(Matrix *coordinates);
  void set_sigma(double sigma); 
  void GiveInitMatrix(Matrix *init_data);
	bool IsDiverging(double objective); 
  bool IsOptimizationOver(Matrix &coordinates, Matrix &gradient, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);
  
 private:
  fx_module *module_;
  double sigma_;
	size_t num_of_columns_;
	size_t num_of_rows_;
	size_t new_dim_;
	size_t offset_h_;
  size_t offset_tw_;
  size_t offset_th_;
  size_t offset_v_; 
	ArrayList<size_t> rows_;
	ArrayList<size_t> columns_;
  ArrayList<double>  values_;
  Matrix objective_factor_;
  size_t number_of_cones_;
  double desired_duality_gap_;
  double gradient_tolerance_;
  double v_accuracy_;
};

#include "sdp_objectives_impl.h"
#endif
