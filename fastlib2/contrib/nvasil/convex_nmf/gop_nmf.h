/*
 * =====================================================================================
 * 
 *       Filename:  gop_nmf.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  07/19/2008 12:05:37 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
#ifndef GOP_NMF_ENGINE_H_
#define GOP_NMF_ENGINE_H_
#include "fastlib/fastlib.h"
#include "../l_bfgs/l_bfgs.h"

class RelaxedNmf {
 public:
  void Init(ArrayList<index_t> &rows,
            ArrayList<index_t> &columns,
            ArrayList<double> &values,
            index_t new_dim,
            Matrix &x_lower_bound,
            Matrix &x_upper_bound
           );
  void Destruct();
  // The following are required by LBFGS
  void ComputeGradient(Matrix &coordinates, Matrix *gradient);
  void ComputeObjective(Matrix &coordinates, double *objective);
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
  index_t num_of_rows_;
  index_t num_of_cols_;
  index_t new_dim_;
  Vector x_lower_bound_;
  Vector x_upper_bound_;
  Vector A_linear_term_;
  Vector B_linear_term_;
  ArrayList<index_t> rows_;
  ArrayList<index_t> columns_;
  ArrayList<double> values_;
  Matrix x_lower_bound_;
  Matrix y_upper_bound_;
  double soft_lower_bound_;
};

class GopNmfEngine {
 public:
  struct SolutionPack {
   public:
     SolutionPack() {
     }
     ~SolutionPack() {
       
     }
     SolutionPack();
     Matrix solution_;
     std::pair<Matrix, Matrix> box_; 
  };
  
  typedef LBfgs<RelaxedNmf> Optimizer; 
  void Init(fx_module *module, Matrix &data_points);
  void ComputeGlobalOptimum();
    
 private:
  fx_module *module_;
  fx_module *l_bfgs_module_;
  Matrix x_upper_bound_;
  Matric x_lower_bound_;
  double desired_global_optimum_gap_;
  double grad_tolerance_;
  std::map<double, SolutionPack> lower_solution_;
  std::pair<double, Matrix> upper_solution_;
  ArrayList<double> rows_;
  ArrayList<double> columns_;
  ArrayList<double> values_;
  index_t num_of_rows_;
  index_t num_of_columns_;
  index_t prunes_;
  void Split(Matrix &lower_bound, Matrix &upper_bound, 
          Matrix *left_lower_bound, Matrix *left_upper_bound,
          Matrix *right_lower_bound, Matrix *right_upper_bound) 
  void PreprocessData(Matrix &data_mat);
};


#include "gop_nmf_impl.h"

#endif
