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
#include <map>
#include "fastlib/fastlib.h"
#include "../l_bfgs/l_bfgs.h"
#include "geometric_nmf.h"

class RelaxedNmf {
 public:
  void Init(ArrayList<index_t> &rows,
            ArrayList<index_t> &columns,
            ArrayList<double> &values,
            index_t new_dim, // new dimension of the factorization
            double grad_tolerance, // if the norm gradient is less than the tolerance
                                   // then it terminates
            Matrix &x_lower_bound, // the initial lower bound for x (optimization variable)
            Matrix &x_upper_bound  // the initial upper bound for x (optimization variable)
           );
  void Init(fx_module *module,
            ArrayList<index_t> &rows,
            ArrayList<index_t> &columns,
            ArrayList<double> &values,
            Matrix &x_lower_bound, // the initial lower bound for x (optimization variable)
            Matrix &x_upper_bound  // the initial upper bound for x (optimization variable)
           );
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
  // number of rows of the original matrix
  index_t num_of_rows_;
  // number of columns of the original matrix
  index_t num_of_columns_;
  // offset of the H matrix on the coordinate variable
  index_t h_offset_;
  index_t w_offset_;
  double values_sq_norm_;
  index_t new_dim_;
  // constant term for the LP relaxation part
  Vector a_linear_term_;
  // linear term for the LP relaxation part
  Vector b_linear_term_;
  ArrayList<index_t> rows_;
  ArrayList<index_t> columns_;
  ArrayList<double> values_;
  // lower bound for the optimization variable
  Matrix x_lower_bound_;
  // upper bound for the optimization variable
  Matrix x_upper_bound_;
  // soft lower bound of the relaxation
  double soft_lower_bound_;
  // tolerance for the gradient norm
  double grad_tolerance_;
  double previous_objective_;

  inline double ComputeExpTaylorApproximation(double x, index_t order);
  index_t ComputeExpTaylorOrder(double error);
};

class RelaxedNmf1 {
 public:
  void Init(ArrayList<index_t> &rows,
            ArrayList<index_t> &columns,
            ArrayList<double> &values,
            index_t new_dim, // new dimension of the factorization
            double grad_tolerance, // if the norm gradient is less than the tolerance
                                   // then it terminates
            Matrix &x_lower_bound, // the initial lower bound for x (optimization variable)
            Matrix &x_upper_bound  // the initial upper bound for x (optimization variable)
           );
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
  // number of rows of the original matrix
  index_t num_of_rows_;
  // number of columns of the original matrix
  index_t num_of_columns_;
  // offset of the H matrix on the coordinate variable
  index_t h_offset_;
  index_t w_offset_;
  double values_sq_norm_;
  index_t new_dim_;
  // constant term for the LP relaxation part
  Vector a_linear_term_;
  // linear term for the LP relaxation part
  Vector b_linear_term_;
  ArrayList<index_t> rows_;
  ArrayList<index_t> columns_;
  ArrayList<double> values_;
  // lower bound for the optimization variable
  Matrix x_lower_bound_;
  // upper bound for the optimization variable
  Matrix x_upper_bound_;
  // soft lower bound of the relaxation
  double soft_lower_bound_;
  // tolerance for the gradient norm
  double grad_tolerance_;
  //  the penalty barrier
  double sigma_;
};


class GopNmfEngine {
 public:
  struct SolutionPack {
   public:
     SolutionPack() {
     }
     ~SolutionPack() {
       
     }
     Matrix solution_;
     std::pair<Matrix, Matrix> box_; 
  };
  
  typedef LBfgs<RelaxedNmf> LowerOptimizer; 
  typedef LBfgs<GeometricNmf> UpperOptimizer;
  void Init(fx_module *module, Matrix &data_points);
  void ComputeGlobalOptimum();
  void ComputeTighterGlobalOptimum();  

 private:
  fx_module *module_;
  fx_module *l_bfgs_module_;
  Matrix x_upper_bound_;
  Matrix x_lower_bound_;
  double desired_global_optimum_gap_;
  double grad_tolerance_;
  std::multimap<double, SolutionPack> lower_solution_;
  std::pair<double, Matrix> upper_solution_;
  ArrayList<index_t> rows_;
  ArrayList<index_t> columns_;
  ArrayList<double> values_;
  index_t num_of_rows_;
  index_t num_of_columns_;
  index_t new_dim_;
  index_t soft_prunes_;
  index_t hard_prunes_;
  double soft_pruned_volume_;
  double hard_pruned_volume_;
  double total_volume_;
  index_t iteration_;

  void Split(Matrix &lower_bound, Matrix &upper_bound, 
          Matrix *left_lower_bound, Matrix *left_upper_bound,
          Matrix *right_lower_bound, Matrix *right_upper_bound); 
  void PreprocessData(Matrix &data_mat);
  double ComputeVolume(Matrix &lower_bound, Matrix &upper_bound);
};


#include "gop_nmf_impl.h"

#endif
