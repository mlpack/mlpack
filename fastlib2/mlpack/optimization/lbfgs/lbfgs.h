/*
 * =====================================================================================
 * 
 *       Filename:  lbfgs.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  03/04/2008 10:09:29 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef LBFGS_H_
#define LBFGS_H_

#include "fastlib/fastlib.h"
#include <string>

/**
 * @author Nikolaos Vasiloglou (nvasil@ieee.org)
 * @file lbfgs.h
 * 
 * This class implements the L-BFGS method as desribed in:
 *
 * @book{nocedal1999no,
 *       title={{Numerical Optimization}},
 *       author={Nocedal, J. and Wright, S.J.},
 *       year={1999},
 *       publisher={Springer}
 * }
 * */

const fx_entry_doc lbfgs_entries[] = {
  {"num_of_points", FX_PARAM, FX_INT, NULL,
   "  The number of points for the optimization variable.\n"},
  {"sigma", FX_PARAM, FX_DOUBLE, NULL,
   "  The initial penalty parameter on the augmented lagrangian.\n"},
  {"objective_factor", FX_PARAM, FX_DOUBLE, NULL,
   "  obsolete.\n"},
  {"eta", FX_PARAM, FX_DOUBLE, NULL,
   "  wolfe parameter.\n"},
  {"gamma", FX_PARAM, FX_DOUBLE, NULL,
   "  sigma increase rate, after inner loop is done sigma is multiplied by gamma.\n"},
  {"new_dimension", FX_PARAM, FX_INT, NULL,
   "  The dimension of the points\n"},
  {"desired_feasibility", FX_PARAM, FX_DOUBLE, NULL,
   "  Since this is used with augmented lagrangian, we need to know "
     "when the  feasibility is sufficient.\n"},
  {"feasibility_tolerance", FX_PARAM, FX_DOUBLE, NULL,
   "  if the feasibility is not improved by that quantity, then it stops.\n"},
  {"wolfe_sigma1", FX_PARAM, FX_DOUBLE, NULL,
   "  wolfe parameter.\n"},
  {"wolfe_sigma2", FX_PARAM, FX_DOUBLE, NULL,
   "  wolfe parameter.\n"},
  {"min_beta", FX_PARAM, FX_DOUBLE, NULL,
   "  wolfe parameter.\n"},
  {"wolfe_beta", FX_PARAM, FX_DOUBLE, NULL,
   "  wolfe parameter.\n"},
  {"step_size", FX_PARAM, FX_DOUBLE, NULL,
   "  Initial step size for the wolfe search.\n"},
  {"silent", FX_PARAM, FX_BOOL, NULL,
   "  if true then it doesn't emmit updates.\n"},
  {"show_warnings", FX_PARAM, FX_BOOL, NULL,
   "  if true then it does show warnings.\n"},
  {"use_default_termination", FX_PARAM, FX_BOOL, NULL,
   "  let this module decide where to terminate. If false then"
   " the objective function decides .\n"},
  {"norm_grad_tolerance", FX_PARAM, FX_DOUBLE, NULL,
   "  If the norm of the gradient doesn't change more than "
     "this quantity between two iterations and the use_default_termination "
     "is set, the algorithm terminates.\n"},
  {"max_iterations", FX_PARAM, FX_INT, NULL,
   "  maximum number of iterations required.\n"},
  {"mem_bfgs", FX_PARAM, FX_INT, NULL,
   "  the limited memory of BFGS.\n"},
  {"log_file", FX_PARAM, FX_STR, NULL,
   " file to log the output.\n"},
  {"iterations", FX_RESULT, FX_INT, NULL,
   "  iterations until convergence.\n"},
  {"feasibility_error", FX_RESULT, FX_DOUBLE, NULL,
   "  the fesibility error achived by termination.\n"},
  {"final_sigma", FX_RESULT, FX_DOUBLE, NULL,
   "  the last penalty parameter used\n"},
  {"objective", FX_RESULT, FX_DOUBLE, NULL,
   "  the objective achieved by termination.\n"},
  {"wolfe_step", FX_TIMER, FX_CUSTOM, NULL,
   "  Time spent computing the wolfe step.\n"},
  {"bfgs_step", FX_TIMER, FX_CUSTOM, NULL,
   "  Time spent computing the bfgs step.\n"},
  {"update_bfgs", FX_TIMER, FX_CUSTOM, NULL,
   "  Time spent computing the bfgs updating.\n"},

  FX_ENTRY_DOC_DONE
};

const fx_module_doc lbfgs_doc = {
  lbfgs_entries, NULL,
  "The LBFGS module for optimization.\n"
};


template<typename OptimizedFunction>
class Lbfgs {
 public:
  void Init(OptimizedFunction *optimized_function, datanode* module);
  void Destruct();
  void ComputeLocalOptimumBFGS();
  void ReportProgress();
  void ReportProgressFile(std::string file);
  void CopyCoordinates(Matrix *result);
  void Reset(); 
  void set_coordinates(Matrix &coordinates);
  void set_desired_feasibility(double desired_feasibility);
  void set_feasibility_tolerance(double feasibility_tolerance);
  void set_norm_grad_tolerance(double norm_grad_tolerance);
  void set_max_iterations(index_t max_iterations);
  Matrix *coordinates();
  double sigma();
  void set_sigma(double sigma);

 private:
  void InitOptimization_();
  void ComputeWolfeStep_();
  void UpdateLagrangeMult_();
  success_t ComputeWolfeStep_(double *step, Matrix &direction);
  success_t ComputeBFGS_(double *step, Matrix &grad, index_t memory);
  success_t UpdateBFGS_();
  success_t UpdateBFGS_(index_t index_bfgs);
  void BoundConstrain();
  std::string ComputeProgress_();
  void ReportProgressFile_();

  struct datanode* module_;
  OptimizedFunction  *optimized_function_;
  index_t num_of_iterations_;
  index_t num_of_points_;
  index_t new_dimension_;
  double sigma_;
  double objective_factor_;
  double eta_;
  double gamma_;
  double step_;
  double desired_feasibility_;
  double feasibility_tolerance_;
  double norm_grad_tolerance_;
  double wolfe_sigma1_;
  double wolfe_sigma2_;
  double wolfe_beta_;
  double min_beta_;
  bool silent_;
  bool show_warnings_;
  bool use_default_termination_;
  ArrayList<Matrix> s_bfgs_;
  ArrayList<Matrix> y_bfgs_;
  Vector ro_bfgs_;
  index_t index_bfgs_;
  Matrix coordinates_;
  Matrix previous_coordinates_;
  Matrix gradient_;
  Matrix previous_gradient_;
  index_t max_iterations_;
  double step_size_;
  // the memory of bfgs 
  index_t mem_bfgs_;
  FILE *fp_log_;
};

#include "lbfgs_impl.h"
#endif //LBFGS_H_
