/*
 * =====================================================================================
 * 
 *       Filename:  non_convex_mvu.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  02/03/2008 11:33:23 AM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef NON_CONVEX_MVU_H__
#define NON_CONVEX_MVU_H__

#include <string>
#include "fastlib/fastlib.h"
#include "mlpack/allknn/allknn.h"

class NonConvexMVUTest;
class NonConvexMVU {
 public:
  friend class NonConvexMVUTest;
  NonConvexMVU::NonConvexMVU();
  void Init(std::string data_file, index_t knns);
  void Init(std::string data_file, index_t knns, index_t leaf_size);
  /**
   * Computes a local optimum for a given rank
   * PROBLEM_TYPE
   * 0 = Feasibility problem
   * 1 = Equality Constraints for the distances
   * 2 = Inequality Constraints for the distances
   */
  template<int PROBLEM_TYPE>
  void ComputeLocalOptimumBFGS();
  // eta < 1
  void set_eta(double eta);
  // gamma > 1
  void set_gamma(double gamma);
  void set_sigma(double sigma);
  void set_step_size(double step_size);
  void set_max_iterations(index_t max_iterations);
  void set_new_dimension(index_t new_dimension);
  void set_distance_tolerance(double tolerance); 
  void set_gradient_tolerance(double tolerance);
  /**
   *  sigma for armijo rule somewhere between 1e-5 to 1e-1
   */
  void set_armijo_sigma(double armijo_sigma);
  /**
   *  beta for armijo rule somewhere between 0.5 to 0.1
   */
  void set_armijo_beta(double armijo_beta);
  /**
   * Set the memory for the BFGS method
   */
  void set_mem_bfgs(index_t mem_bfgs);
  Matrix &coordinates();

 private:
  AllkNN allknn_;
  index_t leaf_size_;
  index_t num_of_points_;
  index_t dimension_;
  index_t knns_;
  ArrayList<double> distances_;
  ArrayList<std::pair<index_t, index_t> > neighbor_pairs_;
  index_t num_of_pairs_;
  // Lagrange multipliers for distance constraints
  Vector lagrange_mult_; 
  double sigma_;
  double eta_;
  double gamma_;
  double previous_feasibility_error_;
  double step_size_;
  double gradient_tolerance_;
  double distance_tolerance_;
  double armijo_sigma_;
  double armijo_beta_;
  index_t max_iterations_;
  index_t new_dimension_;
  Matrix coordinates_;
  Matrix gradient_;
  Matrix data_;
  // These parameters are used for limited BFGS
 
  //ro_k = 1/(y^T * s)
  Vector ro_bfgs_; 
  // the memory of bfgs 
  index_t mem_bfgs_;
  // s_k = x_{k+1}-x_{k};
  ArrayList<Matrix> s_bfgs_;
  // y_k = g_{k+1} -g_k (g is the gradient)
  ArrayList<Matrix> y_bfgs_;
  // 
  index_t index_bfgs_;
  // previous gradient
  Matrix previous_gradient_;
  // previous coordinates
  Matrix previous_coordinates_;
  template<int PROBLEM_TYPE> 
  void InitOptimization_(); 
  template<int PROBLEM_TYPE>
  void UpdateLagrangeMult_();
  template<int PROPLEM_TYPE>
  void LocalSearch_(double *step, Matrix &grad);
  template<int PROPLEM_TYPE>
  void ComputeBFGS_(double *step, Matrix &grad, index_t memory);
  void InitBFGS();
  void UpdateBFGS_();
  void UpdateBFGS_(index_t index_bfgs);
  template<int PROBLEM_TYPE>
  double ComputeLagrangian_(Matrix &coordinates);
  template<int PROBLEM_TYPE>
  void ComputeFeasibilityError_(double *distance_constraint, 
                                double *centering_constraint);
  template<int PROBLEM_TYPE>
  double ComputeFeasibilityError_();
  template<int PROBLEM_TYPE>
  void ComputeGradient_();
  template<int PROBLEM_TYPE>
  double ComputeObjective_(Matrix &coord);
  void Variance_(Matrix &coord, Vector *variance);
  void RemoveMean_(Matrix &mat);
  void ConsolidateNeighbors_(ArrayList<index_t> &from_tree_ind,
      ArrayList<double>  &from_tree_dist,
      ArrayList<std::pair<index_t, index_t> > *neighbor_pairs,
      ArrayList<double> *distances,
      index_t *num_of_pairs);
};

#include "non_convex_mvu_impl.h"
#endif //NON_CONVEX_MVU_H__
