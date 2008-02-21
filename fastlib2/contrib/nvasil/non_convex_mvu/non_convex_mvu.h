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
#include "contrib/nvasil/allkfn/allkfn.h"

class NonConvexMVUTest;
class NonConvexMVU {
 public:
  friend class NonConvexMVUTest;
  NonConvexMVU::NonConvexMVU();
  void Init(std::string data_file, index_t knns, index_t kfns);
  void Init(std::string data_file, index_t knns, index_t kfns, index_t leaf_size);
  /**
   * Computes a local optimum for a given rank
   * PROBLEM_TYPE
   * 0 = Feasibility problem
   * 1 = Equality Constraints for the distances
   * 2 = Inequality Constraints for the distances
   * 3 = Equality Constraint but the objective is to maximize furthest
   *     neighbor distances
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
  /**
   *  sigma1 and sigma2 for wolfe rule somewhere between 1e-5 to 1e-1
   *  and sigma1 < sigma2
   */
  void set_wolfe_sigma(double wolfe_sigma1, double wolfe_sigma2);
  /**
   *  beta for wolfe rule somewhere between 0.5 to 0.1
   */
  void set_wolfe_beta(double wolfe_beta);
  /**
   * Set the memory for the BFGS method
   */
  void set_mem_bfgs(index_t mem_bfgs);
  Matrix &coordinates();

 private:
  AllkNN allknn_;
  AllkFN allkfn_;
  index_t leaf_size_;
  index_t num_of_points_;
  index_t dimension_;
  index_t knns_;  //  k nearest neighbors
  index_t kfns_;   //  k furthest neighbors 
  ArrayList<double> nearest_distances_;
  // the nearest neighbor pairs for the constraints
  ArrayList<std::pair<index_t, index_t> > nearest_neighbor_pairs_;
  index_t num_of_nearest_pairs_;
  ArrayList<std::pair<index_t, index_t> > furthest_neighbor_pairs_;
  index_t num_of_furthest_pairs_;
  ArrayList<double> furthest_distances_;
  
  // Lagrange multipliers for distance constraints
  Vector lagrange_mult_; 
  double sigma_;
  double eta_;
  double gamma_;
  double previous_feasibility_error_;
  double step_size_;
  double distance_tolerance_;
  double wolfe_sigma1_;
  double wolfe_sigma2_;
  double wolfe_beta_;
  double max_violation_of_distance_constraint_;
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
  void LocalSearch_(double *step, Matrix &direction);
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
  void ComputeGradient_(Matrix &coord, Matrix *grad);
  template<int PROBLEM_TYPE>
  double ComputeObjective_(Matrix &coord);
  void Variance_(Matrix &coord, Vector *variance);
  void RemoveMean_(Matrix &mat);
  void ConsolidateNeighbors_(ArrayList<index_t> &from_tree_ind,
      ArrayList<double>  &from_tree_dist,
      index_t num_of_neighbors,
      ArrayList<std::pair<index_t, index_t> > *neighbor_pairs,
      ArrayList<double> *distances,
      index_t *num_of_pairs);
};

#include "non_convex_mvu_impl.h"
#endif //NON_CONVEX_MVU_H__
