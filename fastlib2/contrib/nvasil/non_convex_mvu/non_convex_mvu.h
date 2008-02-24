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

#define OPT_PARAM_ objective_mode, constraints_mode, gradient_mode
#define TEMPLATE_TAG_ template<ObjectiveEnum objective_mode, \
                               ConstraintsEnum constraints_mode, \
                               GradientEnum gradient_mode>
enum GradientEnum {
  DeterministicGrad=0, 
  StochasticGrad=1
};
enum ObjectiveEnum {
  Feasibility=0,
  MaxVariance=1,
  MinVariance=2,
  MaxFurthestNeighbors=3
};
enum ConstraintsEnum {
  EqualityOnNearest=0,
  InequalityOnNearest=1,
  InequalityOnFurthest=2
};
 
class NonConvexMVUTest;
class NonConvexMVU {
 public:
  friend class NonConvexMVUTest;
  NonConvexMVU::NonConvexMVU();
  template<GradientEnum gradient_mode>
  void Init(std::string data_file);
  template<GradientEnum gradient_mode>
  void Init(Matrix &data);
   
  /**
   * Computes a local optimum for a given rank
   * PROBLEM_TYPE
  */
  TEMPLATE_TAG_
  void ComputeLocalOptimumBFGS();
  TEMPLATE_TAG_
  void ComputeLocalOptimumSGD();
  void set_knns(index_t knns);
  void set_kfns(index_t kfns);
  void set_leaf_size(index_t leaf_size);
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
  // the nearest neighbor pairs for the constraints
  ArrayList<std::pair<index_t, index_t> > nearest_neighbor_pairs_;
  ArrayList<index_t> nearest_neighbors_;
  ArrayList<double> nearest_distances_;
  index_t num_of_nearest_pairs_;
  ArrayList<std::pair<index_t, index_t> > furthest_neighbor_pairs_;
  ArrayList<index_t> furthest_neighbors_;
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
  TEMPLATE_TAG_
  void InitOptimization_(); 
  TEMPLATE_TAG_
  void UpdateLagrangeMult_();
  TEMPLATE_TAG_
  void LocalSearch_(double *step, Matrix &direction);
  TEMPLATE_TAG_
  void ComputeBFGS_(double *step, Matrix &grad, index_t memory);
  void InitBFGS();
  void UpdateBFGS_();
  void UpdateBFGS_(index_t index_bfgs);
  TEMPLATE_TAG_
  double ComputeLagrangian_(Matrix &coordinates);
  TEMPLATE_TAG_
  void ComputeFeasibilityError_(double *distance_constraint, 
                                double *centering_constraint);
  TEMPLATE_TAG_
  double ComputeFeasibilityError_();
  TEMPLATE_TAG_
  void ComputeGradient_(Matrix &coord, Matrix *grad);
  TEMPLATE_TAG_
  void ComputePairGradient_(index_t p1, index_t chosen_neighbor, 
    Matrix &coord, Vector *gradient1, Vector *gradient2);
 
  TEMPLATE_TAG_
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
#undef TEMPLATE_TAG_
#endif //NON_CONVEX_MVU_H__
