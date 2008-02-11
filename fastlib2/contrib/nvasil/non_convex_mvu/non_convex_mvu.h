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
  void ComputeLocalOptimum();
  // eta < 1
  void set_eta(double eta);
  // gamma > 1
  void set_gamma(double gamma);
  void set_step_size(double step_size);
  void set_max_iterations(index_t max_iterations);
  void set_new_dimension(index_t new_dimension);
  void set_tolerance(double tolerance); 
  /**
   *  sigma for armijo rule somewhere between 1e-5 to 1e-1
   */
  void set_armijo_sigma(double armijo_sigma);
  /**
   *  beta for armijo rule somewhere between 0.5 to 0.1
   */
  void set_armijo_beta(double armijo_beta);
  Matrix &coordinates();

 private:
  AllkNN allknn_;
  index_t leaf_size_;
  index_t num_of_points_;
  index_t dimension_;
  index_t knns_;
  ArrayList<double> distances_;
  ArrayList<index_t> neighbors_;
  // Lagrange multipliers for distance constraints
  Vector lagrange_mult_; 
  // Lagrange multiplier for the centering constraint
  // We want the final coordinates to have zero mean
  Vector centering_lagrange_mult_;
  double sigma_;
  double eta_;
  double gamma_;
  double previous_feasibility_error_;
  double step_size_;
  double tolerance_;
  double armijo_sigma_;
  double armijo_beta_;
  index_t max_iterations_;
  index_t new_dimension_;
  Matrix coordinates_;
  Matrix gradient_;
  Matrix data_;

  void UpdateLagrangeMult_();
  void LocalSearch_(double *step);
  double ComputeLagrangian_(Matrix &coordinates);
  void ComputeFeasibilityError_(double *distance_constraint, 
                                double *centering_constraint);
  double ComputeFeasibilityError_();
  void ComputeGradient_();
  double ComputeObjective_(Matrix &coord);
};

#include "non_convex_mvu_impl.h"
#endif //NON_CONVEX_MVU_H__
