/*
 * =====================================================================================
 * 
 *       Filename:  mvu_classification.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  10/07/2008 02:24:09 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef MVU_CLASSIFICATION_H_
#define MVU_CLASSIFICATION_H_

#include "fastlib/fastlib.h"
#include "mlpack/allknn/allknn.h"
#include "contrib/nvasil/allkfn/allkfn.h"
#include "mvu_objectives.h"
#include "../l_bfgs/optimization_utils.h"

class MaxFurthestNeighborsSemiSupervised {
public:
  static const index_t MAX_KNNS=30;
  /**
   * Initialize with this one if you want to load points from matrices
   * and then compute neighbors
   */
  void Init(fx_module *module, Matrix &labaled_data, 
      Matrix &unlabeled_data);
  /**
   * Use this one if you want to load the nearest neighbors and 
   * the furthest neighbors from a file
   */
  void Init(fx_module *module);
  void Destruct();
  void ComputeGradient(Matrix &coordinates, Matrix *gradient);
  void ComputeObjective(Matrix &coordinates, double *objective);
  void ComputeFeasibilityError(Matrix &coordinates, double *error);
  double ComputeLagrangian(Matrix &coordinates);
  void UpdateLagrangeMult(Matrix &coordinates);
  void Project(Matrix *coordinates);
  void set_sigma(double sigma); 
  void set_lagrange_mult(double val);
  bool IsDiverging(double objective); 
  bool IsOptimizationOver(Matrix &coordinates, 
      Matrix &gradient, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, 
      Matrix &gradient, double step);
  void GiveInitMatrix(Matrix *init_matrix);
 index_t num_of_points();
 
private:
  datanode *module_;
  AllkNN allknn_;
  AllkFN allkfn_;
  index_t knns_;
  index_t leaf_size_;
  index_t num_of_labeled_;
  index_t num_of_unlabeled_;
  index_t labeled_offset_;
  index_t unlabeled_offset_;
  ArrayList<std::pair<index_t, index_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  Vector eq_lagrange_mult_;
  index_t num_of_nearest_pairs_;
  index_t num_of_furthest_pairs_;
  ArrayList<std::pair<index_t, index_t> > furthest_neighbor_pairs_;
  ArrayList<double> furthest_distances_;
  double sum_of_furthest_distances_;
  double sigma_;
  index_t num_of_points_;
  index_t new_dimension_;
  double grad_tolerance_;
  double desired_feasibility_error_;
  double infeasibility1_;
  double previous_infeasibility1_;
  double sum_of_nearest_distances_;
};

class MaxFurthestNeighborsSvmSemiSupervised {
public:
  static const index_t MAX_KNNS=30;
  static const double MARGIN=0.01;
  /**
   * Initialize with this one if you want to load points from matrices
   * and then compute neighbors
   */
  void Init(fx_module *module, Matrix &labeled_data, 
      Matrix &unlabeled_data, Matrix &labels);
  /**
   * Use this one if you want to load the nearest neighbors and 
   * the furthest neighbors from a file
   */
  void Init(fx_module *module);
  void Init(fx_module *module, Matrix &labels);
  void Destruct();
  void ComputeGradient(Matrix &coordinates, Matrix *gradient);
  void ComputeObjective(Matrix &coordinates, double *objective);
  void ComputeFeasibilityError(Matrix &coordinates, double *error);
  double ComputeLagrangian(Matrix &coordinates);
  void UpdateLagrangeMult(Matrix &coordinates);
  void Project(Matrix *coordinates);
  void set_sigma(double sigma); 
  void set_lagrange_mult(double val);
  bool IsDiverging(double objective); 
  bool IsOptimizationOver(Matrix &coordinates, 
      Matrix &gradient, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, 
      Matrix &gradient, double step);
  void GiveInitMatrix(Matrix *init_matrix);
 index_t num_of_points();
 void anchors(ArrayList<index_t> *anch);
 
private:
  datanode *module_;
  AllkNN allknn_;
  AllkFN allkfn_;
  index_t knns_;
  index_t leaf_size_;
  index_t num_of_labeled_;
  index_t num_of_unlabeled_;
  index_t labeled_offset_;
  index_t unlabeled_offset_;
  Matrix svm_signs_;
  ArrayList<std::pair<index_t, index_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  Vector eq_lagrange_mult_;
  Vector ineq_lagrange_mult_;
  index_t num_of_nearest_pairs_;
  index_t num_of_furthest_pairs_;
  ArrayList<std::pair<index_t, index_t> > furthest_neighbor_pairs_;
  ArrayList<double> furthest_distances_;
  double sum_of_furthest_distances_;
  double sigma_;
  double sigma1_;
  double regularizer_;
  index_t num_of_points_;
  index_t num_of_classes_;
  index_t new_dimension_;
  double grad_tolerance_;
  double desired_feasibility_error_;
  double infeasibility1_;
  double previous_infeasibility1_;
  double sum_of_nearest_distances_;
  ArrayList<index_t> anchors_;
  
};

// This version picks a point that doesn't belong to the dataset
class MaxFurthestNeighborsSvmSemiSupervised1 {
public:
  static const index_t MAX_KNNS=30;
  static const double MARGIN=0.01;
  /**
   * Initialize with this one if you want to load points from matrices
   * and then compute neighbors
   */
  void Init(fx_module *module, Matrix &labeled_data, 
      Matrix &unlabeled_data, Matrix &labels);
  /**
   * Use this one if you want to load the nearest neighbors and 
   * the furthest neighbors from a file
   */
  void Init(fx_module *module);
  void Init(fx_module *module, Matrix &labels);
  void Destruct();
  void ComputeGradient(Matrix &coordinates, Matrix *gradient);
  void ComputeObjective(Matrix &coordinates, double *objective);
  void ComputeFeasibilityError(Matrix &coordinates, double *error);
  double ComputeLagrangian(Matrix &coordinates);
  void UpdateLagrangeMult(Matrix &coordinates);
  void Project(Matrix *coordinates);
  void set_sigma(double sigma); 
  void set_lagrange_mult(double val);
  bool IsDiverging(double objective); 
  bool IsOptimizationOver(Matrix &coordinates, 
      Matrix &gradient, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, 
      Matrix &gradient, double step);
  void GiveInitMatrix(Matrix *init_matrix);
 index_t num_of_points();
 void anchors(ArrayList<index_t> *anch);
 
private:
  datanode *module_;
  AllkNN allknn_;
  AllkFN allkfn_;
  index_t knns_;
  index_t leaf_size_;
  index_t num_of_labeled_;
  index_t num_of_unlabeled_;
  index_t labeled_offset_;
  index_t unlabeled_offset_;
  Matrix svm_signs_;
  ArrayList<std::pair<index_t, index_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  Vector eq_lagrange_mult_;
  Vector ineq_lagrange_mult_;
  index_t num_of_nearest_pairs_;
  index_t num_of_furthest_pairs_;
  ArrayList<std::pair<index_t, index_t> > furthest_neighbor_pairs_;
  ArrayList<double> furthest_distances_;
  double sum_of_furthest_distances_;
  double sigma_;
  double sigma1_;
  double regularizer_;
  index_t num_of_points_;
  index_t num_of_classes_;
  index_t new_dimension_;
  double grad_tolerance_;
  double desired_feasibility_error_;
  double infeasibility1_;
  double previous_infeasibility1_;
  double sum_of_nearest_distances_;
  ArrayList<index_t> anchors_;
  
};

#include "mvu_classification_impl.h"
#endif
