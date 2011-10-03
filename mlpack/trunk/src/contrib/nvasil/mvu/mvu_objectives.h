/*
 * =====================================================================================
 * 
 *       Filename:  mvu_objectives.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  03/05/2008 12:00:56 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef MVU_OBJECTIVES_H_
#define MVU_OBJECTIVES_H_
#include "fastlib/fastlib.h"
#include "mlpack/allknn/allknn.h"
#include "contrib/nvasil/allkfn/allkfn.h"
#include "../l_bfgs/optimization_utils.h"

class MaxVariance {
 public:
  static const size_t MAX_KNNS=30;
  void Init(fx_module *module, Matrix &data);
  void Init(fx_module *module);
  void Destruct();
  void ComputeGradient(Matrix &coordinates, Matrix *gradient);
  void ComputeObjective(Matrix &coordinates, double *objective);
  void ComputeFeasibilityError(Matrix &coordinates, double *error);
  double ComputeLagrangian(Matrix &coordinates);
  void UpdateLagrangeMult(Matrix &coordinates);
  void Project(Matrix *coordinates);
  void set_sigma(double sigma); 
  bool IsDiverging(double objective); 
  bool IsOptimizationOver(Matrix &coordinates, 
      Matrix &gradient, double step) { return false;}
  bool IsIntermediateStepOver(Matrix &coordinates, 
      Matrix &gradient, double step) {return false;}
  void GiveInitMatrix(Matrix *init_data);
 size_t num_of_points();

 private:
  datanode *module_;
  AllkNN allknn_;
  size_t knns_;
  size_t leaf_size_;
  ArrayList<std::pair<size_t, size_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  Vector eq_lagrange_mult_;
  size_t num_of_nearest_pairs_;
  double sigma_;
  double sum_of_furthest_distances_;
  size_t num_of_points_;
  size_t new_dimension_;
};

class MaxVarianceInequalityOnFurthest {
 public:
  static const size_t MAX_KNNS=30;
  void Init(datanode *module, Matrix &data);
  void Destruct();
  void ComputeGradient(Matrix &coordinates, Matrix *gradient);
  void ComputeObjective(Matrix &coordinates, double *objective);
  void ComputeFeasibilityError(Matrix &coordinates, double *error);
  double ComputeLagrangian(Matrix &coordinates);
  void UpdateLagrangeMult(Matrix &coordinates);
  void Project(Matrix *coordinates);
  void set_sigma(double sigma);
  bool IsDiverging(double objective); 
  bool IsOptimizationOver(Matrix &coordinates, 
      Matrix &gradient, double step) {return false;}
  bool IsIntermediateStepOver(Matrix &coordinates, 
      Matrix &gradient, double step) {return false;}
  void GiveInitMatrix(Matrix *init_data);

 private:
  datanode *module_;
  AllkNN allknn_;
  AllkFN allkfn_;
  size_t knns_;
  size_t leaf_size_;
  ArrayList<std::pair<size_t, size_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  Vector eq_lagrange_mult_;
  Vector ineq_lagrange_mult_;
  size_t num_of_nearest_pairs_;
  size_t num_of_furthest_pairs_;
  ArrayList<std::pair<size_t, size_t> > furthest_neighbor_pairs_;
  ArrayList<double> furthest_distances_;
  double sigma_;
  double sum_of_furthest_distances_;
  size_t new_dimension_;
};

class MaxFurthestNeighbors {
public:
  static const size_t MAX_KNNS=30;
  void Init(fx_module *module, Matrix &data);
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
      Matrix &gradient, double step) ;
  bool IsIntermediateStepOver(Matrix &coordinates, 
      Matrix &gradient, double step); 
  size_t num_of_points();
  void GiveInitMatrix(Matrix *init_data);

private:
  datanode *module_;
  AllkNN allknn_;
  AllkFN allkfn_;
  size_t knns_;
  size_t leaf_size_;
  ArrayList<std::pair<size_t, size_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  Vector eq_lagrange_mult_;
  size_t num_of_nearest_pairs_;
  size_t num_of_furthest_pairs_;
  ArrayList<std::pair<size_t, size_t> > furthest_neighbor_pairs_;
  ArrayList<double> furthest_distances_;
  double sum_of_furthest_distances_;
  double sigma_;
  size_t num_of_points_;
  size_t new_dimension_;
  double infeasibility1_;
  double previous_infeasibility1_;
  double desired_feasibility_error_;
  double infeasibility_tolerance_;
  double sum_of_nearest_distances_;
  double grad_tolerance_;
};

class MaxVarianceUtils {
 public:
  static void ConsolidateNeighbors(ArrayList<size_t> &from_tree_ind,
      ArrayList<double>  &from_tree_dist,
      size_t num_of_neighbors,
      size_t chosen_neighbors,
      ArrayList<std::pair<size_t, size_t> > *neighbor_pairs,
      ArrayList<double> *distances,
      size_t *num_of_pairs);
  static void EstimateKnns(ArrayList<size_t> &neares_neighbors,
                                       ArrayList<double> &nearest_distances,
                                       size_t maximum_knns, 
                                       size_t num_of_points,
                                       size_t dimension,
                                       size_t *optimum_knns); 
};

#include "mvu_objectives_impl.h"
#endif //MVU_OBJECTIVES_H_
