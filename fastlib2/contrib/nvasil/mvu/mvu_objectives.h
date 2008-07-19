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
  static const index_t MAX_KNNS=30;
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
      Matrix &gradient, double step) { return false;}
  bool IsIntermediateStepOver(Matrix &coordinates, 
      Matrix &gradient, double step) {return false;}
 
 private:
  datanode *module_;
  AllkNN allknn_;
  index_t knns_;
  index_t leaf_size_;
  ArrayList<std::pair<index_t, index_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  Vector eq_lagrange_mult_;
  index_t num_of_nearest_pairs_;
  double sigma_;
  double sum_of_furthest_distances_;
};

class MaxVarianceInequalityOnFurthest {
 public:
  static const index_t MAX_KNNS=30;
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
 
 private:
  datanode *module_;
  AllkNN allknn_;
  AllkFN allkfn_;
  index_t knns_;
  index_t leaf_size_;
  ArrayList<std::pair<index_t, index_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  Vector eq_lagrange_mult_;
  Vector ineq_lagrange_mult_;
  index_t num_of_nearest_pairs_;
  index_t num_of_furthest_pairs_;
  ArrayList<std::pair<index_t, index_t> > furthest_neighbor_pairs_;
  ArrayList<double> furthest_distances_;
  double sigma_;
  double sum_of_furthest_distances_;
};

class MaxFurthestNeighbors {
public:
  static const index_t MAX_KNNS=30;
  void Init(datanode *module, Matrix &data);
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
      Matrix &gradient, double step) {return false;}
  bool IsIntermediateStepOver(Matrix &coordinates, 
      Matrix &gradient, double step) {return false;}
 
private:
  datanode *module_;
  AllkNN allknn_;
  AllkFN allkfn_;
  index_t knns_;
  index_t leaf_size_;
  ArrayList<std::pair<index_t, index_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  Vector eq_lagrange_mult_;
  index_t num_of_nearest_pairs_;
  index_t num_of_furthest_pairs_;
  ArrayList<std::pair<index_t, index_t> > furthest_neighbor_pairs_;
  ArrayList<double> furthest_distances_;
  double sum_of_furthest_distances_;
  double sigma_;
};

class MaxVarianceUtils {
 public:
  static void ConsolidateNeighbors(ArrayList<index_t> &from_tree_ind,
      ArrayList<double>  &from_tree_dist,
      index_t num_of_neighbors,
      index_t chosen_neighbors,
      ArrayList<std::pair<index_t, index_t> > *neighbor_pairs,
      ArrayList<double> *distances,
      index_t *num_of_pairs);
  static void EstimateKnns(ArrayList<index_t> &neares_neighbors,
                                       ArrayList<double> &nearest_distances,
                                       index_t maximum_knns, 
                                       index_t num_of_points,
                                       index_t dimension,
                                       index_t *optimum_knns); 
};

#include "mvu_objectives_impl.h"
#endif //MVU_OBJECTIVES_H_
