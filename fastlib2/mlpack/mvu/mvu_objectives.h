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
#include "mlpack/allkfn/allkfn.h"
#include "fastlib/optimization/lbfgs/optimization_utils.h"

const fx_entry_doc mvu_entries[] = {
  {"new_dimension", FX_REQUIRED, FX_INT, NULL,
   " the number fo dimensions for the unfolded"},
  {"nearest_neighbor_file", FX_PARAM, FX_STR, NULL,
   " file with the nearest neighbor pairs and the squared distances \n"
   " defaults to nearest.txt"},
  {"furthest_neighbor_file", FX_PARAM, FX_STR, NULL,
   " file with the nearest neighbor pairs and the squared distances "},
  {"knns", FX_PARAM, FX_INT, NULL,
   " number of nearest neighbors to build the graph\n"
   " if you choose the option with the nearest file you don't need to specify it"},
 {"leaf_size", FX_PARAM, FX_INT, NULL,
   " leaf_size for the tree.\n "
   " if you choose the option with the nearest file you don't need to specify it"},
FX_ENTRY_DOC_DONE
};

const fx_module_doc mvu_doc = {
  mvu_entries, NULL,
  " This program computes the Maximum Variance Unfolding"
  " and the Maximum Futhest Neighbor Unfolding as presented "
  " in the paper: \n"
  " @conference{vasiloglou2008ssm,\n"
  "   title={{Scalable semidefinite manifold learning}},\n"
  "   author={Vasiloglou, N. and Gray, A.G. and Anderson, D.V.},\n"
  "   booktitle={Machine Learning for Signal Processing, 2008. MLSP 2008. IEEE Workshop on},\n"
  "   pages={368--373},\n"
  "   year={2008}\n"
  " }\n"
};

class MaxVariance {
 public:
  static const index_t MAX_KNNS=30;
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
      Matrix &gradient, double step) { return true; }
  void GiveInitMatrix(Matrix *init_data);
 index_t num_of_points();

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
  index_t num_of_points_;
  index_t new_dimension_;
};

class MaxFurthestNeighbors {
public:
  static const index_t MAX_KNNS=30;
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
  index_t num_of_points();
  void GiveInitMatrix(Matrix *init_data);

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
  index_t num_of_points_;
  index_t new_dimension_;
  double infeasibility1_;
  double previous_infeasibility1_;
  double desired_feasibility_error_;
  double infeasibility_tolerance_;
  double sum_of_nearest_distances_;
  double grad_tolerance_;
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
