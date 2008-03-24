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
  void Init(datanode *module, Matrix &data);
  void ComputeGradient(Matrix &coordinates, Matrix *gradient);
  void ComputeObjective(Matrix &coordinates, double *objective);
  void ComputeFeasibilityError(Matrix &coordinates, double *error);
  double ComputeLagrangian(Matrix &coordinates);
  void UpdateLagrangeMult(Matrix &coordinates);
  void Project(Matrix *coordinates);
  void set_sigma(double sigma); 
  bool IsDiverging(double objective); 
 
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

  void ConsolidateNeighbors_(ArrayList<index_t> &from_tree_ind,
      ArrayList<double>  &from_tree_dist,
      index_t num_of_neighbors,
      ArrayList<std::pair<index_t, index_t> > *neighbor_pairs,
      ArrayList<double> *distances,
      index_t *num_of_pairs);
};

class MaxVarianceInequalityOnFurthest {
 public:
  void Init(datanode *module, Matrix &data);
  void ComputeGradient(Matrix &coordinates, Matrix *gradient);
  void ComputeObjective(Matrix &coordinates, double *objective);
  void ComputeFeasibilityError(Matrix &coordinates, double *error);
  double ComputeLagrangian(Matrix &coordinates);
  void UpdateLagrangeMult(Matrix &coordinates);
  void Project(Matrix *coordinates);
  void set_sigma(double sigma);
  bool IsDiverging(double objective); 
  
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

  void ConsolidateNeighbors_(ArrayList<index_t> &from_tree_ind,
      ArrayList<double>  &from_tree_dist,
      index_t num_of_neighbors,
      ArrayList<std::pair<index_t, index_t> > *neighbor_pairs,
      ArrayList<double> *distances,
      index_t *num_of_pairs);
};

class MaxFurthestNeighbors {
public:
  void Init(datanode *module, Matrix &data);
  void ComputeGradient(Matrix &coordinates, Matrix *gradient);
  void ComputeObjective(Matrix &coordinates, double *objective);
  void ComputeFeasibilityError(Matrix &coordinates, double *error);
  double ComputeLagrangian(Matrix &coordinates);
  void UpdateLagrangeMult(Matrix &coordinates);
  void Project(Matrix *coordinates);
  void set_sigma(double sigma); 
  bool IsDiverging(double objective); 
 
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

  void ConsolidateNeighbors_(ArrayList<index_t> &from_tree_ind,
      ArrayList<double>  &from_tree_dist,
      index_t num_of_neighbors,
      ArrayList<std::pair<index_t, index_t> > *neighbor_pairs,
      ArrayList<double> *distances,
      index_t *num_of_pairs);

};
#include "mvu_objectives_impl.h"
#endif //MVU_OBJECTIVES_H_
