#ifndef MBP_KERHEL_H
#define MBP_KERHEL_H

#include "fastlib/fastlib.h"

class MultibodyPotentialKernel {

 private:

  virtual double PositiveEvaluateCommon_(const Matrix &squared_distances) = 0;

  virtual double NegativeEvaluateCommon_(const Matrix &squared_distances) = 0;

 public:

  Matrix min_squared_distances;

  Matrix max_squared_distances;

  virtual ~MultibodyPotentialKernel() {
  }

  virtual void Init(double bandwidth_in = -1.0) = 0;

  template<typename Global, typename Tree>
  bool ComputePairwiseDistances(const Global &globals, 
				const ArrayList<Tree *> &nodes) {
    
    for(index_t first_index = 0; first_index < nodes.size(); first_index++) {
      for(index_t second_index = first_index + 1; second_index < nodes.size(); 
	  second_index++) {
	
	double min_squared_distance =
	  nodes[first_index]->bound().MinDistanceSq
	  (nodes[second_index]->bound());

	double max_squared_distance =
	  nodes[first_index]->bound().MaxDistanceSq
	  (nodes[second_index]->bound());

	// Set the distances, mirroring them across diagonals, just in
	// case they are needed.
	min_squared_distances.set(first_index, second_index, 
				  min_squared_distance);
	min_squared_distances.set(second_index, first_index,
				  min_squared_distance);
	max_squared_distances.set(first_index, second_index, 
				  max_squared_distance);
	max_squared_distances.set(second_index, first_index, 
				  max_squared_distance);
      }
    }

    return true;
  }

  template<typename Global>
  void ComputePairwiseDistances(const Global &globals, 
				const ArrayList<Matrix *> &sets,
				const ArrayList<index_t> &indices) {
    
    for(index_t first_index = 0; first_index < sets.size(); first_index++) {
      const Matrix *first_set = sets[first_index];
      
      for(index_t second_index = first_index + 1; second_index < sets.size(); 
	  second_index++) {
	const Matrix *second_set = sets[second_index];

	double squared_distance =
	  la::DistanceSqEuclidean
	  (globals.dimension, first_set->GetColumnPtr(indices[first_index]),
	   second_set->GetColumnPtr(indices[second_index]));

	min_squared_distances.set(first_index, second_index, squared_distance);
	min_squared_distances.set(second_index, first_index, squared_distance);
      }
    }   
  }

  template<typename Global, typename QueryResult>
  void EvaluateMain(const Global &globals, const ArrayList<Matrix *> &sets,
		    QueryResult &results) {

    // Compute the pairwise distances among the points.
    ComputePairwiseDistances(globals, sets, 
			     globals.hybrid_node_chosen_indices);

    // Call the kernels.
    PositiveEvaluate(globals.hybrid_node_chosen_indices, sets, results);
    NegativeEvaluate(globals.hybrid_node_chosen_indices, sets, results);
  }

  template<typename Global>
  void EvaluateMain(const Global &globals, const ArrayList<Matrix *> &sets,
		    double *negative_result, double *positive_result) {

    // Compute the pairwise distances among the points.
    ComputePairwiseDistances(globals, sets, 
			     globals.hybrid_node_chosen_indices);

    // Call the kernels.
    *positive_result = 
      PositiveEvaluate(globals.hybrid_node_chosen_indices, sets);
    *negative_result =
      NegativeEvaluate(globals.hybrid_node_chosen_indices, sets);
  }

  template<typename Global, typename Tree, typename Delta>
  bool ComputeFiniteDifference(const Global &globals,
			       const ArrayList<Tree *> &nodes, Delta &delta) {
    
    // Compute the pairwise distances among the nodes.
    ComputePairwiseDistances(globals, nodes);
    
    // Call the kernels and compute the delta change.
    bool flag = PositiveEvaluate(nodes, delta);

    if(flag) {
      flag = NegativeEvaluate(nodes, delta);
    }
    return flag;
  }

  virtual double Gradient(double distance) = 0;

  double PositiveEvaluate(const ArrayList<index_t> &indices,
			  const ArrayList<Matrix *> &sets) {

    double positive_potential = PositiveEvaluateCommon_(min_squared_distances);
    return positive_potential;
  }

  double NegativeEvaluate(const ArrayList<index_t> &indices,
			  const ArrayList<Matrix *> &sets) {

    double negative_potential = NegativeEvaluateCommon_(min_squared_distances);
    return negative_potential;
  }

  template<typename QueryResult>
  void PositiveEvaluate(const ArrayList<index_t> &indices,
			const ArrayList<Matrix *> &sets, 
			QueryResult &results) {

    double positive_potential = PositiveEvaluateCommon_(min_squared_distances);
    
    for(index_t i = 0; i < sets.size(); i++) {
      results.positive_potential_bound[indices[i]] += positive_potential;
      results.positive_potential_e[indices[i]] += positive_potential;
    }
  }

  template<typename QueryResult>
  void NegativeEvaluate(const ArrayList<index_t> &indices,
			const ArrayList<Matrix *> &sets,
			QueryResult &results) {

    double negative_potential = NegativeEvaluateCommon_(min_squared_distances);
    
    for(index_t i = 0; i < sets.size(); i++) {
      results.negative_potential_bound[indices[i]] += negative_potential;
      results.negative_potential_e[indices[i]] += negative_potential;
    }
  }

  template<typename Tree, typename Delta>
  bool PositiveEvaluate(const ArrayList<Tree *> &nodes, Delta &delta) {

    double min_positive_potential = 
      PositiveEvaluateCommon_(max_squared_distances);

    if(isinf(min_positive_potential) || isnan(min_positive_potential)) {
      return false;
    }

    double max_positive_potential =
      PositiveEvaluateCommon_(min_squared_distances);
    
    if(isinf(max_positive_potential) || isnan(max_positive_potential)) {
      return false;
    }

    for(index_t i = 0; i < nodes.size(); i++) {
      delta.positive_potential_bound[i].Init
	(delta.n_pruned[i] * min_positive_potential, 
	 delta.n_pruned[i] * max_positive_potential);
      delta.positive_potential_e[i] = 0.5 * delta.n_pruned[i] *
	(min_positive_potential + max_positive_potential);
      delta.used_error[i] = delta.used_error[i] + 0.5 * delta.n_pruned[i] * 
	(max_positive_potential - min_positive_potential);
    }
    return true;
  }
  
  template<typename Tree, typename Delta>
  bool NegativeEvaluate(const ArrayList<Tree *> &nodes, Delta &delta) {

    double min_negative_potential = 
      NegativeEvaluateCommon_(min_squared_distances);

    if(isinf(min_negative_potential) || isnan(min_negative_potential)) {
      return false;
    }

    double max_negative_potential =
      NegativeEvaluateCommon_(max_squared_distances);
    
    if(isinf(max_negative_potential) || isnan(max_negative_potential)) {
      return false;
    }

    for(index_t i = 0; i < nodes.size(); i++) {
      delta.negative_potential_bound[i].Init
	(delta.n_pruned[i] * min_negative_potential,
	 delta.n_pruned[i] * max_negative_potential);
      delta.negative_potential_e[i] = 0.5 * delta.n_pruned[i] *
	(min_negative_potential + max_negative_potential);
      delta.used_error[i] = delta.used_error[i] + 0.5 * delta.n_pruned[i] *
	(max_negative_potential - min_negative_potential);
    }
    return true;
  }

  void SetZero() {
    min_squared_distances.SetZero();
    max_squared_distances.SetZero();
  }
};

#endif
