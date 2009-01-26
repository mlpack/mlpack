#ifndef MBP_KERHEL_H
#define MBP_KERHEL_H

#include "fastlib/fastlib.h"

class MultibodyPotentialKernel {

 public:

  Matrix min_squared_distances;

  Matrix max_squared_distances;

  virtual ~MultibodyPotentialKernel() {
  }

  virtual void Init(double bandwidth_in = -1.0) = 0;

  template<typename Global, typename Tree>
  void ComputePairwiseDistances(const Global &globals, 
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
    PositiveEvaluate(globals.hybrid_node_chosen_indices, sets, 
		     results.positive_potential_bound);
    NegativeEvaluate(globals.hybrid_node_chosen_indices, sets, 
		     results.negative_potential_bound);
  }

  virtual void PositiveEvaluate
  (const ArrayList<index_t> &indices, const ArrayList<Matrix *> &sets,
   ArrayList<DRange> &positive_potential_bounds) = 0;

  virtual void NegativeEvaluate
  (const ArrayList<index_t> &indices, const ArrayList<Matrix *> &sets,
   ArrayList<DRange> &negative_potential_bounds) = 0;

  void SetZero() {

    min_squared_distances.SetZero();
    max_squared_distances.SetZero();
  }
};

#endif
