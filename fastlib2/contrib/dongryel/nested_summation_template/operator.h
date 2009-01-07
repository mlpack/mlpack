#ifndef OPERATOR_H
#define OPERATOR_H

#include "fastlib/fastlib.h"

class Operator {

 private:

  /** @brief The nested operators under this operator.
   */
  ArrayList<Operator *> operators_;

  /** @brief The list of dataset indices involved with this operator.
   */
  ArrayList<index_t> dataset_indices_;

  /** @brief The list of restrictions for each dataset index.
   */
  std::map<index_t, std::vector<int> > *restrictions_;

  /** @brief The ordered list of datasets.
   */
  ArrayList<Matrix *> datasets_;

  /** @brief A boolean flag that specifies whether the recursive
   *         result should be negated or not.
   */
  bool is_positive_;

  /** @brief A boolean flag that specifies whether the recursive
   *         result should be inverted or not.
   */
  bool should_be_inverted_;

  OT_DEF_BASIC(Operator) {
    OT_MY_OBJECT(operators_);
    OT_MY_OBJECT(indices_);
    OT_PTR_NULLABLE(restrictions_);
    OT_MY_OBJECT(is_positive_);
    OT_MY_OBJECT(should_be_inverted_);
  }

  void ChoosePpointIndex
  (std::map<index_t, index_t> &previous_constant_dataset_indices,
   index_t current_dataset_index) {
    
    // Choose a new point index subject to the existing restrictions...
    index_t new_point_index;
    bool done_flag = true;

    // Get the list of restrictions associated with the current
    // dataset index.
    std::map<index_t, std::vector<int> >::iterator restriction = 
      restrictions_->find(current_dataset_index);
    const std::vector<int> &restriction_vector = (*restriction).second;

    // The pointer to the current dataset.
    const Matrix *current_dataset = datasets_[current_dataset_index];

    if(restriction != restrictions_->end()) {
      
      do {
	
	// Reset the flag.
	done_flag = true;
	
	// Randomly choose the point index
	new_point_index = math::RandInt(0, current_dataset->n_cols());

	for(index_t n = 0; n < restriction_vector.size(); n++) {

	  // Look up the point index chosen for the current
	  // restriction...
	  index_t restriction_dataset_index = restriction_vector[n];

	  if(previous_constant_dataset_indices.find() !=
	     previous_constant_dataset_indices.end()) {

	    done_flag = false;
	    break;
	  }
	}
	
	// Repeat until all restrictions are satisfied...

      } while(!done_flag);
    }
    else {
      new_point_index = math::RandInt(0, current_dataset->n_cols());
    }

    previous_constant_data_indices[current_dataset_index] = new_point_index;
  }

  void ChoosePointIndices
  (std::map<index_t, index_t> &previous_constant_dataset_indices) {

    // Go through each dataset index to be chosen at this level, and
    // select an index, if required.
    for(index_t i = 0; i < dataset_indices_.size(); i++) {

      index_t current_dataset_index = dataset_indices_[i];

      if(previous_constant_dataset_indices.find(current_dataset_index) == 
	 previous_constant_dataset_indices.end()) {
	
	ChoosePointIndex(previous_constant_dataset_indices, 
			 current_dataset_index);
      }
    }
  }

 public:

  /** @brief Evaluate the operator exactly.
   */
  virtual double NaiveCompute
  (std::map<index_t, index_t> &constant_dataset_indices) = 0;

  /** @brief Evaluate the operator using Monte Carlo.
   */
  virtual double MonteCarloCompute
  (std::map<index_t, index_t> &constant_dataset_indices) = 0;

};

#endif
