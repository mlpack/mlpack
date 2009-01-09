#ifndef FUNCTION_H
#define FUNCTION_H

#include "fastlib/fastlib.h"
#include "operator.h"

class Function: public Operator {

 protected:

  /** @brief The dataset indices that should be used to evaluate the
   *         function.
   */
  ArrayList<index_t> involved_dataset_indices_;

  /** @brief Returns a pairwise squared distance between two points.
   */
  double SquaredDistance_
  (std::map<index_t, index_t> &constant_dataset_indices) {

    // Look up the point indices involved with the function and
    // compute the pairwise distance.
    std::map<index_t, index_t>::iterator first_point_index_iterator = 
      constant_dataset_indices.find(involved_dataset_indices_[0]);
    std::map<index_t, index_t>::iterator second_point_index_iterator =
      constant_dataset_indices.find(involved_dataset_indices_[1]);
    index_t first_point_index = (*first_point_index_iterator).second;
    index_t second_point_index = (*second_point_index_iterator).second;
    index_t dimension = ((*datasets_)[involved_dataset_indices_[0]])->n_rows();
    const double *first_point = 
      ((*datasets_)[involved_dataset_indices_[0]])->GetColumnPtr
      (first_point_index);
    const double *second_point = 
      ((*datasets_)[involved_dataset_indices_[1]])->GetColumnPtr
      (second_point_index);
    
    return la::DistanceSqEuclidean(dimension, first_point, second_point);
  }

 public:

  /** @brief A function evaluation is evaluated exactly for naive and
   *         Monte Carlo style computation.
   */
  double MonteCarloCompute
  (std::map<index_t, index_t> &constant_dataset_indices) {
    return NaiveCompute(constant_dataset_indices);
  }

};

template<typename TKernel>
class KernelFunction: public Function {

 private:

  TKernel kernel_;

 public:

  void Init(double bandwidth_in) {
    kernel_.Init(bandwidth_in);
  }
  
  double NaiveCompute(std::map<index_t, index_t> &constant_dataset_indices) {

    double sqdist = SquaredDistance_(constant_dataset_indices);
    return kernel_.EvalUnnormOnSq(sqdist);
  }

};

#endif
