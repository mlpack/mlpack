#ifndef FUNCTCLIN_H
#define FUNCTCLIN_H

#include "mlpack/core.h"
#include "operator.h"

class Function: public Operator {

 protected:

  /** @brief The dataset indices that should be used to evaluate the
   *         function.
   */
   std::vector<size_t> involved_dataset_indices_;

  /** @brief Returns a pairwise squared distance between two points.
   */
  double SquaredDistance_
  (std::map<size_t, size_t> &constant_dataset_indices) {

    // Look up the point indices involved with the function and
    // compute the pairwise distance.
    std::map<size_t, size_t>::iterator first_point_index_iterator =
      constant_dataset_indices.find(involved_dataset_indices_[0]);
    std::map<size_t, size_t>::iterator second_point_index_iterator =
      constant_dataset_indices.find(involved_dataset_indices_[1]);
    size_t first_point_index = (*first_point_index_iterator).second;
    size_t second_point_index = (*second_point_index_iterator).second;
    size_t dimension = (datasets_[involved_dataset_indices_[0]]).n_rows;
    double total = 0.0;
    for (size_t d = 0; d < dimension; ++d)
    {
      double diff = (datasets_[involved_dataset_indices_[0]])(d, first_point_index)-
                    (datasets_[involved_dataset_indices_[1]])(d, second_point_index);
      total = total + diff * diff;
    }
    /*const double *first_point =
      ((*datasets_)[involved_dataset_indices_[0]])->GetColumnPtr
      (first_point_index);
    const double *second_point =
      ((*datasets_)[involved_dataset_indices_[1]])->GetColumnPtr
      (second_point_index);

    return la::DistanceSqEuclidean(dimension, first_point, second_point);*/
    return total;
  }

};

template<typename TKernel>
class KernelFunction: public Function {

 private:

  TKernel kernel_;

 public:

  void InitKernelFunction(size_t first_dataset_index,
			  size_t second_dataset_index, double bandwidth_in) {

    involved_dataset_indices_.resize(2, 0);
    involved_dataset_indices_[0] = first_dataset_index;
    involved_dataset_indices_[1] = second_dataset_index;
    kernel_.Init(bandwidth_in);
  }
  
  double NaiveCompute(std::map<size_t, size_t> &constant_dataset_indices) {

    double sqdist = SquaredDistance_(constant_dataset_indices);
    return kernel_.EvalUnnormOnSq(sqdist);
  }

  /** @brief A function evaluation is evaluated exactly for naive and
   *         Monte Carlo style computation.
   */
  double MonteCarloCompute
  (std::vector<Strata> &list_of_strata,
   std::map<size_t, size_t> &constant_dataset_indices,
   double relative_error, double probability) {
    
    return NaiveCompute(constant_dataset_indices);
  }

};

#endif
