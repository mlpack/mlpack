#ifndef KDE_CV_H
#define KDE_CV_H

#include "contrib/dongryel/nested_summation_template/function.h"
#include "contrib/dongryel/nested_summation_template/operator.h"
#include "contrib/dongryel/nested_summation_template/ratio.h"
#include "contrib/dongryel/nested_summation_template/sum.h"

template<typename TKernel>
class KdeCV {

 private:

  /** @brief The root of the operator.
   */
  Operator *root_;

  /** @brief The restrictions associated with the operation.
   */
  std::map<index_t, std::vector<index_t> > restrictions_;

  /** @brief The list of datasets associated with the operation.
   */
  ArrayList<Matrix *> datasets_;

 public:

  void set_bandwidth(double bandwidth_in) {
    
  }

  void Init(Matrix &reference_set) {

    // Push in the pointer twice for the dataset since the query
    // equals the reference...
    datasets_.Init(2);
    datasets_[0] = &reference_set;
    datasets_[1] = &reference_set;

    // Allocate the kernel function...
    KernelFunction<TKernel> *kernel_function = new KernelFunction<TKernel>();
    kernel_function->Init(-1, &restrictions_, &datasets_, true, false);

    // Just initialize to some bandwidths for a moment...
    kernel_function->InitKernelFunction(0, 1, 0.1);

    // Allocate the inner sum.
    Sum *inner_sum = new Sum();
    inner_sum->Init(1, &restrictions_, &datasets_, true, false);

    // Allocate the outer sum.
    root_ = new Sum();
    root_->Init(0, &restrictions_, &datasets_, true, false);
    
    // The outer sum owns the inner sum, which owns the kernel
    // function.
    root_->add_child_operator(inner_sum);
    inner_sum->add_child_operator(kernel_function);
  }
  
  double NaiveCompute() {
    
    // Set of dataset indices that have been chosen throughout the
    // computation; this acts as a stack of arguments.
    std::map<index_t, index_t> constant_dataset_indices;

    printf("Starting!\n");
    return root_->NaiveCompute(constant_dataset_indices);
  }

  double BaseNaiveCompute() {

    double sum = 0.0;
    TKernel kernel;
    kernel.Init(0.1);
    const Matrix &references = *(datasets_[0]);
    for(index_t i = 0; i < references.n_cols(); i++) {
      const double *point_i = references.GetColumnPtr(i);
      for(index_t j = 0; j < references.n_cols(); j++) {
	const double *point_j = references.GetColumnPtr(j);
	int dimension = references.n_rows();
	
	double dsqd = la::DistanceSqEuclidean(dimension, point_i, point_j);
	
	sum += kernel.EvalUnnormOnSq(dsqd);
      }
    }
    return sum;
  }

};

#endif
