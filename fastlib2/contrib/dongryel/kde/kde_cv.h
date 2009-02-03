#ifndef KDE_CV_H
#define KDE_CV_H

#include "contrib/dongryel/proximity_project/general_spacetree.h"
#include "contrib/dongryel/proximity_project/gen_kdtree.h"
#include "contrib/dongryel/nested_summation_template/function.h"
#include "contrib/dongryel/nested_summation_template/operator.h"
#include "contrib/dongryel/nested_summation_template/ratio.h"
#include "contrib/dongryel/nested_summation_template/sum.h"

template<typename TKernel>
class KdeCV {

 private:

  /** @brief The type of the tree used for the algorithm.
   */
  typedef GeneralBinarySpaceTree<DHrectBound<2>, Matrix > TreeType;

  /** @brief The root of the operator.
   */
  Operator *operator_root_;

  /** @brief The restrictions associated with the operation.
   */
  std::map<index_t, std::vector<index_t> > restrictions_;

  /** @brief The list of datasets associated with the operation.
   */
  ArrayList<Matrix *> datasets_;

  /** @brief The copy of the reference set.
   */
  Matrix reference_set_;

  /** @brief The tree used for the stratified sampling.
   */
  TreeType *data_root_;

 public:

  void set_bandwidth(double bandwidth_in) {
    
  }

  void Init(const Matrix &reference_set_in) {

    // Make a copy of the dataset and make a tree out of it.
    reference_set_.Copy(reference_set_in);
    data_root_ = proximity::MakeGenKdTree<double, TreeType,
      proximity::GenKdTreeMedianSplitter>(reference_set_, 40, 
					  (ArrayList<index_t> *) NULL, NULL);

    // Push in the pointer twice for the dataset since the query
    // equals the reference...
    datasets_.Init(2);
    datasets_[0] = &reference_set_;
    datasets_[1] = &reference_set_;

    // Allocate the kernel function...
    KernelFunction<TKernel> *kernel_function = new KernelFunction<TKernel>();
    kernel_function->Init(-1, &restrictions_, &datasets_, true, false);

    // Just initialize to some bandwidths for a moment...
    kernel_function->InitKernelFunction(0, 1, 0.1);

    // Allocate the inner sum.
    Sum *inner_sum = new Sum();
    inner_sum->Init(1, &restrictions_, &datasets_, true, false);

    // Allocate the outer sum.
    operator_root_ = new Sum();
    operator_root_->Init(0, &restrictions_, &datasets_, true, false);
    
    // The outer sum owns the inner sum, which owns the kernel
    // function.
    operator_root_->add_child_operator(inner_sum);
    inner_sum->add_child_operator(kernel_function);
  }
  
  double MonteCarloCompute() {

    /*
    // Set of dataset indices that have been chosen throughout the
    // computation; this acts as a stack of arguments.
    std::map<index_t, index_t> constant_dataset_indices;

    return operator_root_->MonteCarloCompute(constant_dataset_indices);   
    */
    return 0;
  }

  double NaiveCompute() {
    
    // Set of dataset indices that have been chosen throughout the
    // computation; this acts as a stack of arguments.
    std::map<index_t, index_t> constant_dataset_indices;

    return operator_root_->NaiveCompute(constant_dataset_indices);
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
