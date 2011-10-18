#ifndef KDE_CV_H
#define KDE_CV_H

#include "mlpack/core/tree/hrectbound.h"

#include "../proximity_project/general_spacetree.h"
#include "../proximity_project/gen_kdtree.h"
#include "../nested_summation_template/function.h"
#include "../nested_summation_template/operator.h"
#include "../nested_summation_template/ratio.h"
//#include <log.h>
#include "../nested_summation_template/sum.h"

template<typename TKernel>
class KdeCV {

 private:

  /** @brief The type of the tree used for the algorithm.
   */
  typedef GeneralBinarySpaceTree<bound::HRectBound<2>, arma::mat > TreeType;

  /** @brief The root of the operator.
   */
  Operator *operator_root_;

  /** @brief The restrictions associated with the operation.
   */
  std::map<size_t, std::vector<size_t> > restrictions_;

  /** @brief The list of datasets associated with the operation.
   */
  std::vector<arma::mat> datasets_;

  /** @brief The copy of the reference set.
   */
  arma::mat reference_set_;

  /** @brief The tree used for the stratified sampling.
   */
  TreeType *data_root_;

 public:

  void set_bandwidth(double bandwidth_in) {

  }

  void Init(const arma::mat &reference_set_in) {

    // Make a copy of the dataset and make a tree out of it.
    reference_set_ = arma::mat(reference_set_in.n_rows, reference_set_in.n_cols);
    for (size_t row = 0; row < reference_set_in.n_rows; ++row)
    {
      for (size_t column = 0; column < reference_set_in.n_cols; ++column)
      {
        reference_set_(row,column) = reference_set_in(row, column);
      }
    }
    data_root_ = proximity::MakeGenKdTree<double, TreeType,
      proximity::GenKdTreeMedianSplitter>(reference_set_, 40,
					  (std::vector<size_t> *) NULL, NULL);

    // Push in the pointer twice for the dataset since the query
    // equals the reference...
    datasets_.resize(2);
    datasets_[0] = reference_set_;
    datasets_[1] = reference_set_;

    // Allocate the kernel function...
    KernelFunction<TKernel> *kernel_function = new KernelFunction<TKernel>();
    kernel_function->Init(-1, restrictions_, datasets_, true, false);

    // Just initialize to some bandwidths for a moment...
    kernel_function->InitKernelFunction(0, 1, 0.1);

    // Allocate the inner sum.
    Sum *inner_sum = new Sum();
    inner_sum->Init((size_t)1, restrictions_, datasets_, true, false);

    // Allocate the outer sum.
    operator_root_ = new Sum();
    operator_root_->Init((size_t)0, restrictions_, datasets_, true, false);
    
    // The outer sum owns the inner sum, which owns the kernel
    // function.
    operator_root_->add_child_operator(inner_sum);
    inner_sum->add_child_operator(kernel_function);
  }
  
  double MonteCarloCompute() {

    /*
    // Set of dataset indices that have been chosen throughout the
    // computation; this acts as a stack of arguments.
    std::map<size_t, size_t> constant_dataset_indices;

    return operator_root_->MonteCarloCompute(constant_dataset_indices);   
    */
    return 0;
  }

  double NaiveCompute()
  {
    // Set of dataset indices that have been chosen throughout the
    // computation; this acts as a stack of arguments.
    std::map<size_t, size_t> constant_dataset_indices;

    return operator_root_->NaiveCompute(constant_dataset_indices);
  }

  double BaseNaiveCompute()
  {
    double sum = 0.0;
    TKernel kernel;
    kernel.Init(0.1);
    const arma::mat references = datasets_[0];
    int dimension = references.n_rows;
    for(size_t i = 0; i < references.n_cols; i++)
    {
      for(size_t j = 0; j < references.n_cols; j++)
      {
        double dsqd = 0.0;
        for (size_t d = 0; d < dimension; ++d)
        {
          double diff = references(d,i) - references(d,j);
          dsqd += diff * diff;
        }
        sum += kernel.EvalUnnormOnSq(dsqd);
      }
    }
    return sum;
  }

};

#endif
