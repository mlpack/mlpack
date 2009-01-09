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

 public:

  void set_bandwidth(double bandwidth_in) {
    
  }

  void Init() {

    KernelFunction<TKernel> *kernel_function = new KernelFunction<TKernel>();
    Sum *inner_sum = new Sum();
    root_ = new Sum();
    
    root_->add_child_operator(inner_sum);
    inner_sum->add_child_operator(kernel_function);
  }
  
};

#endif
