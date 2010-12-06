/** @file abstract_kernel.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_METRIC_KERNELS_ABSTRACT_KERNEL_H
#define CORE_METRIC_KERNELS_ABSTRACT_KERNEL_H

#include "core/math/range.h"

namespace core {
namespace metric_kernels {
class AbstractKernel {

  public:

    virtual std::string name() const = 0;

    virtual ~AbstractKernel() {
    }

    virtual double bandwidth_sq() const = 0;

    virtual void Init(double bandwidth_in, int dims) = 0;

    virtual void Init(double bandwidth_in) = 0;

    virtual double EvalUnnorm(double dist) const = 0;

    virtual double EvalUnnormOnSq(double sqdist) const = 0;

    virtual core::math::Range RangeUnnormOnSq(
      const core::math::Range& range) const = 0;

    virtual double MaxUnnormValue() const = 0;

    virtual double CalcNormConstant(int dims) const = 0;
};
};
};

#endif
