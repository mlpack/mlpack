/*
 * loglossl1reg.h
 *
 *  Created on: Feb 14, 2011
 *      Author: tqlong
 */

#ifndef LOGLOSSL1REG_H_
#define LOGLOSSL1REG_H_

#include "l2lossl1reg.h"
#include <boost/math/special_functions/log1p.hpp>

template <typename D>
class LogLossL1Reg : public L2LossL1Reg<D>
{
public:
  typedef D                    dataset_type;
  typedef L2LossL1Reg<D>       __Base;
  typedef typename D::x_type   x_type;
  typedef typename D::y_type   y_type;
  typedef typename D::s_type   s_type;
  typedef typename D::x_type   result_type;
  LogLossL1Reg(const dataset_type& data) : L2LossL1Reg<D>(data) {}

  x_type calGrad(const x_type& w, double bias, double &bias_grad) {
    x_type g(__Base::dim());
    g.fill(0.0);
    bias_grad = 0;
    for (int i = 0; i < __Base::n(); i++) {
      double ywx = __Base::data_.y(i)*dot(__Base::data_.x(i), w)+bias;
      double s = 1/(1+exp(-ywx))*exp(-ywx)*(-__Base::data_.y(i));
      g += s*__Base::data_.x(i);
      bias_grad += s;
    }
    return g;
  }

  double residual(const x_type& w, double bias) {
    double r = 0;
    for (int i = 0; i < __Base::n(); i++) {
      double ywx = __Base::data_.y(i)*dot(__Base::data_.x(i), w)+bias;
      r += log1p(exp(-ywx));
    }
    return r;
  }
};

#endif /* LOGLOSSL1REG_H_ */
