/*
 * l2lossl1reg.h
 *
 *  Created on: Feb 14, 2011
 *      Author: tqlong
 */

#ifndef L2LOSSL1REG_H_
#define L2LOSSL1REG_H_

template <typename D>
class L2LossL1Reg {
public:
  typedef D         dataset_type;
  typedef typename D::x_type x_type;
  typedef typename D::y_type y_type;
  typedef typename D::s_type s_type;
  typedef typename D::x_type result_type;
  L2LossL1Reg(const dataset_type& data);
  virtual ~L2LossL1Reg();

  void run();
  void setParameter(const std::vector<double>& params);
  inline const result_type& result() const { return weight_; }
  inline int dim() const { return data_.dim(); }
  inline int n() const { return data_.n(); }
protected:
  x_type calGrad(const x_type& w);
  x_type l1SoftThreshold(const x_type& z, double mu);

  const dataset_type& data_;
  result_type weight_;

  int maxIter_;
  double L_, lambda_;
  double atol_, rtol_;
};

#include "l2lossl1reg_impl.h"

#endif /* L2LOSSL1REG_H_ */

