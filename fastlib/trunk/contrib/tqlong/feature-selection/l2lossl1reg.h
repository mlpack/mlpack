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
  void save(const char* fileName, arma::file_type type = arma::arma_ascii);
  inline const result_type& weight() const { return weight_; }
  inline double bias() const { return bias_; }
  inline int dim() const { return data_.dim(); }
  inline int n() const { return data_.n(); }
protected:
  virtual x_type calGrad(const x_type& w, double bias, double &bias_grad);
  x_type l1SoftThreshold(const x_type& z, double mu);
  inline double l1SoftThreshold(double z, double mu) { return z > mu ? z-mu : (z < -mu ? z+mu : 0); }
  void terminationConditions(const x_type& w, double bias);
  virtual double residual(const x_type& w, double bias);

  const dataset_type& data_;
  result_type weight_;
  double bias_;

  int maxIter_;
  double L_, lambda_;
  double atol_, rtol_;

  double r0_, r_;
  bool terminated_;
};

#include "l2lossl1reg_impl.h"

#endif /* L2LOSSL1REG_H_ */

