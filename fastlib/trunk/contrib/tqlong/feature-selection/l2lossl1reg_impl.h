/*
 * l2lossl1reg_impl.h
 *
 *  Created on: Feb 14, 2011
 *      Author: tqlong
 */

#ifndef L2LOSSL1REG_IMPL_H_
#define L2LOSSL1REG_IMPL_H_

#include "l2lossl1reg.h"
#include <armadillo>
#include <iostream>

using namespace arma;

template <typename D>
L2LossL1Reg<D>::L2LossL1Reg(const dataset_type& data) : data_(data)
{
}

template <typename D>
L2LossL1Reg<D>::~L2LossL1Reg()
{
}

template <typename D>
void L2LossL1Reg<D>::run()
{
  weight_ = x_type(dim()).fill(0.0);
  L_ = 0;

  for (int i = 0; i < n(); i++)
    L_ += dot(data_.x(i), data_.x(i));

  for (int iter = 0; iter < maxIter_; iter++) {
    x_type grad = calGrad(weight_);
    x_type z = weight_ - grad/L_;
    x_type new_weight = l1SoftThreshold(z, lambda_/L_);
    std::cout << "iter = " << iter << " weight = \n" << weight_ << "\n";

    if (norm(new_weight - weight_, 2) < atol_) {
      std::cout << "Small change ... terminated\n";
      break;
    }
    weight_ = new_weight;
  }
}

template <typename D>
void L2LossL1Reg<D>::setParameter(const std::vector<double>& params)
{
  int np = params.size();
  lambda_ = 0.5;
  maxIter_ = 100;
  atol_ = 1e-5;
  if (np > 0) lambda_ = params[0];
  if (np > 1) maxIter_ = params[1];
  if (np > 2) atol_ = params[2];
}

template <typename D>
typename L2LossL1Reg<D>::x_type L2LossL1Reg<D>::calGrad(const x_type& w)
{
  x_type g(dim());
  g.fill(0.0);
  for (int i = 0; i < n(); i++)
    g += (dot(data_.x(i), w)-data_.y(i)) * data_.x(i);
  return g;
}

template <typename D>
typename L2LossL1Reg<D>::x_type L2LossL1Reg<D>::l1SoftThreshold(const x_type& z, double mu)
{
  x_type w(dim());
  for (int i = 0; i < dim(); i++) {
    if (z[i] > mu) w[i] = z[i] - mu;
    else if (z[i] < -mu) w[i] = z[i]+mu;
    else w[i] = 0;
  }
  return w;
}
#endif
