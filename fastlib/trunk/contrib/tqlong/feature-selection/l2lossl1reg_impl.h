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
  weight_ = randu(dim())*4*lambda_;
  bias_ = 0;
  L_ = 1;

  for (int i = 0; i < n(); i++)
    L_ += dot(data_.x(i), data_.x(i));
  r0_ = residual(weight_, bias_);
  r_ = r0_;

  for (int iter = 0; iter < maxIter_; iter++) {
    double bias_grad;
    x_type grad = calGrad(weight_, bias_, bias_grad);
    x_type z = weight_ - grad/L_;
    double bz = bias_ - bias_grad/L_;
    x_type new_weight = l1SoftThreshold(z, lambda_/L_);
    double new_bias = l1SoftThreshold(bz, lambda_/L_);

    double old_r = r_;
    r_ = residual(new_weight, new_bias);
//    double dbias = new_bias - bias_;
//    vec dweight = new_weight - weight_;
//    double s = (r_ - old_r - dot(dweight, grad)+bias_grad*dbias)/(0.5*(pow(norm(dweight, 2),2)+dbias*dbias));
////    std::cout << "old_r = " << old_r << " new_r = " << r_ << "\n";
//    if (s > L_) {
//      L_ *= 1.5;
//      //if (L_ >= s) L_ = s;
//      iter--;
//      std::cout << "Increase L = " << L_ << "\n";
//      continue;
//    }

    terminationConditions(new_weight, new_bias);
    std::cout << "iter = " << iter
        //<< " weight = \n" << weight_
        << " objective = " << r_ + lambda_*norm(weight_,1)
        << "\n";

    weight_ = new_weight;
    bias_ = new_bias;
    if (terminated_) {
      std::cout << "terminated\n";
      break;
    }
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
typename L2LossL1Reg<D>::x_type L2LossL1Reg<D>::calGrad(const x_type& w, double bias, double &bias_grad)
{
  x_type g(dim());
  g.fill(0.0);
  bias_grad = 0;
  for (int i = 0; i < n(); i++) {
    double r = dot(data_.x(i), w)+bias-data_.y(i);
    g += r * data_.x(i);
    bias_grad += r;
  }
  return g;
}

template <typename D>
typename L2LossL1Reg<D>::x_type L2LossL1Reg<D>::l1SoftThreshold(const x_type& z, double mu)
{
  x_type w(dim());
  for (int i = 0; i < dim(); i++)
    w[i] = l1SoftThreshold(z[i], mu);
  return w;
}

template <typename D>
void L2LossL1Reg<D>::terminationConditions(const x_type& w, double bias)
{
  terminated_ = false;

  if (r_ < atol_) {
    std::cout << "Small residual ... ";
    terminated_ = true;
  }
  if (norm(w-weight_,2) < atol_) {
    std::cout << "Small variable change ... ";
    terminated_ = true;
  }
}

template <typename D>
double L2LossL1Reg<D>::residual(const x_type& w, double bias)
{
  double r = 0;
  for (int i = 0; i < n(); i++) {
    double d = dot(data_.x(i), w)+bias - data_.y(i);
    r += d*d;
  }
  return 0.5*r;
}

template <typename D>
void L2LossL1Reg<D>::save(const char* fileName, arma::file_type type)
{
  std::ofstream f(fileName);
  weight_.save(f, type);
  f << "\n";
  f << "bias = \n" << bias_ << "\n";
  f.close();
}
#endif
