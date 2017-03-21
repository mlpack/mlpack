/**
 * @file shapiro_wilk_test.hpp
 *
 * Implementation of Exteneded Shapiro Wilk Test for normality
 * Refrences: https://tinyurl.com/q9gx9m2
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_SHAPIRO_WILK_TEST_IMPL_HPP
#define MLPACK_CORE_MATH_SHAPIRO_WILK_TEST_IMPL_HPP


#include <mlpack/prereqs.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/distributions/normal.hpp>

namespace mlpack {
namespace math {

struct Pval
{
  double pvalue;
  double W;
  double zscore;
  bool accept;
};

inline double InverseNormal(double prob, double mean =0 , double sd =1)
{
        boost::math::normal_distribution<>myNormal (mean, sd);
        return quantile(myNormal, prob);
};
template<typename eT>
Pval Shapiro(arma::Mat<eT>& dist, double alpha=0.05)
{
  //All the variable are consitent with refrence provided above
  std::mt19937 engine;  // Mersenne twister random number engine
  std::normal_distribution<> distr(0, 1.0);
  dist.reshape(dist.n_rows*dist.n_cols, 1);
  arma::vec B = arma::sort(dist);
  arma::vec m(B.n_rows);
  //Computes the m values.
  for(size_t i = 0; i < m.n_rows; i++)
      m(i) = InverseNormal((i+1 - 0.375) / (m.n_rows  +.25));

  double m_scalar = arma::dot(m,m);
  double u = 1/ sqrt(m.n_rows);
  arma::vec upoly(5);
  
  for (size_t i = 5; i > 0; i--)
  {
    upoly(5-i) = pow(u,i);
  }
  //upoly = [u^5, u^4, u^3, u^2, u^1, 1]
  upoly.insert_rows(upoly.n_rows,arma::ones(1));
  arma::vec a(m.n_rows);
  //Create of vector of constant for findin values of a[n]
  arma::vec a_n;
  a_n << -2.706056f << 4.434685f << -2.07119f << -.147981f << 0.221 << .0f;
  a_n[a_n.n_rows -1] = m(m.n_rows-1)*pow(m_scalar,-0.5);

  //Create of vector of constant for findin values of a[n]
  arma::vec a_n1;
  a_n1 << -3.582633f << 5.682633f << -1.752461f << -.293762f << .042981f << .0f;
  a_n1[a_n1.n_rows -1] = m(m.n_rows-2)*pow(m_scalar,-0.5);

  //a[n] = sigma_i(a_n[i] * upoly[i])
  a(a.n_rows-1) = arma::dot(a_n, upoly);
  a(a.n_rows-2) = arma::dot(a_n1, upoly);
  //a[i] = a[n-1-i]
  a(0) = -a(a.n_rows-1);
  a(1) = -a(a.n_rows-2);
   
  long double den = 1.0f - (2.0f* pow(a(a.n_rows -1),2.0f)) - (2.0f* pow(a(a.n_rows-2),2.0f));
  long double num = m_scalar - (2.0f * pow(m(m.n_rows -1),2.0f)) - (2.0f * pow(m(m.n_rows -2),2.0f));
  long double eps = float(num / den); 

  //a(i) = m(i) / sqrt(eps) for 2<= i <= n-2
  for (size_t i = 2; i < a.n_rows-2; ++i)
  {
    a(i) = m(i)/sqrt(eps);
  }
  //find w = sigma(a_i*x_i)^2 / sigma[(x_i-mean)^2]
  double mean = arma::mean(B);
  num = pow(arma::dot(a, B),2);
  den = arma::sum(pow(B-mean,2));
  double w = num/den;

  //Aproximate the z score with w
  double zmean = 0.0038915f*pow(log(B.n_rows),3) - 0.083751f*pow(log(B.n_rows),2) - 0.31082f*pow(log(B.n_rows),1) - 1.5861f;
  double temp = 0.0030302f*pow(log(B.n_rows),2) - 0.082676f*pow(log(B.n_rows),1) -0.4803f;
  double zvariance = exp(temp);
  double zscore = (log(1- w) - zmean) / zvariance;
  //find p value with respect to mean=0 and var=1 gaussian
  boost::math::normal stddist(0.0, 1.0);
  double pvalue = cdf(stddist, fabs(zscore));
  bool accept = pvalue>alpha?true:false;
  Pval ret = {pvalue, w, zscore, accept};
  return ret;
}
} // namespace math
} // namespace mlpack

#endif
