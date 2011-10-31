/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file optimizers.cpp
 *
 * Implementation of the optimizers
 */
#include <mlpack/core.h>
#include "optimizers.hpp"

using namespace mlpack;
using namespace gmm;

void NelderMead::Eval(arma::mat& pts) {
  size_t dim = dimension();
  size_t num_func_eval;
  size_t i, j, ihi, ilo, inhi;
  size_t mpts = dim + 1;
  double sum, swap;
  arma::vec psum;
  long double swap_y, rtol, ytry, TINY = 1.0e-10;
  arma::vec y;
  arma::vec param_passed;

  // Default value is 1e-5.
  long double tol = CLI::GetParam<double>("opt/tolerance");

  // Default value is 50000.
  size_t NMAX = CLI::GetParam<int>("opt/MAX_FUNC_EVAL");

  param_passed.set_size(dim);
  psum.set_size(dim);
  num_func_eval = 0;
  y.set_size(mpts);
  for(i = 0; i < mpts; i++) {
    param_passed = pts.row(i);
    y[i] = (*func_ptr_)(param_passed, data_);
  }

  for(;;) {
    ilo = 0;
    ihi = y[0] > y[1] ? (inhi = 1, 0) : (inhi = 0, 1);
    for (i = 0; i < mpts; i++) {
      if(y[i] <= y[ilo])
        ilo = i;
      if(y[i] > y[ihi]) {
        inhi = ihi;
        ihi = i;
      }
      else if((y[i] > y[inhi]) && (i != ihi))
        inhi = i;
    }

    rtol = 2.0 * fabs(y[ihi] - y[ilo]) / (fabs(y[ihi]) + fabs(y[ilo]) + TINY);
    if (rtol < tol) {
      swap_y = y[0];
      y[0] = y[ilo];
      y[ilo] = swap_y;
      for (i = 0; i < dim; i++) {
        swap = pts(0, i);
        pts(0, i) = pts(ilo, i);
        pts(ilo, i) = swap;
      }

      break;
    }

    if (num_func_eval > NMAX) {
      Log::Warn << "Nelder-Mead: Maximum number of function evaluations "
          "exceeded." << std::endl;
      break;
    }

    num_func_eval += 2;

    // Beginning a new iteration.
    // Extrapolating by a factor of -1.0 through the face of the simplex
    // across from the high point, i.e, reflect the simplex from the high point
    for (j = 0 ; j < dim; j++) {
      sum = 0.0;
      for(i = 0; i < mpts; i++)
        if (i != ihi)
          sum += pts(i, j);

      psum[j] = sum / dim;
    }

    ytry = ModSimplex_(pts, y, psum, ihi, -1.0);

    if(ytry <= y[ilo]) {
      // result better than best point
      // so additional extrapolation by a factor of 2
      ytry = ModSimplex_(pts, y, psum, ihi, 2.0);
    } else if(ytry >= y[ihi]) {
      // result worse than the worst point
      // so there is a lower intermediate point,
      // i.e., do a one dimensional contraction
      ytry = ModSimplex_(pts, y, psum, ihi, 0.5);
      if(ytry > y[ihi]) {
        // Can't get rid of the high point,
        // try to contract around the best point
        for (i = 0; i < mpts; i++) {
          if (i != ilo) {
            for (j = 0; j < dim; j++)
              pts(i, j) = psum[j] = 0.5 * (pts(i, j) + pts(ilo, j));

            param_passed = psum;
            y[i] = (*func_ptr_)(param_passed, data());
          }
        }
        num_func_eval += dim;

        for (j = 0 ; j < dim ; j++) {
          sum = 0.0;
          for (i = 0 ; i < mpts ; i++)
            if (i != ihi)
              sum += pts(i, j);

          psum[j] = sum / dim;
        }
      }
    }
    else
      --num_func_eval;
  }

  CLI::GetParam<int>("opt/func_evals") = num_func_eval;
}

long double NelderMead::ModSimplex_(arma::mat& pts, arma::vec& y,
                                    arma::vec& psum, size_t ihi,
                                    float fac) {

  size_t j, dim = dimension();
  long double ytry;
  arma::vec ptry(dim);
  arma::vec param_passed(dim);

  for (j = 0; j < dim; j++)
    ptry[j] = psum[j] * (1 - fac) + pts(ihi, j) * fac;

  param_passed = ptry;
  ytry = (*func_ptr_)(param_passed, data());
  if (ytry < y[ihi]) {
    y[ihi] = ytry;
    for (j = 0; j < dim; j++)
      pts(ihi, j) = ptry[j];
  }

  return ytry;
}

void QuasiNewton::Eval(arma::vec& pt) {
  size_t n = dimension(), iters;
  size_t i, its;
  // Default value is 200.
  size_t MAXIMUM_ITERATCLINS = CLI::GetParam<int>("opt/MAX_ITERS");

  long double temp_1, temp_2, temp_3, temp_4, f_previous, f_min,
    maximum_step_length, sum = 0.0, sumdg, sumxi, temp, test;
  arma::vec dgrad, grad, hdgrad, xi;
  arma::vec pold, pnew;
  arma::mat hessian;

  // Default value is 3.0e-8.
  double EPSILON = CLI::GetParam<double>("opt/EPSILON");

  // Default value is 1.0e-5.
  double TOLERANCE = CLI::GetParam<double>("optTOLERANCE");

  // Default value is 100.
  double MAX_STEP_SIZE = CLI::GetParam<double>("opt/MAX_STEP_SIZE");

  // Default value is 1.0e-7.
  double g_tol = CLI::GetParam<double>("opt/gtol");

  dgrad.set_size(n);
  grad.set_size(n);
  hdgrad.set_size(n);
  hessian.set_size(n, n);
  pnew.set_size(n);
  xi.set_size(n);
  pold = pt;
  f_previous = (*func_ptr_)(pold, data(), grad);
  arma::vec tmp;
  tmp.ones(n);
  hessian.diag() = tmp;

  xi = -1 * grad;
  sum = dot(pold, pold);

  double fmax;
  if(sqrt(sum) > (float) n)
    fmax = sqrt(sum);
  else
    fmax = (float) n;

  maximum_step_length = MAX_STEP_SIZE * fmax;

  for (its = 0; its < MAXIMUM_ITERATCLINS; its++) {
    dgrad = grad;
    LineSearch_(pold, f_previous, grad, xi,
                pnew, f_min, maximum_step_length);
    f_previous = f_min;
    xi = pold - pnew;
    pold = pnew;
    pt = pold;

    test = 0.0;
    for (i = 0; i < n; i++) {
      if (fabs(pold[i]) > 1.0)
        fmax = fabs(pold[i]);
      else
        fmax = 1.0;

      temp = fabs(xi[i]) / fmax;
      if (temp > test)
        test = temp;
    }

    if (test < TOLERANCE) {
      iters = its;
      CLI::GetParam<int>("opt/iters") = iters;
      return;
    }

    test = 0.0;
    if (f_min > 1.0)
      temp_1 = f_min;
    else
      temp_1 = 1.0;

    for (i = 0; i < n; i++) {
      if (fabs(pold[i]) > 1.0)
        fmax = pold[i];
      else
        fmax = 1.0;

      temp = fabs(grad[i]) * fmax / temp_1;
      if (temp > test)
        test = temp;
    }

    if (test < g_tol) {
      iters = its;
      CLI::GetParam<int>("opt/iters") = iters;
      return;
    }

    dgrad -= grad;
    dgrad *= -1.0;
    hdgrad = hessian * dgrad;

    temp_2 = dot(dgrad, xi);
    temp_4 = dot(dgrad, hdgrad);
    sumdg = dot(dgrad, dgrad);
    sumxi = dot(xi, xi);

    if (temp_2 > sqrt(EPSILON * sumdg * sumxi)) {
      temp_2 = 1.0 / temp_2;
      temp_3 = 1.0 / temp_4;

      dgrad = temp_2 * xi;
      dgrad -= temp_3 * hdgrad;

      hessian += temp_2 * (xi * trans(xi));
      hessian -= temp_3 * (hdgrad * trans(hdgrad));
      hessian += temp_4 * (dgrad * trans(dgrad));
    }

    xi = -hessian * grad;
  }

  Log::Warn << "Too many iterations in Quasi-Newton: giving up." << std::endl;
}

void QuasiNewton::LineSearch_(arma::vec& pold, long double fold,
                              arma::vec& grad, arma::vec& xi,
                              arma::vec& pnew, long double& f_min,
                              long double maximum_step_length) {

  size_t i, n = dimension();
  long double a, step_length, previous_step_length = 0.0,
    minimum_step_length, b, disc, previous_f_value = 0.0,
    rhs1, rhs2, slope, sum, temp, test, temp_step_length,
    MIN_DECREASE = 1.0e-4, TOLERANCE = 1.0e-7;

  sum = sqrt(dot(xi, xi));
  if(sum > maximum_step_length)
    xi *= (maximum_step_length / sum);

  slope = dot(grad, xi);
  if (slope >= 0.0)
    return;

  test = 0.0;
  for (i = 0; i < n; i++) {
    double fmax;
    fmax = (fabs(pold[i]) > 1.0 ? fabs(pold[i]) : 1.0);
    temp = fabs(xi[i]) / fmax;
    if (temp > test)
      test = temp;
  }

  minimum_step_length = TOLERANCE / test;
  step_length = 1.0;

  for (;;) {
    pnew = pold + step_length * xi;

    f_min = (*func_ptr_)(pnew, data(), grad);

    if (step_length < minimum_step_length) {
      pnew = pold;
      return;
    } else if(f_min <= fold + MIN_DECREASE * step_length * slope) {
      return;
    } else {
      if (step_length == 1.0)
        temp_step_length = -slope / (2.0 * (f_min - fold - slope));
      else {
        rhs1 = f_min - fold - step_length * slope;
        rhs2 = previous_f_value - fold - previous_step_length * slope;
        a = (rhs1 / (step_length * step_length)
            - rhs2 / (previous_step_length * previous_step_length))
            / (step_length - previous_step_length);
        b = (-previous_step_length * rhs1 / (step_length * step_length) +
            step_length * rhs2 / (previous_step_length * previous_step_length))
            / (step_length - previous_step_length);

        if (a == 0.0)
          temp_step_length = -slope / (2.0*b);
        else {
          disc = b * b - 3.0 * a * slope;
          if(disc < 0.0)
            temp_step_length = 0.5 * step_length;
          else if (b <= 0.0)
            temp_step_length = (-b + sqrt(disc)) / (3.0 * a);
          else
            temp_step_length = -slope / (b + sqrt(disc));
        }

        if(temp_step_length > 0.5 * step_length)
          temp_step_length = 0.5 * step_length;
      }
    }

    previous_step_length = step_length;
    previous_f_value = f_min;
    step_length = (temp_step_length > 0.1 * step_length) ? temp_step_length
        : 0.1 * step_length;
  }
}
