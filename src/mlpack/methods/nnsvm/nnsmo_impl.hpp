/**
 * @file nnsmo_impl.hpp
 *
 * Implementation of the non-negative SMO algorithm.
 */
#ifndef __MLPACK_METHODS_NNSVM_NNSMO_IMPL_HPP
#define __MLPACK_METHODS_NNSVM_NNSMO_IMPL_HPP

namespace mlpack {
namespace nnsvm {

// return the support vector, the support alpha vector and the weight vector of the trained NNSVM
template<typename TKernel>
void NNSMO<TKernel>::GetNNSVM(arma::mat& support_vectors,
                              arma::vec& support_alpha,
                              arma::vec& w) const
{
  size_t n_support = 0;
  size_t i_support = 0;

  for (size_t i = 0; i < n_data_; i++)
  {
    if (alpha_[i] != 0)
    {
      n_support++;
    }
  }

  support_vectors.set_size(dataset_.n_rows - 1, n_support);
  support_alpha.set_size(n_support);

  for (size_t i = 0; i < n_data_; i++)
  {
    if (alpha_[i] != 0)
    {
      arma::vec source;
      arma::vec dest;
      GetVector_(i, source);
      dest = support_vectors.unsafe_col(i_support);
      dest = source;
      support_alpha[i_support] = alpha_[i] * GetLabelSign_(i);
      i_support++;
    }
  }

  w.set_size(n_feature_);
  for (size_t s = 0; s < n_feature_; s++)
    w[s] = math::ClampNonNegative(VTA_[s]);
}

//NNSMO training for 2-classes
template<typename TKernel>
void NNSMO<TKernel>::Train()
{
  bool examine_all = true;
  size_t num_changed = 0;
  size_t n_iter = 0;
  size_t counter = 1000;
  if (counter > n_data_)
    counter = n_data_;

  // calculate kernel_cache_sign_: [k_ij* y_i* y_j]
  CalcKernels_();
  while ((num_changed > 0 || examine_all))
  {
    //NNSMO training iterations
    num_changed = TrainIteration_(examine_all);

    if (examine_all)
      examine_all = false;
    else if (num_changed == 0)
      examine_all = true;

    //if exceed the maximum number of iterations, finished
    if (++n_iter == max_iter_)
    {
      fprintf(stderr, "Max iterations Reached! \n");
      break;
    }

    //for every max(n_data_, 1000) iterations, show progress
    if (n_iter % counter == 0 )
      fprintf(stderr, ".");
  }

  //compute the final objective value
  double obj = sum_alpha_ - w_square_sum_/2;
  fprintf(stderr, "iter=%zu, %zu, %f, %f, %f, obj=%f \n", n_iter, num_changed,
      thresh_, sum_alpha_, w_square_sum_, obj);
}

//NNSMO training iteration
template<typename TKernel>
size_t NNSMO<TKernel>::TrainIteration_(bool examine_all)
{
  size_t num_changed = 0;

  for (size_t i = 0; i < n_data_; i++)
  {
    if ((examine_all || !IsBound_(alpha_[i])) && TryChange_(i))
      num_changed++;
  }

  return num_changed;
}

// try to find the working set
//  outer loop: alpha_j, KKT violation
//  inner loop: alpha_i, maximum objective value increase with respective to
//  alpha_i, j
template<typename TKernel>
bool NNSMO<TKernel>::TryChange_(size_t j)
{
  double error_j = Error_(j);
  double rj = error_j * GetLabelSign_(j);


  if (!((rj < -NNSMO_TOLERANCE && alpha_[j] < c_)
      || (rj > NNSMO_TOLERANCE && alpha_[j] > 0)))
    return false; // nothing to change

  // first try the one we suspect to have the largest yield
  size_t i = -1;
  double df_max = 0;
  for (size_t k = 0; k < n_data_; k++)
  {
    double df_k = CalculateDF_(k, j, error_j);
    if (df_k > df_max)
    {
      df_max = df_k;
      i = k;
    }
  }

  if (i != (size_t) -1 && TakeStep_(i, j, error_j))
  {
    return true;
  }

  return false;
}

// Compute the increase of objective value with respect to updating of alpha_i,
// alpha_j
template<typename TKernel>
double NNSMO<TKernel>::CalculateDF_(size_t i, size_t j, double error_j)
{
  // 1. check i,j
  if (i == j)
  {
    return -1;
  }
  int yi = GetLabelSign_(i);
  int yj = GetLabelSign_(j);
  double alpha_i = alpha_[i];
  double alpha_j = alpha_[j];
  double l;
  double u;
  int s = (yi == yj) ? 1 : -1;
  double error_i = Error_(i);
  double r;

  // 2. compute L, H of alpha_j
  if (s < 0)
  {
    mlpack::Log::Assert(s == -1);
    r = alpha_j - alpha_i; // target values are not equal
  }
  else
  {
    r = alpha_j + alpha_i - c_; // target values are equal
  }

  l = math::ClampNonNegative(r);
  u = c_ + math::ClampNonPositive(r);

  if (l >= u - NNSMO_ZERO)
  {
    // TODO: might put in some tolerance
    return -1;
  }

  // 3. compute eta using cached kernel values
  double kii = EvalKernel_(i, i);
  double kij = EvalKernel_(i, j);
  double kjj = EvalKernel_(j, j);
  double eta = +2*kij - kii - kjj;

  // calculate alpha_j^{new}
  if (eta < 0)
  {
    alpha_j = alpha_[j] - yj * (error_i - error_j) / eta;
    alpha_j = FixAlpha_(math::ClampRange(alpha_j, l, u));
  }
  else
  {
    return -1;
  }
  alpha_j = FixAlpha_(alpha_j);
  double delta_alpha_j = alpha_j - alpha_[j];

  // check if there is progress
  if (fabs(delta_alpha_j) < eps_ * (alpha_j + alpha_[j] + eps_))
  {
    return -1;
  }

  // 4. compute increase of objective value
  arma::vec w(n_feature_);
  for (size_t s = 0; s < n_feature_; s++)
  {
    double VTdA_s = (dataset_(s, j) - dataset_(s, i)) * yj * delta_alpha_j;
    w[s] = math::ClampNonNegative(VTA_[s] + VTdA_s);
  }
  w_square_sum_ = dot(w, w);
  double delta_f = w_square_sum_ / 2;
  if (yi != yj)
    delta_f += 2* delta_alpha_j;

  return delta_f;
}

// Update alpha_i, alpha_j, as well as the VTA_, negation of intercept: thresh_
// and the error cache: error_
template<typename TKernel>
bool NNSMO<TKernel>::TakeStep_(size_t i, size_t j, double error_j)
{
  //1. check i,j
  if (i == j)
  {
    return false;
  }

  int yi = GetLabelSign_(i);
  int yj = GetLabelSign_(j);
  double alpha_i = alpha_[i];
  double alpha_j = alpha_[j];
  double l;
  double u;
  int s = (yi == yj) ? 1 : -1;
  double error_i = Error_(i);
  double r;

  //2. compute L, H of alpha_j
  if (s < 0)
  {
    mlpack::Log::Assert(s == -1);
    r = alpha_j - alpha_i; // target values are not equal
  }
  else
  {
    r = alpha_j + alpha_i - c_; // target values are equal
  }
  l = math::ClampNonNegative(r);
  u = c_ + math::ClampNonPositive(r);

  if (l >= u - NNSMO_ZERO)
  {
    // TODO: might put in some tolerance
    return false;
  }

  // 3. compute eta using cached kernel values
  double kii = EvalKernel_(i, i);
  double kij = EvalKernel_(i, j);
  double kjj = EvalKernel_(j, j);
  double eta = +2*kij - kii - kjj;

  // calculate alpha_j^{new}
  if (eta < 0)
  {
    alpha_j = alpha_[j] - yj * (error_i - error_j) / eta;
    alpha_j = FixAlpha_(math::ClampRange(alpha_j, l, u));
  }
  else
  {
    return false;
  }
  alpha_j = FixAlpha_(alpha_j);
  double delta_alpha_j = alpha_j - alpha_[j];

  // check if there is progress
  if (fabs(delta_alpha_j) < eps_*(alpha_j + alpha_[j] + eps_))
  {
    return false;
  }

  // calculate alpha_i^new
  alpha_i = alpha_i - (s)*(delta_alpha_j);
  if (alpha_i < NNSMO_ZERO)
  {
    alpha_j += s * alpha_i;
    alpha_i = 0;
  }
  else if (alpha_i > c_ - NNSMO_ZERO)
  {
    double t = alpha_i - c_;
    alpha_j += s * t;
    alpha_i = c_;
  }
  double delta_alpha_i = alpha_i - alpha_[i];
  delta_alpha_j = alpha_j - alpha_[j];

  // 4. update VTA_, w_square_sum_
  arma::vec w(n_feature_);
  for (size_t s = 0; s < n_feature_; s++)
  {
    double VTdA_s = dataset_(s, i) * yi * delta_alpha_i + dataset_(s, j) * yj * delta_alpha_j;
    VTA_[s] += VTdA_s;
    w[s] = math::ClampNonNegative(VTA_[s]);
  }
  w_square_sum_ = dot(w, w);

  // update alpha_, sum_alpha_
  alpha_[i] = alpha_i;
  alpha_[j] = alpha_j;
  sum_alpha_ += delta_alpha_i + delta_alpha_j;

  // update error cache and threshold
  double thresh_sum = 0;
  size_t nb_count = 0;
  for (size_t k = 0; k < n_data_ ; k++)
  {
    arma::vec x_k;
    GetVector_(k, x_k);
    error_[k] = dot(w, x_k) - GetLabelSign_(k);
    if(!IsBound_(alpha_[k]))
    {
      thresh_sum += error_[k];
      nb_count++;
    }
  }
  if(nb_count > 0)
    thresh_ = thresh_sum/nb_count;

  // update error cache using the new threshold
  for (size_t k = 0; k < n_data_ ; k++)
    error_[k] -= thresh_;

  return true;
}

}; // namespace nnsvm
}; // namespace mlpack

#endif // __MLPACK_METHODS_NNSVM_NNSMO_IMPL_HPP
