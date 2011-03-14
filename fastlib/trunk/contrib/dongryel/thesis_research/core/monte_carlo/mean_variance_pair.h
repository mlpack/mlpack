/** @file mean_variance_pair.h
 *
 *  The class implementation that represents a running sample mean and
 *  variance pair.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_MONTE_CARLO_MEAN_VARIANCE_PAIR_H
#define CORE_MONTE_CARLO_MEAN_VARIANCE_PAIR_H

#include <math.h>
#include "core/math/range.h"

namespace core {
namespace monte_carlo {

/** @brief The mean variance pair class.
 */
class MeanVariancePair {

  private:

    /** @brief The number of samples gathered.
     */
    int num_samples_;

    /** @brief The sample mean.
     */
    double sample_mean_;

    /** @brief The sample variance.
     */
    double sample_variance_;

  public:

    /** @brief The default constructor that sets everything to zero.
     */
    MeanVariancePair() {
      SetZero();
    }

    /** @brief The copy constructor.
     */
    MeanVariancePair(const MeanVariancePair &pair_in) {
      CopyValues(pair_in);
    }

    /** @brief Copies another MeanVariancePair object.
     */
    void CopyValues(const MeanVariancePair &pair_in) {
      num_samples_ = pair_in.num_samples();
      sample_mean_ = pair_in.sample_mean();
      sample_variance_ = pair_in.sample_variance();
    }

    /** @brief The assignment operator.
     */
    void operator=(const MeanVariancePair &pair_in) {
      this->CopyValues(pair_in);
    }

    /** @brief The number of samples gathered is returned.
     */
    int num_samples() const {
      return num_samples_;
    }

    /** @brief Returns the sample mean.
     */
    double sample_mean() const {
      return sample_mean_;
    }

    /** @brief Returns the variance of the sample mean.
     */
    double sample_mean_variance() const {
      return sample_variance_ / static_cast<double>(num_samples_ - 1);
    }

    /** @brief Returns the sample variance.
     */
    double sample_variance() const {

      // Note that this function scales this way to return the proper
      // variance.
      return sample_variance_ * (
               static_cast<double>(num_samples_) /
               static_cast<double>(num_samples_ - 1));
    }

    /** @brief Returns a scaled interval centered around the current
     *         sample mean with the given standard deviation factor.
     */
    void scaled_interval(
      double scale_in, double standard_deviation_factor,
      core::math::Range *interval_out) const {

      // Compute the sample mean variance.
      double sample_mv = this->sample_mean_variance();
      double error = standard_deviation_factor * sqrt(sample_mv);

      // In case no sample has been collected, then we need to set the
      // error to zero (since the variance will be infinite).
      if(num_samples_ == 0) {
        error = 0;
      }

      // Compute the interval.
      interval_out->lo = scale_in * (sample_mean_ - error);
      interval_out->hi = scale_in * (sample_mean_ + error);
    }

    /** @brief Sets everything to zero.
     */
    void SetZero() {
      num_samples_ = 0;
      sample_mean_ = 0;
      sample_variance_ = 0;
    }

    /** @brief Pushes a sample in.
     */
    void push_back(double sample) {

      // Update the number of samples.
      num_samples_++;
      double delta = sample - sample_mean_;

      // Update the sample mean.
      sample_mean_ = sample_mean_ + delta / ((double) num_samples_);

      // Update the sample variance.
      sample_variance_ = ((num_samples_ - 1) * sample_variance_ +
                          delta * (sample - sample_mean_)) /
                         ((double) num_samples_);
    }
};
}
}

#endif
