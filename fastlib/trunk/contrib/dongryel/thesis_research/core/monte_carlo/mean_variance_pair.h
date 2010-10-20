/** @file mean_variance_pair.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_MONTE_CARLO_MEAN_VARIANCE_PAIR_H
#define CORE_MONTE_CARLO_MEAN_VARIANCE_PAIR_H

namespace core {
namespace monte_carlo {
class MeanVariancePair {

  private:
    int num_samples_;

    double sample_mean_;

    double sample_variance_;

  public:

    MeanVariancePair() {
      SetZero();
    }

    void Copy(const MeanVariancePair &pair_in) {
      num_samples_ = pair_in.num_samples();
      sample_mean_ = pair_in.sample_mean();
      sample_variance_ = pair_in.sample_variance();
    }

    int num_samples() const {
      return num_samples_;
    }

    double sample_mean() const {
      return sample_mean_;
    }

    double sample_mean_variance() const {
      return sample_variance_ / ((double) num_samples_);
    }

    double sample_variance() const {
      return sample_variance_;
    }

    void scaled_interval(
      double scale_in, double standard_deviation_factor,
      core::math::Range *interval_out) const {

      // Compute the sample mean variance.
      double sample_mean_variance = this->sample_mean_variance();
      double error = standard_deviation_factor * sqrt(sample_mean_variance);

      // In case no sample has been collected, then we need to set the
      // error to zero (since the variance will be infinite).
      if(num_samples_ == 0) {
        error = 0;
      }

      // Compute the interval.
      interval_out->lo = scale_in * (sample_mean_ - error);
      interval_out->hi = scale_in * (sample_mean_ + error);
    }

    void SetZero() {
      num_samples_ = 0;
      sample_mean_ = 0;
      sample_variance_ = 0;
    }

    void Add(const MeanVariancePair &mv_pair_in) {
      sample_mean_ += mv_pair_in.sample_mean();
      sample_variance_ += mv_pair_in.sample_variance();
    }

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
};
};

#endif
