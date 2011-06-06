/** @file mean_variance_pair.h
 *
 *  The class implementation that represents a running sample mean and
 *  variance pair.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_MONTE_CARLO_MEAN_VARIANCE_PAIR_H
#define CORE_MONTE_CARLO_MEAN_VARIANCE_PAIR_H

#include <boost/serialization/serialization.hpp>
#include <math.h>
#include "core/math/math_lib.h"
#include "core/math/range.h"

namespace core {
namespace monte_carlo {

/** @brief The mean variance pair class.
 */
class MeanVariancePair {

  private:

    // For BOOST serialization.
    friend class boost::serialization::access;

    /** @brief Sets the total number of terms from which the samples
     *         are collected.
     */
    int total_num_terms_;

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

    /** @brief Serializes the mean variance pair object.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & total_num_terms_;
      ar & num_samples_;
      ar & sample_mean_;
      ar & sample_variance_;
    }

    /** @brief Sets the sample mean.
     */
    void set_sample_mean(double new_sample_mean) {
      sample_mean_ = new_sample_mean;
    }

    /** @brief Sets the total number of terms.
     */
    void set_total_num_terms(int total_num_terms_in) {
      total_num_terms_ = total_num_terms_in;
    }

    /** @brief Returns the total number of terms.
     */
    int total_num_terms() const {
      return total_num_terms_;
    }

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
      total_num_terms_ = pair_in.total_num_terms();
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
      if(num_samples_ <= 1) {
        return 0.0;
      }
      else {
        return sample_variance_ / static_cast<double>(num_samples_ - 1);
      }
    }

    /** @brief Returns the sample variance.
     */
    double sample_variance() const {

      // Note that this function scales this way to return the proper
      // variance.
      if(num_samples_ <= 1) {
        return 0.0;
      }
      else {
        return sample_variance_ * (
                 static_cast<double>(num_samples_) /
                 static_cast<double>(num_samples_ - 1));
      }
    }

    /** @brief Returns a scaled interval centered around the current
     *         sample mean with the given standard deviation factor.
     */
    void scaled_interval(
      double scale_in, double standard_deviation_factor,
      core::math::Range *interval_out) const {

      // Compute the interval.
      double scaled_dev = this->scaled_deviation(
                            scale_in, standard_deviation_factor);
      interval_out->lo = scale_in * sample_mean_ - scaled_dev;
      interval_out->hi = scale_in * sample_mean_ + scaled_dev;
    }

    /** @brief Returns the scaled deviation.
     */
    double scaled_deviation(
      double scale_in, double num_standard_deviations) const {

      // Compute the sample mean variance.
      double sample_mv = this->sample_mean_variance();
      double error = num_standard_deviations * sqrt(sample_mv);

      // In case at most one sample has been collected, then we need
      // to set the error to zero (since the variance will be
      // infinite).
      if(num_samples_ <= 1) {
        error = 0;
      }
      return scale_in * error;
    }

    void scale(double scale_in) {

      // Variance is multiplied by the square of the scale.
      sample_variance_ *= core::math::Sqr(scale_in);

      // The mean is multiplied by the scale.
      sample_mean_ *= scale_in;
    }

    /** @brief Sets everything to zero.
     */
    void SetZero() {
      total_num_terms_ = 0;
      num_samples_ = 0;
      sample_mean_ = 0;
      sample_variance_ = 0;
    }

    /** @brief Sets everything to zero and sets the total number of
     *         terms represented by this object to a given number.
     */
    void SetZero(int total_num_terms_in) {
      this->SetZero();
      total_num_terms_ = total_num_terms_in;
    }

    /** @brief Pushes another mean variance pair information. Assumes
     *         that the addition is done in an asymptotic normal way.
     */
    void CombineWith(const core::monte_carlo::MeanVariancePair &v) {

      // If the incoming mean variance pair is empty, then do not do
      // anything.
      if(v.total_num_terms() == 0) {
        return;
      }

      // Update the sample mean.
      sample_mean_ =
        (total_num_terms_ * sample_mean_ +
         v.total_num_terms() * v.sample_mean()) /
        static_cast<double>(total_num_terms_ + v.total_num_terms());

      // Update the sample variance.
      double first_portion =
        static_cast<double>(total_num_terms_) /
        static_cast<double>(total_num_terms_ + v.total_num_terms());
      double second_portion = 1.0 - first_portion;
      sample_variance_ =
        (num_samples_ + v.num_samples()) *
        (core::math::Sqr(first_portion) * this->sample_mean_variance() +
         core::math::Sqr(second_portion) * v.sample_mean_variance());

      // Finally update the total number of terms and the number of
      // samples.
      total_num_terms_ = total_num_terms_ + v.total_num_terms();
      num_samples_ = num_samples_ + v.num_samples();
    }

    /** @brief Pushes another mean variance pair information with a
     *         scale factor. Assumes that the addition is done in an
     *         asymptotic normal way.
     */
    void ScaledCombineWith(
      double scale_in, const core::monte_carlo::MeanVariancePair &v) {

      core::monte_carlo::MeanVariancePair scaled_v;
      scaled_v.CopyValues(v);
      scaled_v.scale(scale_in);
      this->CombineWith(scaled_v);
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

    /** @brief Removes the number from the mean variance. This
     *         decreases the number of terms and the samples at the
     *         same time.
     */
    void pop(double sample) {

      double new_sample_mean =
        (static_cast<double>(num_samples_) /
         static_cast<double>(num_samples_ - 1)) * sample_mean_ -
        sample / static_cast<double>(num_samples_ - 1);

      // Set the new sample variance.
      sample_variance_ =
        (num_samples_ * sample_variance_ -
         (sample - sample_mean_) * (sample - new_sample_mean)) /
        static_cast<double>(num_samples_ - 1);

      // Set the new sample mean.
      sample_mean_ = new_sample_mean;

      // Decrease the number of samples and the total number of terms
      // represented by this object.
      num_samples_--;
      total_num_terms_--;
    }
};
}
}

#endif
