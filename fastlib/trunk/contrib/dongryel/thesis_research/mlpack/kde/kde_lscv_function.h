#ifndef FL_LITE_MLPACK_KDE_KDE_LSCV_FUNCTION_H
#define FL_LITE_MLPACK_KDE_KDE_LSCV_FUNCTION_H

#include "fastlib/math/fl_math.h"
#include "fastlib/monte_carlo/multitree_monte_carlo_dev.h"
#include "mlpack/kde/kde.h"

namespace fl {
namespace ml {

template<typename TemplateArgs>
class KdeLscvFunction: public virtual
    fl::ml::MultitreeMonteCarlo<typename TemplateArgs::TableType> {

  public:

    typedef typename TemplateArgs::ComputationType::Result_t Result_t;

    typedef typename TemplateArgs::MetricType Metric_t;

  private:

    class KdeLscvOuterSum {

      public:
        typedef typename KdeLscvFunction<TemplateArgs>::Metric_t MetricType;

      private:

        fl::ml::Kde<TemplateArgs> *kde_instance_;

        fl::math::GaussianStarKernel<double> convolution_kernel_;

        const MetricType *metric_;

        double bandwidth_in_log_scale_;

      public:

        const MetricType *metric() {
          return metric_;
        }

        const fl::math::GaussianStarKernel<double> &convolution_kernel() const {
          return convolution_kernel_;
        }

        fl::ml::Kde<TemplateArgs> *kde_instance() {
          return kde_instance_;
        }

        void set_bandwidth_in_log_scale(double bandwidth_in_log_scale_in) {
          bandwidth_in_log_scale_ = bandwidth_in_log_scale_in;
          double bandwidth_in = exp(bandwidth_in_log_scale_in);
          convolution_kernel_.Init(
            bandwidth_in,
            kde_instance_->reference_table()->n_attributes());
          kde_instance_->global().set_bandwidth(bandwidth_in);
        }

        void Init(fl::ml::Kde<TemplateArgs> &kde_instance_in,
                  const MetricType &metric_in) {
          kde_instance_ = &kde_instance_in;

          // Set to the initial bandwidth.
          convolution_kernel_.Init(
            sqrt(kde_instance_->global().kernel().bandwidth_sq()),
            kde_instance_->reference_table()->n_attributes());
          bandwidth_in_log_scale_ = log(
                                      sqrt(convolution_kernel_.bandwidth_sq()));
          metric_ = &metric_in;
        }

        void Allocate(std::vector<double> *v) const {
          v->resize(3);
          (*v)[0] = 0;
          (*v)[1] = 0;
          (*v)[2] = 0;
        }

        void Allocate(std::vector< std::pair<double, double> > *v) const {
          v->resize(3);

          // The LSCV score.
          (*v)[0] = std::pair<double, double>(0.0, 0.0);
          (*v)[1] = std::pair<double, double>(0.0, 0.0);

          // The gradient with respect to the bandwidth in log scale.
          (*v)[2] = std::pair<double, double>(0.0, 0.0);
        }

        void Allocate(std::vector<fl::ml::MeanVariancePair> *v) const {
          v->resize(3);
          ((*v)[0]).SetZero();
          ((*v)[1]).SetZero();
          ((*v)[2]).SetZero();
        }

        void Summarize(
          const std::vector< std::pair<double, double> > &v,
          std::vector< std::pair<double, double> > *summary_mean_variance_pair) const {

          summary_mean_variance_pair->resize(3);

          // Merely copy over the LSCV score and the gradient.
          (*summary_mean_variance_pair)[0] = v[0];
          (*summary_mean_variance_pair)[1] = v[1];
          (*summary_mean_variance_pair)[2] = v[2];
        }

        template<typename MultitreeMonteCarloType>
        void Compute(std::vector<int> &chosen_variable_arguments,
                     MultitreeMonteCarloType &multitree_montecarlo,
                     std::vector<double> *set_of_results) const {

          typename TemplateArgs::TableType::Point_t first_point;
          typename TemplateArgs::TableType::Point_t second_point;
          multitree_montecarlo.get(0, chosen_variable_arguments[0], &first_point);
          multitree_montecarlo.get(1, chosen_variable_arguments[1], &second_point);

          double half_dimension = ((double)first_point.length()) / 2.0;
          double dimension = first_point.length();
          double distsq = metric_->DistanceSq(first_point, second_point);
          double neg_inv_bandwidth_2sq =
            - 1.0 / (2.0 * fl::math::Sqr(exp(bandwidth_in_log_scale_)));
          double factor = pow(2.0, -dimension / 2.0 - 1);
          double first_kernel_value =
            factor * exp(distsq * neg_inv_bandwidth_2sq * 0.5);
          double second_kernel_value = - exp(distsq * neg_inv_bandwidth_2sq);
          set_of_results->resize(0);

          // Push in the LSCV score.
          set_of_results->push_back(first_kernel_value);
          set_of_results->push_back(second_kernel_value);

          // Push in the (part of the) gradient.
          double part_gradient =
            dimension * exp(-0.5 * distsq * exp(-2 * bandwidth_in_log_scale_)) -
            pow(2, 1.0 - half_dimension) * dimension *
            exp(-0.25 * distsq * exp(-2 * bandwidth_in_log_scale_)) -
            distsq * exp(-0.5 * distsq * exp(-2 * bandwidth_in_log_scale_)) +
            pow(2, -half_dimension) * distsq *
            exp(-0.25 * distsq * exp(-2 * bandwidth_in_log_scale_) -
                2 * bandwidth_in_log_scale_);

          set_of_results->push_back(part_gradient);
        }
    };

    double NaiveScore_() {
      typename TemplateArgs::TableType *reference_table = outer_sum_.kde_instance()->global().reference_table();

      double naive_score = 0;
      for(int i = 0; i < reference_table->n_entries(); i++) {
        typename TemplateArgs::TableType::Point_t outer_point;
        reference_table->get(i, &outer_point);
        for(int j = 0; j < reference_table->n_entries(); j++) {
          typename TemplateArgs::TableType::Point_t inner_point;
          reference_table->get(j, &inner_point);

          double distsq = outer_sum_.metric()->DistanceSq(outer_point, inner_point);
          double kernel_value = outer_sum_.convolution_kernel().EvalUnnormOnSq(distsq);
          naive_score += kernel_value;
        }
      }

      double correction = 2.0 *
                          outer_sum_.kde_instance()->global().kernel().EvalUnnormOnSq(0.0) /
                          (outer_sum_.kde_instance()->global().kernel().CalcNormConstant(point_dimension_) *
                           ((double) num_points_));

      naive_score /= ((double) fl::math::Sqr(reference_table->n_entries()));
      naive_score *= outer_sum_.convolution_kernel().CalcMultiplicativeNormConstant(
                       reference_table->n_attributes());
      naive_score += correction;

      return naive_score;
    }

  private:

    int point_dimension_;

    int num_points_;

    KdeLscvOuterSum outer_sum_;

    Result_t *query_results_;

    std::vector< std::pair<double, double> > mean_variance_pairs_;

  public:

    double plugin_bandwidth() {
      double avg_sdev = 0;
      typename TemplateArgs::TableType *table =
        outer_sum_.kde_instance()->reference_table();
      fl::data::MonolithicPoint<double> mean_vector;
      mean_vector.Init(table->n_attributes());
      mean_vector.SetZero();

      // First compute the mean vector.
      for(index_t i = 0; i < table->n_entries(); i++) {
        typename TemplateArgs::TableType::Point_t point;
        table->get(i, &point);
        fl::la::AddTo(point, &mean_vector);
      }
      fl::la::SelfScale(1.0 / ((double) num_points_), &mean_vector);

      // Loop over the dataset again and compute variance along each
      // dimension.
      for(index_t j = 0; j < point_dimension_; j++) {
        double sdev = 0;
        for(index_t i = 0; i < num_points_; i++) {
          typename TemplateArgs::TableType::Point_t point;
          table->get(i, &point);
          sdev += math::Sqr(point[j] - mean_vector[j]);
        }
        sdev /= ((double) num_points_ - 1);
        sdev = sqrt(sdev);
        avg_sdev += sdev;
      }
      avg_sdev /= ((double) point_dimension_);

      double plugin_bw =
        pow((4.0 / (point_dimension_ + 2.0)), 1.0 / (point_dimension_ + 4.0)) * avg_sdev *
        pow(num_points_, -1.0 / (point_dimension_ + 4.0));

      return plugin_bw;
    }

    void Init(fl::ml::Kde<TemplateArgs> &kde_instance_in,
              const Metric_t &metric_in,
              Result_t &query_results_in) {

      // Set the arguments for the base class.
      fl::ml::MultitreeMonteCarlo<typename TemplateArgs::TableType>::add_variable_argument(
        *(kde_instance_in.reference_table()), std::vector<int>());
      fl::ml::MultitreeMonteCarlo<typename TemplateArgs::TableType>::add_variable_argument(
        *(kde_instance_in.reference_table()), std::vector<int>());

      // Set the error level.
      fl::ml::MultitreeMonteCarlo<typename TemplateArgs::TableType>::set_error(
        kde_instance_in.global().relative_error(), kde_instance_in.global().probability());

      // Set the arguments for the derived class.
      point_dimension_ =
        kde_instance_in.global().reference_table()->n_attributes();
      num_points_ = kde_instance_in.global().reference_table()->n_entries();
      outer_sum_.Init(kde_instance_in, metric_in);
      query_results_ = &query_results_in;
    }

    double Evaluate(const fl::data::MonolithicPoint<double> &x) {

      // Make sure the bandwidth is set before evaluating.
      outer_sum_.set_bandwidth_in_log_scale(x[0]);
      fl::ml::MultitreeMonteCarlo<typename TemplateArgs::TableType>::Compute(
        outer_sum_, &mean_variance_pairs_);

      double correction = 2.0 *
                          outer_sum_.kde_instance()->global().kernel().EvalUnnormOnSq(0.0) /
                          (outer_sum_.kde_instance()->global().kernel().CalcNormConstant(point_dimension_) *
                           ((double) num_points_));

      double lscv_score = (mean_variance_pairs_[0].first + mean_variance_pairs_[1].first) *
                          outer_sum_.convolution_kernel().CalcMultiplicativeNormConstant(
                            point_dimension_) + correction;

      fl::logger->Message() << "Bandwidth value of " << exp(x[0]) <<
                            " has the least squares "
                            "cross-validation score of " << lscv_score;
      return lscv_score;
    }

    void Gradient(const fl::data::MonolithicPoint<double> &x,
                  fl::data::MonolithicPoint<double> *gradient) {

      // Add the correction factor and return.
      double half_dimension = ((double) point_dimension_) / 2.0;
      double correction =
        (- pow(2, 1.0 - half_dimension))
        * point_dimension_ * pow(exp(2 * x[0]), -half_dimension) *
        pow(fl::math::template Const<double>::PI, -half_dimension) /
        ((double) num_points_);

      (*gradient)[0] = mean_variance_pairs_[2].first *
                       outer_sum_.convolution_kernel().CalcMultiplicativeNormConstant(
                         point_dimension_) + correction;
    }

    int num_dimensions() const {
      return 1;
    }
};
};
};

#endif
