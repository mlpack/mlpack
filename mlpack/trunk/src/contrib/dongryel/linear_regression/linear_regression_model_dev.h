#ifndef FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_MODEL_DEV_H
#define FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_MODEL_DEV_H

// #include "boost/math/distributions/students_t.hpp"
#include "linear_regression_model.h"

std::deque<int> &LinearRegressionModel::active_column_indices() {
  return factorization_.active_column_indices();
}


std::deque<int> &LinearRegressionModel::inactive_column_indices() {
  return factorization_.inactive_column_indices();
}


double LinearRegressionModel::aic_score() const {
  return aic_score_;
}


const Matrix *LinearRegressionModel::table() const {
  return table_;
}

/*
const std::string &LinearRegressionModel::column_name(
  int column_index) const {

  const std::vector<std::string> &features = table_->data()->labels();
  return features[ column_index ];
}
*/


void LinearRegressionModel::Init(
  const Matrix &table_in,
  const std::deque<int> &initial_active_column_indices,
  int initial_right_hand_side_index,
  bool include_bias_term_in,
  double conf_prob_in) {

  // Set the confidence level.
  conf_prob_ = conf_prob_in;

  // Set the table pointer.
  table_ = &table_in;

  // Initialize the QR factor.
  factorization_.Init(table_in, initial_active_column_indices,
                      initial_right_hand_side_index,
                      include_bias_term_in);

  // Initialize the coefficients.
  int num_coefficients = table_in.n_rows();
  if (include_bias_term_in) {
    num_coefficients++;
  }

  // Allocate space and initialize bunches of stuffs.
  coefficients_.Init(num_coefficients);
  coefficients_.SetZero();
  standard_errors_.Init(num_coefficients);
  standard_errors_.SetZero();
  confidence_interval_los_.Init(num_coefficients);
  confidence_interval_los_.SetZero();
  confidence_interval_his_.Init(num_coefficients);
  confidence_interval_his_.SetZero();
  t_statistics_.Init(num_coefficients);
  t_statistics_.SetZero();
  p_values_.Init(num_coefficients);
  p_values_.SetZero();

  adjusted_r_squared_ = 0;
  f_statistic_ = 0;
  r_squared_ = 0;
  sigma_ = 0;
  aic_score_ = 0;
}


int LinearRegressionModel::active_right_hand_side_column_index()
const {
  return factorization_.active_right_hand_side_column_index();
}


void LinearRegressionModel::set_active_right_hand_side_column_index(
  int right_hand_side_column_index_in) {

  // Set the right hand side column index, which will trigger the
  // QR factor update automatically.
  factorization_.set_active_right_hand_side_column_index(
    right_hand_side_column_index_in);
}


template<typename ResultType>
void LinearRegressionModel::Predict(const Matrix &query_table,
                                    ResultType *result_out) const {

  // Allocate space for the result.
  int prediction_index = factorization_.active_right_hand_side_column_index();
  const std::deque<int> &active_column_indices =
    factorization_.active_column_indices();
  result_out->Init(query_table);

  // Initialize the residual sum of squares.
  result_out->residual_sum_of_squares() = 0.0;

  for (int i = 0; i < query_table.n_cols(); i++) {

    Vector point;
    query_table.MakeColumnVector(i, &point);
    result_out->predictions()[i] = 0.0;
    int j = 0;
    for (std::deque<int>::const_iterator it = active_column_indices.begin();
         it != active_column_indices.end(); it++, j++) {
      int column_index = *it;
      if (column_index < point.length()) {
        result_out->predictions()[i] += coefficients_[j] *
                                        point[column_index];
      }
      else {
        result_out->predictions()[i] += coefficients_[j];
      }
    }

    // If it is leave-one-out, then we can compute the residual sum of
    // squares.
    if ((&query_table) == table_) {
      result_out->residual_sum_of_squares() +=
        math::Sqr(result_out->predictions()[i] - point[prediction_index]);
    }
  }
}


void LinearRegressionModel::MakeColumnActive(int column_index) {
  factorization_.MakeColumnActive(column_index);
}


void LinearRegressionModel::MakeColumnInactive(int column_index) {
  factorization_.MakeColumnInactive(column_index);
}


void LinearRegressionModel::Solve() {
  factorization_.Solve((Vector *) NULL, &coefficients_);
}


double LinearRegressionModel::CorrelationCoefficient(
  int first_attribute_index,
  int second_attribute_index) const {

  // Compute the average of each attribute index.
  double first_attribute_average = 0;
  double second_attribute_average = 0;
  for (int i = 0; i < table_->n_cols(); i++) {
    Vector point;
    table_->MakeColumnVector(i, &point);
    first_attribute_average += point[first_attribute_index];
    second_attribute_average += point[second_attribute_index];
  }
  first_attribute_average /= ((double) table_->n_cols());
  second_attribute_average /= ((double) table_->n_cols());

  // Compute the variance of each attribute index.
  double first_attribute_variance = 0;
  double second_attribute_variance = 0;
  double covariance = 0;
  for (int i = 0; i < table_->n_cols(); i++) {
    Vector point;
    table_->MakeColumnVector(i, &point);
    covariance += (point[first_attribute_index] -
                   first_attribute_average) *
                  (point[second_attribute_index] -
                   second_attribute_average);
    first_attribute_variance +=
      math::Sqr(point[first_attribute_index] -
                first_attribute_average);
    second_attribute_variance +=
      math::Sqr(point[second_attribute_index] -
                second_attribute_average);
  }
  covariance /= ((double) table_->n_cols());
  first_attribute_variance /= ((double) table_->n_cols() - 1);
  second_attribute_variance /= ((double) table_->n_cols() - 1);

  return covariance / sqrt(first_attribute_variance *
                           second_attribute_variance);
}


double LinearRegressionModel::SquaredCorrelationCoefficient(
  const LinearRegressionResult &result) const {

  int dimension = active_right_hand_side_column_index();

  // Compute the average of the observed values.
  double avg_observed_value = 0;

  for (int i = 0; i < table_->n_cols(); i++) {
    Vector point;
    table_->MakeColumnVector(i, &point);
    avg_observed_value += point[dimension];
  }
  avg_observed_value /= ((double) table_->n_cols());

  // Compute something proportional to the variance of the observed
  // values, and the sum of squared residuals of the predictions
  // against the observations.
  double variance = 0;
  for (int i = 0; i < table_->n_cols(); i++) {
    Vector point;
    table_->MakeColumnVector(i, &point);
    variance += math::Sqr(point[dimension] - avg_observed_value);
  }
  return (variance - result.residual_sum_of_squares()) / variance;
}


double LinearRegressionModel::VarianceInflationFactor(
  const LinearRegressionResult &result) const {

  double denominator = 1.0 - SquaredCorrelationCoefficient(result);

  if (!isnan(denominator)) {
    if (fabs(denominator) >= 1e-2) {
      return 1.0 / denominator;
    }
    else {
      return 100;
    }
  }
  else {
    return 0.0;
  }
}


double LinearRegressionModel::FStatistic() {
  double numerator = r_squared_ /
                     ((double) factorization_.active_column_indices().size() - 1);
  double denominator = (1.0 - r_squared_) /
                       ((double) table_->n_cols() -
                        factorization_.active_column_indices().size());
  return numerator / denominator;
}


double LinearRegressionModel::
AdjustedSquaredCorrelationCoefficient() {
  int num_points = table_->n_cols();
  int num_coefficients = factorization_.active_column_indices().size();
  double factor = (((double) num_points - 1)) /
                  ((double)(num_points - num_coefficients));
  return 1.0 - (1.0 - r_squared_) * factor;
}

/*

void LinearRegressionModel::ComputeModelStatistics(
  const LinearRegressionResult &result) {

  // Declare the student t-distribution and find out the
  // appropriate quantile for the confidence interval
  // (currently hardcoded to 90 % centered confidence).
  boost::math::students_t_distribution<double> distribution(
    factorization_.active_column_indices().size());

  double t_score = quantile(distribution, 0.5 + 0.5 * conf_prob_);

  double variance = result.residual_sum_of_squares() /
                    (table_->n_cols() -
                     factorization_.active_column_indices().size());

  // Store the computed standard deviation of the predictions.
  sigma_ = sqrt(variance);

  fl::data::MonolithicPoint<double> dummy_vector;
  dummy_vector.Init(factorization_.active_column_indices().size());
  dummy_vector.SetZero();
  fl::data::MonolithicPoint<double> first_vector;
  first_vector.Init(factorization_.n_rows());
  first_vector.SetZero();
  fl::data::MonolithicPoint<double> second_vector;
  second_vector.Init(factorization_.active_column_indices().size());
  second_vector.SetZero();

  for (int i = 0; i < factorization_.active_column_indices().size(); i++) {
    dummy_vector[i] = 1.0;
    if (i > 0) {
      dummy_vector[i - 1] = 0.0;
    }
    factorization_.TransposeSolve(dummy_vector, &first_vector);
    factorization_.Solve(&first_vector, &second_vector);
    standard_errors_[i] = sqrt(variance * second_vector[i]);
    confidence_interval_los_[i] =
      coefficients_[i] - t_score * standard_errors_[i];
    confidence_interval_his_[i] =
      coefficients_[i] + t_score * standard_errors_[i];

    // Compute t-statistics.
    t_statistics_[i] = coefficients_[i] / standard_errors_[i];

    // Compute p-values.
    // Here we take the absolute value of the t-statistics since
    // we want to push all p-values toward the right end.
    double min_t_statistic = std::min(t_statistics_[i],
                                      -(t_statistics_[i]));
    double max_t_statistic = std::max(t_statistics_[i],
                                      -(t_statistics_[i]));
    p_values_[i] =
      1.0 - (cdf(distribution, max_t_statistic) -
             cdf(distribution, min_t_statistic));
  }

  // Compute the r-squared coefficients (normal and adjusted).
  r_squared_ = SquaredCorrelationCoefficient(result);
  adjusted_r_squared_ = AdjustedSquaredCorrelationCoefficient();

  // Compute the f-statistic between the final refined model and
  // the null model, i.e. the model with all zero coefficients.
  f_statistic_ = FStatistic();

  // Compute the AIC score.
  aic_score_ = ComputeAICScore(result);
}
*/


double LinearRegressionModel::ComputeAICScore(
  const LinearRegressionResult &result) {

  // Compute the squared errors from the predictions.
  double aic_score = result.residual_sum_of_squares();
  aic_score /= ((double) table_->n_cols());
  aic_score = log(aic_score);
  aic_score *= ((double) table_->n_cols());
  aic_score += (2 * factorization_.active_column_indices().size());
  return aic_score;
}


/*
void LinearRegressionModel::ExportHelper_(
  const Vector &source,
  Vector &destination) const {

  for (int i = 0; i < destination.length(); i++) {
    destination.set(i, 0.0);
  }

  // Put the bias term at the position of the prediction index, if
  // available. Remember that the last coefficient is for the bias
  // term, if available.
  if (table_->n_rows() + 1 == source.length()) {
    destination.set(
      factorization_.active_right_hand_side_column_index(),
      source[ factorization_.active_column_indices().size() - 1]);
  }

  // Copy the rest.
  int j = 0;
  for (std::deque<int>::const_iterator active_it =
         factorization_.active_column_indices().begin();
       active_it != factorization_.active_column_indices().end();
       active_it++, j++) {

    // Skip the bias term.
    if (*active_it < table_->n_rows()) {
      destination.set(*active_it, source[j]);
    }
  }
}
*/

void LinearRegressionModel::ClearInactiveColumns() {
  factorization_.ClearInactiveColumns();
}

/*
void LinearRegressionModel::Export(
  Matrix &coefficients_table,
  Matrix &standard_errors_table,
  Matrix &confidence_interval_los_table,
  Matrix &confidence_interval_his_table,
  Matrix &t_statistics_table,
  Matrix &p_values_table,
  Matrix &adjusted_r_squared_table,
  Matrix &f_statistic_table,
  Matrix &r_squared_table,
  Matrix &sigma_table) const {

  // Create the table for dumping the coefficients.
  Vector point;
  coefficients_table.Init(std::vector<int>(1, coefficients_.length()),
                          std::vector<int>(), 1);
  coefficients_table.get(0, &point);
  ExportHelper_(coefficients_, point);

  // Create the table for dumping the standard errors.
  standard_errors_table.Init(std::vector<int>(1, coefficients_.length()),
                             std::vector<int>(), 1);
  standard_errors_table.get(0, &point);
  ExportHelper_(standard_errors_, point);

  // Create the tables for dumping the confidence intervals.
  confidence_interval_los_table.Init(
    std::vector<int>(1, coefficients_.length()), std::vector<int>(), 1);
  confidence_interval_his_table.Init(
    std::vector<int>(1, coefficients_.length()), std::vector<int>(), 1);
  confidence_interval_los_table.get(0, &point);
  ExportHelper_(confidence_interval_los_, point);
  confidence_interval_his_table.get(0, &point);
  ExportHelper_(confidence_interval_his_, point);

  // Create the table for dumping the t-statistic values.
  t_statistics_table.Init(
    std::vector<int>(1, coefficients_.length()), std::vector<int>(), 1);
  t_statistics_table.get(0, &point);
  ExportHelper_(t_statistics_, point);

  // Create the table for p-values.
  p_values_table.Init(
    std::vector<int>(1, coefficients_.length()), std::vector<int>(), 1);
  p_values_table.get(0, &point);
  ExportHelper_(p_values_, point);

  // Create the table for adjusted r-square statistic.
  adjusted_r_squared_table.Init(std::vector<int>(1, 1), std::vector<int>(), 1);
  adjusted_r_squared_table.get(0, &point);
  point.set(0, adjusted_r_squared_);
  f_statistic_table.Init(std::vector<int>(1, 1), std::vector<int>(), 1);
  f_statistic_table.get(0, &point);
  point.set(0, f_statistic_);
  r_squared_table.Init(std::vector<int>(1, 1), std::vector<int>(), 1);
  r_squared_table.get(0, &point);
  point.set(0, r_squared_);
  sigma_table.Init(std::vector<int>(1, 1), std::vector<int>(), 1);
  sigma_table.get(0, &point);
  point.set(0, sigma_);
}
*/

#endif
