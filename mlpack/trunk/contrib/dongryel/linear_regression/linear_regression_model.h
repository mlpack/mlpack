#ifndef FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_MODEL_H
#define FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_MODEL_H

#include "qr_least_squares_dev.h"
#include "linear_regression_result_dev.h"

class LinearRegressionModel {

  private:

    double conf_prob_;

    const Matrix *table_;

    QRLeastSquares factorization_;

    Vector coefficients_;

    Vector standard_errors_;

    Vector confidence_interval_los_;

    Vector confidence_interval_his_;

    Vector t_statistics_;

    Vector p_values_;

    double adjusted_r_squared_;

    double f_statistic_;

    double r_squared_;

    double sigma_;

    double aic_score_;

  private:

    void ExportHelper_(
      const Vector &source, Vector &destination) const;

  public:

    double aic_score() const;

    const Matrix *table() const;


    //    const std::string &column_name(int column_index) const;

    void Init(const Matrix &table_in,
              const std::deque<int> &initial_active_column_indices,
              int initial_right_hand_side_index,
              bool include_bias_term_in, double conf_prob_in);

    std::deque<int> &active_column_indices();

    std::deque<int> &inactive_column_indices();

    int active_right_hand_side_column_index() const;

    void set_active_right_hand_side_column_index(int right_hand_side_index_in);

    template<typename ResultType>
    void Predict(const Matrix &query_table,
                 ResultType *result_out) const;

    void MakeColumnActive(int column_index);

    void MakeColumnInactive(int column_index);

    void Solve();

    double FStatistic();

    double AdjustedSquaredCorrelationCoefficient();

    void ComputeModelStatistics(
      const LinearRegressionResult &result);

    double SquaredCorrelationCoefficient(
      const LinearRegressionResult &result) const;

    double VarianceInflationFactor(
      const LinearRegressionResult &result) const;

    void Export(Matrix &coefficients_table,
                Matrix &standard_errors_table,
                Matrix &confidence_interval_los_table,
                Matrix &confidence_interval_his_table,
                Matrix &t_statistics_table,
                Matrix &p_values_table,
                Matrix &adjusted_r_squared_table,
                Matrix &f_statistic_table,
                Matrix &r_squared_table,
                Matrix &sigma_table) const;

    double ComputeAICScore(const LinearRegressionResult &result);

    double CorrelationCoefficient(int first_attribute_index,
                                  int second_attribute_index) const;

    void ClearInactiveColumns();
};

#endif
