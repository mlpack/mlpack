/** @file local_regression_statistic.h
 *
 *  The statistics computed from the data in local regression
 *  dual-tree algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_STATISTIC_H
#define MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_STATISTIC_H

namespace mlpack {
namespace local_regression {

class LocalRegressionStatistic {

  private:

    // For Boost serialization.
    friend class boost::serialization::access;

  private:

    template<typename GlobalType>
    void InitCommon_(const GlobalType &global) {

      // Initialize the postponed and the summary.
      postponed_.Init(global);
      summary_.Init(global);

      // Initialize the average information.
      average_info_.Init(
        global.problem_dimension(), global.problem_dimension());
      weighted_average_info_.Init(global.problem_dimension());

      // Initialize the min/max statistics.
      min_average_info_.set_size(
        global.problem_dimension(), global.problem_dimension());
      min_average_info_.fill(std::numeric_limits<double>::max());
      max_average_info_.set_size(
        global.problem_dimension(), global.problem_dimension());
      max_average_info_.fill(- std::numeric_limits<double>::max());
      min_weighted_average_info_.set_size(global.problem_dimension());
      min_weighted_average_info_.fill(std::numeric_limits<double>::max());
      max_weighted_average_info_.set_size(global.problem_dimension());
      max_weighted_average_info_.fill(- std::numeric_limits<double>::max());
    }

  public:

    core::monte_carlo::MeanVariancePairMatrix average_info_;

    core::monte_carlo::MeanVariancePairVector weighted_average_info_;

    arma::mat min_average_info_;

    arma::mat max_average_info_;

    arma::vec min_weighted_average_info_;

    arma::vec max_weighted_average_info_;

    mlpack::local_regression::LocalRegressionPostponed postponed_;

    mlpack::local_regression::LocalRegressionSummary summary_;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & average_info_;
      ar & weighted_average_info_;
      ar & min_average_info_;
      ar & max_average_info_;
      ar & min_weighted_average_info_;
      ar & max_weighted_average_info_;
      ar & postponed_;
      ar & summary_;
    }

    /** @brief Copies another local regression statistic.
     */
    void Copy(const LocalRegressionStatistic &stat_in) {
      postponed_.Copy(stat_in.postponed_);
      summary_.Copy(stat_in.summary_);
    }

    /** @brief The default constructor.
     */
    LocalRegressionStatistic() {
      SetZero();
    }

    /** @brief Sets the postponed and the summary statistics to zero.
     */
    void SetZero() {
      postponed_.SetZero();
      summary_.SetZero();
    }

    void Seed(double initial_pruned_in) {
      postponed_.SetZero();
      summary_.Seed(initial_pruned_in);
    }

    /** @brief Initializes by taking statistics on raw data.
     */
    template<typename GlobalType, typename TreeType>
    void Init(const GlobalType &global, TreeType *node) {

      // Initialize.
      this->InitCommon_(global);

      // Set the number of terms for the average information.
      average_info_.set_total_num_terms(node->count());
      weighted_average_info_.set_total_num_terms(node->count());

      // Accumulate from the raw data.
      typename GlobalType::TableType::TreeIterator node_it =
        const_cast<GlobalType &>(global).
        reference_table()->get_node_iterator(node);
      while(node_it.HasNext()) {

        // Point ID and its weight.
        int point_id;
        double point_weight;

        // Get each point.
        arma::vec point;
        node_it.Next(&point, &point_id, &point_weight);

        // Push the contribution of each point.
        average_info_.get(0, 0).push_back(1.0);
        min_average_info_.at(0, 0) = std::min(min_average_info_.at(0, 0), 1.0);
        max_average_info_.at(0, 0) = std::max(max_average_info_.at(0, 0), 1.0);

        weighted_average_info_[0].push_back(point_weight);
        min_weighted_average_info_[0] =
          std::min(min_weighted_average_info_[0], point_weight);
        max_weighted_average_info_[0] =
          std::max(max_weighted_average_info_[0], point_weight);
        for(int j = 1; j < average_info_.n_cols(); j++) {
          average_info_.get(0, j).push_back(point[j - 1]);
          min_average_info_.at(0, j) =
            std::min(min_average_info_.at(0, j), point[j - 1]);
          max_average_info_.at(0, j) =
            std::max(max_average_info_.at(0, j), point[j - 1]);

          average_info_.get(j, 0).push_back(point[j - 1]);
          min_average_info_.at(j, 0) =
            std::min(min_average_info_.at(j, 0), point[j - 1]);
          max_average_info_.at(j, 0) =
            std::max(max_average_info_.at(j, 0), point[j - 1]);

          weighted_average_info_[j].push_back(point_weight * point[j - 1]);
          min_weighted_average_info_[j] =
            std::min(
              min_weighted_average_info_[j], point_weight * point[j - 1]);
          max_weighted_average_info_[j] =
            std::max(
              max_weighted_average_info_[j], point_weight * point[j - 1]);
          for(int i = 1; i < average_info_.n_rows(); i++) {
            average_info_.get(i, j).push_back(point[i - 1] * point[j - 1]);
            min_average_info_.at(i, j) =
              std::min(
                min_average_info_.at(i, j), point[i - 1] * point[j - 1]);
            max_average_info_.at(i, j) =
              std::max(
                max_average_info_.at(i, j), point[i - 1] * point[j - 1]);
          }
        }
      }

      // Sets the postponed quantities and summary statistics to zero.
      SetZero();
    }

    /** @brief Initializes by combining statistics of two partitions.
     *
     * This lets you build fast bottom-up statistics when building trees.
     */
    template<typename GlobalType, typename TreeType>
    void Init(
      const GlobalType &global,
      TreeType *node,
      const LocalRegressionStatistic &left_stat,
      const LocalRegressionStatistic &right_stat) {

      // Initialize first.
      this->InitCommon_(global);

      // Form the average information by combining from the children
      // information.
      average_info_.CombineWith(left_stat.average_info_);
      average_info_.CombineWith(right_stat.average_info_);
      weighted_average_info_.CombineWith(left_stat.weighted_average_info_);
      weighted_average_info_.CombineWith(right_stat.weighted_average_info_);

      // Form the min/max from the children.
      min_average_info_ = left_stat.min_average_info_;
      max_average_info_ = left_stat.max_average_info_;
      min_weighted_average_info_ = left_stat.min_weighted_average_info_;
      max_weighted_average_info_ = left_stat.max_weighted_average_info_;
      for(unsigned int j = 0; j < min_average_info_.n_cols; j++) {
        min_weighted_average_info_[j] =
          std::min(
            min_weighted_average_info_[j],
            right_stat.min_weighted_average_info_[j]);
        max_weighted_average_info_[j] =
          std::max(
            max_weighted_average_info_[j],
            right_stat.max_weighted_average_info_[j]);
        for(unsigned int i = 0; i < min_average_info_.n_rows; i++) {
          min_average_info_.at(i, j) =
            std::min(
              min_average_info_.at(i, j),
              right_stat.min_average_info_.at(i, j));
          max_average_info_.at(i, j) =
            std::max(
              max_average_info_.at(i, j),
              right_stat.max_average_info_.at(i, j));
        }
      }

      // Sets the postponed quantities and summary statistics to zero.
      SetZero();
    }
};
}
}

#endif
