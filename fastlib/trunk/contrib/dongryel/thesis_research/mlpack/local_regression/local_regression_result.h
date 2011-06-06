/** @file local_regression_result.h
 *
 *  The computed results in local regression in a dual-tree algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_RESULT_H
#define MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_RESULT_H

namespace mlpack {
namespace local_regression {

/** @brief Represents the storage of local regression computation
 *         results.
 */
class LocalRegressionResult {
  private:

    // For BOOST serialization.
    friend class boost::serialization::access;

  public:

    /** @brief The temporary space for extracting the left hand side.
     */
    arma::mat tmp_left_hand_side_;

    /** @brief The temporary space for extracting the right hand side.
     */
    arma::vec tmp_right_hand_side_;

    /** @brief The temporary space used for extracting the
     *         eigenvectors.
     */
    arma::mat tmp_eigenvectors_;

    /** @brief The temporary space used for extracting the
     *         eigenvalues.
     */
    arma::vec tmp_eigenvalues_;

    /** @brief The temporary space for storing the solution vector.
     */
    arma::vec tmp_solution_;

    /** @brief The number of query points.
     */
    int num_query_points_;

    /** @brief The final regression estimates.
     */
    boost::scoped_array<double> regression_estimates_;

    /** @brief The lower bound on the left hand side.
     */
    boost::scoped_array <
    core::monte_carlo::MeanVariancePairMatrix > left_hand_side_l_;

    /** @brief The estimated left hand side.
     */
    boost::scoped_array <
    core::monte_carlo::MeanVariancePairMatrix > left_hand_side_e_;

    /** @brief The upper bound on the left hand side.
     */
    boost::scoped_array <
    core::monte_carlo::MeanVariancePairMatrix > left_hand_side_u_;

    /** @brief The lower bound on the right hand side.
     */
    boost::scoped_array <
    core::monte_carlo::MeanVariancePairVector > right_hand_side_l_;

    /** @brief The estimated right hand side.
     */
    boost::scoped_array <
    core::monte_carlo::MeanVariancePairVector > right_hand_side_e_;

    /** @brief The upper bound on the left hand side.
     */
    boost::scoped_array <
    core::monte_carlo::MeanVariancePairVector > right_hand_side_u_;

    /** @brief The number of points pruned per each query.
     */
    boost::scoped_array<double> pruned_;

    /** @brief The amount of maximum error incurred per each query for
     *         the left hand side.
     */
    boost::scoped_array<double> left_hand_side_used_error_;

    /** @brief The amount of maximum error incurred per each query for
     *         the right hand side.
     */
    boost::scoped_array<double> right_hand_side_used_error_;

    /** @brief Saves the local regression result object.
     */
    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {
      ar & num_query_points_;
      for(unsigned int i = 0; i < num_query_points_; i++) {
        ar & regression_estimates_[i];
        ar & left_hand_side_l_[i];
        ar & left_hand_side_e_[i];
        ar & left_hand_side_u_[i];
        ar & right_hand_side_l_[i];
        ar & right_hand_side_e_[i];
        ar & right_hand_side_u_[i];
        ar & pruned_[i];
        ar & left_hand_side_used_error_[i];
        ar & right_hand_side_used_error_[i];
      }
    }

    /** @brief Loads the local regression result object.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the number of points.
      ar & num_query_points_;

      // Initialize the array.
      this->Init(num_query_points_);

      // Load.
      for(int i = 0; i < num_query_points_; i++) {
        ar & regression_estimates_[i];
        ar & left_hand_side_l_[i];
        ar & left_hand_side_e_[i];
        ar & left_hand_side_u_[i];
        ar & right_hand_side_l_[i];
        ar & right_hand_side_e_[i];
        ar & right_hand_side_u_[i];
        ar & pruned_[i];
        ar & left_hand_side_used_error_[i];
        ar & right_hand_side_used_error_[i];
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    void Seed(int qpoint_index, double initial_pruned_in) {
      pruned_[qpoint_index] = initial_pruned_in;
    }

    /** @brief The default constructor.
     */
    LocalRegressionResult() {
      num_query_points_ = 0;
      SetZero();
    }

    template<typename MetricType, typename GlobalType>
    void PostProcess(
      const MetricType &metric,
      const core::table::DensePoint &qpoint,
      int q_index,
      double q_weight,
      const GlobalType &global,
      const bool is_monochromatic) {

      // Set up the linear system.
      left_hand_side_e_[q_index].sample_means(&tmp_left_hand_side_);
      right_hand_side_e_[q_index].sample_means(&tmp_right_hand_side_);

      if(global.problem_dimension() == 1) {
        regression_estimates_[q_index] = tmp_right_hand_side_[0] /
                                         tmp_left_hand_side_.at(0, 0);
      }
      else {

        // Solve the linear system.
        arma::eig_sym(tmp_eigenvalues_, tmp_eigenvectors_, tmp_left_hand_side_);
        tmp_solution_.zeros(tmp_left_hand_side_.n_rows);
        for(unsigned int i = 0; i < tmp_eigenvalues_.n_elem; i++) {
          double dot_product = arma::dot(
                                 tmp_eigenvectors_.col(i),
                                 tmp_right_hand_side_);
          if(tmp_eigenvalues_[i] > 1e-6) {
            tmp_solution_ +=
              (dot_product / tmp_eigenvalues_[i]) * tmp_eigenvectors_.col(i);
          }
        }

        // Take the dot product with the solution vector to get the
        // regression estimate.
        regression_estimates_[q_index] = tmp_solution_[0];
        for(unsigned int i = 1; i < tmp_solution_.n_elem; i++) {
          regression_estimates_[q_index] += tmp_solution_[i] *
                                            qpoint[i - 1];
        }
      }
    }


    void Print(const std::string &file_name) const {
      FILE *file_output = fopen(file_name.c_str(), "w+");
      for(int i = 0; i < num_query_points_; i++) {
        fprintf(file_output, "%g %g\n", regression_estimates_[i], pruned_[i]);
      }
      fclose(file_output);
    }

    /** @brief Basic allocation for local regression results.
     */
    void Init(int num_points) {

      // Sets the number of points.
      num_query_points_ = num_points;

      // Initialize the array storing the final regression estimates.
      boost::scoped_array<double> tmp_regression_estimates(
        new double[num_query_points_]);
      regression_estimates_.swap(tmp_regression_estimates);

      // Initialize the left hand side lower bound.
      boost::scoped_array <
      core::monte_carlo::MeanVariancePairMatrix >
      tmp_left_hand_side_l(
        new core::monte_carlo::MeanVariancePairMatrix[num_query_points_]);
      left_hand_side_l_.swap(tmp_left_hand_side_l);

      // Initialize the left hand side estimate.
      boost::scoped_array <
      core::monte_carlo::MeanVariancePairMatrix >
      tmp_left_hand_side_e(
        new core::monte_carlo::MeanVariancePairMatrix[num_query_points_]);
      left_hand_side_e_.swap(tmp_left_hand_side_e);

      // Initialize the left hand side upper bound.
      boost::scoped_array <
      core::monte_carlo::MeanVariancePairMatrix >
      tmp_left_hand_side_u(
        new core::monte_carlo::MeanVariancePairMatrix[num_query_points_]);
      left_hand_side_u_.swap(tmp_left_hand_side_u);

      // Initialize the right hand side lower bound.
      boost::scoped_array <
      core::monte_carlo::MeanVariancePairVector >
      tmp_right_hand_side_l(
        new core::monte_carlo::MeanVariancePairVector[num_query_points_]);
      right_hand_side_l_.swap(tmp_right_hand_side_l);

      // Initialize the right hand side estimate.
      boost::scoped_array <
      core::monte_carlo::MeanVariancePairVector >
      tmp_right_hand_side_e(
        new core::monte_carlo::MeanVariancePairVector[num_query_points_]);
      right_hand_side_e_.swap(tmp_right_hand_side_e);

      // Initialize the right hand side upper bound.
      boost::scoped_array <
      core::monte_carlo::MeanVariancePairVector >
      tmp_right_hand_side_u(
        new core::monte_carlo::MeanVariancePairVector[num_query_points_]);
      right_hand_side_u_.swap(tmp_right_hand_side_u);

      // Initialize the pruned quantities.
      boost::scoped_array< double > tmp_pruned(
        new double[num_query_points_]);
      pruned_.swap(tmp_pruned);

      // Initialize the used error quantities.
      boost::scoped_array< double > tmp_left_hand_side_used_error(
        new double[num_query_points_]);
      left_hand_side_used_error_.swap(tmp_left_hand_side_used_error);
      boost::scoped_array< double > tmp_right_hand_side_used_error(
        new double[num_query_points_]);
      right_hand_side_used_error_.swap(tmp_right_hand_side_used_error);
    }

    template<typename GlobalType>
    void Init(const GlobalType &global_in, int num_points) {

      // Basic allocation.
      this->Init(num_points);

      // Allocate Monte Carlo results.
      for(int i = 0; i < num_query_points_; i++) {
        left_hand_side_l_[i].Init(
          global_in.problem_dimension() ,
          global_in.problem_dimension());
        left_hand_side_e_[i].Init(
          global_in.problem_dimension() ,
          global_in.problem_dimension());
        left_hand_side_u_[i].Init(
          global_in.problem_dimension() ,
          global_in.problem_dimension());
        right_hand_side_l_[i].Init(global_in.problem_dimension());
        right_hand_side_e_[i].Init(global_in.problem_dimension());
        right_hand_side_u_[i].Init(global_in.problem_dimension());
      }

      // Set everything to zero.
      SetZero();
    }

    void SetZero() {
      for(int i = 0; i < num_query_points_; i++) {
        left_hand_side_l_[i].SetZero();
        left_hand_side_e_[i].SetZero();
        left_hand_side_u_[i].SetZero();
        right_hand_side_l_[i].SetZero();
        right_hand_side_e_[i].SetZero();
        right_hand_side_u_[i].SetZero();
        pruned_[i] = 0.0;
        left_hand_side_used_error_[i] = 0.0;
        right_hand_side_used_error_[i] = 0.0;
      }
    }

    template<typename GlobalType, typename TreeType, typename DeltaType>
    void ApplyProbabilisticDelta(
      GlobalType &global, TreeType *qnode, double failure_probability,
      const DeltaType &delta_in) {

      // Get the iterator for the query node.
      typename GlobalType::TableType::TreeIterator qnode_it =
        global.query_table()->get_node_iterator(qnode);
      int qpoint_index;

      do {

        // Get each query point.
        qnode_it.Next(&qpoint_index);

        // Accumulate the delta contributions for each component.
        for(int j = 0 ; j < global.problem_dimension() ; j++) {
          const core::monte_carlo::MeanVariancePair
          &right_hand_side_contribution_l =
            (*delta_in.query_deltas_)[qpoint_index].right_hand_side_l_[j];
          const core::monte_carlo::MeanVariancePair
          &right_hand_side_contribution_e =
            (*delta_in.query_deltas_)[qpoint_index].right_hand_side_e_[j];
          const core::monte_carlo::MeanVariancePair
          &right_hand_side_contribution_u =
            (*delta_in.query_deltas_)[qpoint_index].right_hand_side_u_[j];

          // Combine with the query result right hand side.
          right_hand_side_l_[
            qpoint_index][j].CombineWith(right_hand_side_contribution_l[j]);
          right_hand_side_e_[
            qpoint_index][j].CombineWith(right_hand_side_contribution_e[j]);
          right_hand_side_u_[
            qpoint_index][j].CombineWith(right_hand_side_contribution_u[j]);

          for(int i = 0 ; i < global.problem_dimension() ; i++) {
            const core::monte_carlo::MeanVariancePair
            &left_hand_side_contribution_l =
              (*delta_in.query_deltas_)[
                qpoint_index].left_hand_side_l_.get(i, j);
            const core::monte_carlo::MeanVariancePair
            &left_hand_side_contribution_e =
              (*delta_in.query_deltas_)[
                qpoint_index].left_hand_side_e_.get(i, j);
            const core::monte_carlo::MeanVariancePair
            &left_hand_side_contribution_u =
              (*delta_in.query_deltas_)[
                qpoint_index].left_hand_side_u_.get(i, j);

            // Combine with the left hand side for the query result.
            left_hand_side_l_[
              qpoint_index].get(i, j).CombineWith(
                left_hand_side_contribution_l.get(i, j));
            left_hand_side_e_[
              qpoint_index].get(i, j).CombineWith(
                left_hand_side_contribution_e);
            left_hand_side_u_[
              qpoint_index].get(i, j).CombineWith(
                left_hand_side_contribution_u);

          } // end of looping over each row.
        } // end of looping over each column.

        // Add in the incurred error quantities.
        left_hand_side_used_error_[qpoint_index] +=
          (*delta_in.query_deltas_)[qpoint_index].left_hand_side_used_error_;
        right_hand_side_used_error_[qpoint_index] +=
          (*delta_in.query_deltas_)[qpoint_index].right_hand_side_used_error_;

        // Add in the pruned quantities.
        pruned_[qpoint_index] += delta_in.pruned_;
      }
      while(qnode_it.HasNext());
    }

    /** @brief Apply postponed contributions.
     */
    template<typename LocalRegressionPostponedType>
    void ApplyPostponed(
      int q_index, const LocalRegressionPostponedType &postponed_in) {
      left_hand_side_l_[q_index].CombineWith(postponed_in.left_hand_side_l_);
      left_hand_side_e_[q_index].CombineWith(postponed_in.left_hand_side_e_);
      left_hand_side_u_[q_index].CombineWith(postponed_in.left_hand_side_u_);
      right_hand_side_l_[q_index].CombineWith(postponed_in.right_hand_side_l_);
      right_hand_side_e_[q_index].CombineWith(postponed_in.right_hand_side_e_);
      right_hand_side_u_[q_index].CombineWith(postponed_in.right_hand_side_u_);
      pruned_[q_index] = pruned_[q_index] + postponed_in.pruned_;
      left_hand_side_used_error_[q_index] =
        left_hand_side_used_error_[q_index] +
        postponed_in.left_hand_side_used_error_;
      right_hand_side_used_error_[q_index] =
        right_hand_side_used_error_[q_index] +
        postponed_in.right_hand_side_used_error_;
    }

    /** @brief Apply the postponed quantities to the query results
     *         during the final postprocessing stage.
     */
    template<typename GlobalType, typename LocalRegressionPostponedType>
    void FinalApplyPostponed(
      const GlobalType &global,
      const core::table::DensePoint &qpoint,
      int q_index,
      const LocalRegressionPostponedType &postponed_in) {

      // Apply postponed.
      ApplyPostponed(q_index, postponed_in);
    }
};

}
}

#endif
