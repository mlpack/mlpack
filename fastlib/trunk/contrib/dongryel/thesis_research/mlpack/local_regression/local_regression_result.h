/** @file local_regression_result.h
 *
 *  The computed results in local regression in a dual-tree algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_RESULT_H
#define MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_RESULT_H

#include "core/parallel/map_vector.h"

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

    /** @brief The final regression estimates.
     */
    core::parallel::MapVector<double> regression_estimates_;

    /** @brief The lower bound on the left hand side.
     */
    core::parallel::MapVector <
    core::monte_carlo::MeanVariancePairMatrix > left_hand_side_l_;

    /** @brief The estimated left hand side.
     */
    core::parallel::MapVector <
    core::monte_carlo::MeanVariancePairMatrix > left_hand_side_e_;

    /** @brief The upper bound on the left hand side.
     */
    core::parallel::MapVector <
    core::monte_carlo::MeanVariancePairMatrix > left_hand_side_u_;

    /** @brief The lower bound on the right hand side.
     */
    core::parallel::MapVector <
    core::monte_carlo::MeanVariancePairVector > right_hand_side_l_;

    /** @brief The estimated right hand side.
     */
    core::parallel::MapVector <
    core::monte_carlo::MeanVariancePairVector > right_hand_side_e_;

    /** @brief The upper bound on the left hand side.
     */
    core::parallel::MapVector <
    core::monte_carlo::MeanVariancePairVector > right_hand_side_u_;

    /** @brief The number of points pruned per each query.
     */
    core::parallel::MapVector<unsigned long int> pruned_;

    /** @brief The amount of maximum error incurred per each query for
     *         the left hand side.
     */
    core::parallel::MapVector<double> left_hand_side_used_error_;

    /** @brief The amount of maximum error incurred per each query for
     *         the right hand side.
     */
    core::parallel::MapVector<double> right_hand_side_used_error_;


    /** @brief Accumulates the given query result to this query
     *         result.
     */
    void Accumulate(const LocalRegressionResult &result_in) {

      // Do nothing.
    }

    /** @brief Accumulates the query result.
     */
    template<typename DistributedTableType>
    void PostProcess(
      boost::mpi::communicator &world,
      DistributedTableType *distributed_table_in) {

      // This should be function that takes the old_from_new mapping
      // and does the reshuffling of the long query result vector.

    }

    /** @brief Aliases a subset of the given result.
     */
    template<typename TreeIteratorType>
    void Alias(TreeIteratorType &it) {
      regression_estimates_.set_indices_to_save(it);
      left_hand_side_l_.set_indices_to_save(it);
      left_hand_side_e_.set_indices_to_save(it);
      left_hand_side_u_.set_indices_to_save(it);
      right_hand_side_l_.set_indices_to_save(it);
      right_hand_side_e_.set_indices_to_save(it);
      right_hand_side_u_.set_indices_to_save(it);
      pruned_.set_indices_to_save(it);
      left_hand_side_used_error_.set_indices_to_save(it);
      right_hand_side_used_error_.set_indices_to_save(it);
    }

    /** @brief Aliases a subset of the given result.
     */
    template<typename TreeIteratorType>
    void Alias(
      const LocalRegressionResult &result_in, TreeIteratorType &it) {
      regression_estimates_.Alias(result_in.regression_estimates_, it);
      left_hand_side_l_.Alias(result_in.left_hand_side_l_, it);
      left_hand_side_e_.Alias(result_in.left_hand_side_e_, it);
      left_hand_side_u_.Alias(result_in.left_hand_side_u_, it);
      right_hand_side_l_.Alias(result_in.right_hand_side_l_, it);
      right_hand_side_e_.Alias(result_in.right_hand_side_e_, it);
      right_hand_side_u_.Alias(result_in.right_hand_side_u_, it);
      pruned_.Alias(result_in.pruned_, it);
      left_hand_side_used_error_.Alias(
        result_in.left_hand_side_used_error_, it);
      right_hand_side_used_error_.Alias(
        result_in.right_hand_side_used_error_, it);
    }

    /** @brief Aliases a subset of the given result.
     */
    void Alias(const LocalRegressionResult &result_in) {
      regression_estimates_.Alias(result_in.regression_estimates_);
      left_hand_side_l_.Alias(result_in.left_hand_side_l_);
      left_hand_side_e_.Alias(result_in.left_hand_side_e_);
      left_hand_side_u_.Alias(result_in.left_hand_side_u_);
      right_hand_side_l_.Alias(result_in.right_hand_side_l_);
      right_hand_side_e_.Alias(result_in.right_hand_side_e_);
      right_hand_side_u_.Alias(result_in.right_hand_side_u_);
      pruned_.Alias(result_in.pruned_);
      left_hand_side_used_error_.Alias(
        result_in.left_hand_side_used_error_);
      right_hand_side_used_error_.Alias(
        result_in.right_hand_side_used_error_);
    }

    void Copy(const LocalRegressionResult &result_in) {
      regression_estimates_.Copy(result_in.regression_estimates_);
      left_hand_side_l_.Copy(result_in.left_hand_side_l_);
      left_hand_side_e_.Copy(result_in.left_hand_side_e_);
      left_hand_side_u_.Copy(result_in.left_hand_side_u_);
      right_hand_side_l_.Copy(result_in.right_hand_side_l_);
      right_hand_side_e_.Copy(result_in.right_hand_side_e_);
      right_hand_side_u_.Copy(result_in.right_hand_side_u_);
      pruned_.Copy(result_in.pruned_);
      left_hand_side_used_error_.Copy(result_in.left_hand_side_used_error_);
      right_hand_side_used_error_.Copy(result_in.right_hand_side_used_error_);
    }

    LocalRegressionResult(const LocalRegressionResult &result_in) {
      this->operator=(result_in);
    }

    void operator=(const LocalRegressionResult &result_in) {
      this->Copy(result_in);
    }

    /** @brief Saves the local regression result object.
     */
    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {
      ar & regression_estimates_;
      ar & left_hand_side_l_;
      ar & left_hand_side_e_;
      ar & left_hand_side_u_;
      ar & right_hand_side_l_;
      ar & right_hand_side_e_;
      ar & right_hand_side_u_;
      ar & pruned_;
      ar & left_hand_side_used_error_;
      ar & right_hand_side_used_error_;
    }

    /** @brief Loads the local regression result object.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {
      ar & regression_estimates_;
      ar & left_hand_side_l_;
      ar & left_hand_side_e_;
      ar & left_hand_side_u_;
      ar & right_hand_side_l_;
      ar & right_hand_side_e_;
      ar & right_hand_side_u_;
      ar & pruned_;
      ar & left_hand_side_used_error_;
      ar & right_hand_side_used_error_;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    void Seed(int qpoint_index, unsigned long int initial_pruned_in) {
      pruned_[qpoint_index] += initial_pruned_in;
    }

    void SetZero() {
      core::parallel::MapVector <
      core::monte_carlo::MeanVariancePairMatrix >::iterator
      left_hand_side_l_it = left_hand_side_l_.get_iterator();
      core::parallel::MapVector <
      core::monte_carlo::MeanVariancePairMatrix >::iterator
      left_hand_side_e_it = left_hand_side_e_.get_iterator();
      core::parallel::MapVector <
      core::monte_carlo::MeanVariancePairMatrix >::iterator
      left_hand_side_u_it = left_hand_side_u_.get_iterator();
      core::parallel::MapVector <
      core::monte_carlo::MeanVariancePairVector >::iterator
      right_hand_side_l_it = right_hand_side_l_.get_iterator();
      core::parallel::MapVector <
      core::monte_carlo::MeanVariancePairVector >::iterator
      right_hand_side_e_it = right_hand_side_e_.get_iterator();
      core::parallel::MapVector <
      core::monte_carlo::MeanVariancePairVector >::iterator
      right_hand_side_u_it = right_hand_side_u_.get_iterator();
      core::parallel::MapVector <unsigned long int>::iterator
      pruned_it = pruned_.get_iterator();
      core::parallel::MapVector <double>::iterator
      left_hand_side_used_error_it =
        left_hand_side_used_error_.get_iterator();
      core::parallel::MapVector <double>::iterator
      right_hand_side_used_error_it =
        right_hand_side_used_error_.get_iterator();

      for(; left_hand_side_l_it.HasNext();
          left_hand_side_l_it++, left_hand_side_e_it++, left_hand_side_u_it++,
          right_hand_side_l_it++, right_hand_side_e_it++,
          right_hand_side_u_it++, pruned_it++,
          left_hand_side_used_error_it++, right_hand_side_used_error_it++) {
        left_hand_side_l_it->SetZero();
        left_hand_side_e_it->SetZero();
        left_hand_side_u_it->SetZero();
        right_hand_side_l_it->SetZero();
        right_hand_side_e_it->SetZero();
        right_hand_side_u_it->SetZero();
        (*pruned_it) = 0;
        (*left_hand_side_used_error_it) = 0.0;
        (*right_hand_side_used_error_it) = 0.0;
      }
    }

    /** @brief The default constructor.
     */
    LocalRegressionResult() {
      SetZero();
    }

    template<typename GlobalType, typename TableType>
    void PostProcess(const GlobalType &global, TableType &query_table) {
      typename TableType::TreeIterator it =
        query_table.get_node_iterator(query_table.get_tree());
      while(it.HasNext()) {
        int qpoint_id;
        arma::vec qpoint;
        it.Next(&qpoint, &qpoint_id);
        this->PostProcess(global, qpoint, qpoint_id);
      }
    }

    template<typename GlobalType>
    void PostProcess(
      const GlobalType &global,
      const arma::vec &qpoint,
      int q_index) {

      // Set up the linear system.
      left_hand_side_e_[q_index].sample_means(&tmp_left_hand_side_);
      right_hand_side_e_[q_index].sample_means(&tmp_right_hand_side_);

      if(global.problem_dimension() == 1) {
        regression_estimates_[q_index] = 0.0;
        if(tmp_left_hand_side_.at(0, 0) != 0.0) {
          regression_estimates_[q_index] = tmp_right_hand_side_[0] /
                                           tmp_left_hand_side_.at(0, 0);
        }
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

    template<typename MetricType, typename GlobalType>
    void PostProcess(
      const MetricType &metric,
      const arma::vec &qpoint,
      int q_index,
      double q_weight,
      const GlobalType &global,
      const bool is_monochromatic) {

      if(global.do_postprocess()) {
        PostProcess(global, qpoint, q_index);
      }
    }

    void Print(const std::string &file_name) const {
      FILE *file_output = fopen(file_name.c_str(), "w+");
      core::parallel::MapVector<double>::iterator regression_estimates_it =
        regression_estimates_.get_iterator();
      core::parallel::MapVector<unsigned long int>::iterator pruned_it =
        pruned_.get_iterator();
      for(; regression_estimates_it.HasNext(); regression_estimates_it++,
          pruned_it++) {
        fprintf(file_output, "%g %lu\n", *regression_estimates_it, *pruned_it);
      }
      fclose(file_output);
    }

    /** @brief Basic allocation for local regression results.
     */
    void Init(int num_points) {

      // Initialize the array storing the final regression estimates.
      regression_estimates_.Init(num_points);

      // Initialize the left hand side lower bound.
      left_hand_side_l_.Init(num_points);

      // Initialize the left hand side estimate.
      left_hand_side_e_.Init(num_points);

      // Initialize the left hand side upper bound.
      left_hand_side_u_.Init(num_points);

      // Initialize the right hand side lower bound.
      right_hand_side_l_.Init(num_points);

      // Initialize the right hand side estimate.
      right_hand_side_e_.Init(num_points);

      // Initialize the right hand side upper bound.
      right_hand_side_u_.Init(num_points);

      // Initialize the pruned quantities.
      pruned_.Init(num_points);

      // Initialize the used error quantities.
      left_hand_side_used_error_.Init(num_points);
      right_hand_side_used_error_.Init(num_points);
    }

    template<typename GlobalType>
    void Init(const GlobalType &global_in, int num_points) {

      // Basic allocation.
      this->Init(num_points);

      // Allocate Monte Carlo results.
      core::parallel::MapVector <
      core::monte_carlo::MeanVariancePairMatrix >::iterator
      left_hand_side_l_it = left_hand_side_l_.get_iterator();
      core::parallel::MapVector <
      core::monte_carlo::MeanVariancePairMatrix >::iterator
      left_hand_side_e_it = left_hand_side_e_.get_iterator();
      core::parallel::MapVector <
      core::monte_carlo::MeanVariancePairMatrix >::iterator
      left_hand_side_u_it = left_hand_side_u_.get_iterator();
      core::parallel::MapVector <
      core::monte_carlo::MeanVariancePairVector >::iterator
      right_hand_side_l_it = right_hand_side_l_.get_iterator();
      core::parallel::MapVector <
      core::monte_carlo::MeanVariancePairVector >::iterator
      right_hand_side_e_it = right_hand_side_e_.get_iterator();
      core::parallel::MapVector <
      core::monte_carlo::MeanVariancePairVector >::iterator
      right_hand_side_u_it = right_hand_side_u_.get_iterator();
      for(; left_hand_side_l_it.HasNext();
          left_hand_side_l_it++, left_hand_side_e_it++, left_hand_side_u_it++,
          right_hand_side_l_it++, right_hand_side_e_it++,
          right_hand_side_u_it++) {
        left_hand_side_l_it->Init(
          global_in.problem_dimension() ,
          global_in.problem_dimension());
        left_hand_side_e_it->Init(
          global_in.problem_dimension() ,
          global_in.problem_dimension());
        left_hand_side_u_it->Init(
          global_in.problem_dimension() ,
          global_in.problem_dimension());
        right_hand_side_l_it->Init(global_in.problem_dimension());
        right_hand_side_e_it->Init(global_in.problem_dimension());
        right_hand_side_u_it->Init(global_in.problem_dimension());
      }

      // Set everything to zero.
      SetZero();
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
        left_hand_side_l_[
          qpoint_index].CombineWith(
            (*delta_in.query_deltas_)[qpoint_index].left_hand_side_l_);
        left_hand_side_e_[
          qpoint_index].CombineWith(
            (*delta_in.query_deltas_)[qpoint_index].left_hand_side_e_);
        left_hand_side_u_[
          qpoint_index].CombineWith(
            (*delta_in.query_deltas_)[qpoint_index].left_hand_side_u_);
        right_hand_side_l_[
          qpoint_index].CombineWith(
            (*delta_in.query_deltas_)[qpoint_index].right_hand_side_l_);
        right_hand_side_e_[
          qpoint_index].CombineWith(
            (*delta_in.query_deltas_)[qpoint_index].right_hand_side_e_);
        right_hand_side_u_[
          qpoint_index].CombineWith(
            (*delta_in.query_deltas_)[qpoint_index].right_hand_side_u_);

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
      const arma::vec &qpoint,
      int q_index,
      const LocalRegressionPostponedType &postponed_in) {

      // Apply postponed.
      ApplyPostponed(q_index, postponed_in);
    }
};

}
}

#endif
