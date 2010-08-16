/** @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *
 *  @file block_coord_descent.h
 */

#ifndef CONTRIB_DONGRYEL_GP_REGRESSION_BLOCK_COORD_DESCENT_H
#define CONTRIB_DONGRYEL_GP_REGRESSION_BLOCK_COORD_DESCENT_H

#include <algorithm>
#include "fastlib/la/matrix.h"
#include "boost/utility.hpp"

namespace fl {
namespace ml {

class KernelValue {
  public:
    template<typename KernelType>
    static double Compute(
      const Table &table, const KernelType &kernel, int first_point_id,
      int second_point_id, bool is_monochromatic) {

      Vector first_point, second_point;
      table.get(first_point_id, &first_point);
      table.get(second_point_id, &second_point);

      return kernel.Dot(
               first_point, second_point,
               is_monochromatic && first_point_id == second_point_id);
    }
};

class BlockCoordDescentResult: boost::noncopyable {

  private:
    fl::data::MonolithicPoint<double> gradient_;

    fl::data::MonolithicPoint<double> solution_;

  public:

    bool IsConverged() const {
      double squared_l2_norm = fl::dense::ops::Dot(gradient_, gradient_);
      return squared_l2_norm <= 1.0e-8;
    }

    const fl::data::MonolithicPoint<double> &gradient() const {
      return gradient_;
    }

    fl::data::MonolithicPoint<double> &gradient() {
      return gradient_;
    }

    const fl::data::MonolithicPoint<double> &solution() const {
      return solution_;
    }

    fl::data::MonolithicPoint<double> &solution() {
      return solution_;
    }

    void Init(int right_hand_side_dimension_in) {
      gradient_.Init(right_hand_side_dimension_in);
      solution_.Init(right_hand_side_dimension_in);
      gradient_.SetZero();
      solution_.SetZero();
    }

    void Init(bool initialize_result,
              const fl::data::MonolithicPoint<double> &right_hand) {
      if (initialize_result) {
        gradient_.Init(right_hand.length());
        solution_.Init(right_hand.length());
      }
      Reset(right_hand);
    }

    void Reset(const fl::data::MonolithicPoint<double> &right_hand) {
      gradient_.CopyValues(right_hand);
      solution_.SetZero();
    }
};

template<typename double>
class BlockCoordDescentDelta: boost::noncopyable {

  private:

    int num_points_;

    std::vector<int> inactive_set_;

    std::vector<int> active_set_;

    fl::data::MonolithicPoint<double> delta_solution_;

    fl::data::MonolithicPoint<double> temp_gradient_info_;

    fl::dense::Matrix<double, false> inverse_;

    fl::dense::Matrix<double, false> kernel_matrix_;

  public:

    const fl::dense::Matrix<double, false> &kernel_matrix() const {
      return kernel_matrix_;
    }

    fl::dense::Matrix<double, false> &kernel_matrix() {
      return kernel_matrix_;
    }

    const fl::dense::Matrix<double, false> &inverse() const {
      return inverse_;
    }

    fl::dense::Matrix<double, false> &inverse() {
      return inverse_;
    }

    int current_active_set_size() const {
      return active_set_.size();
    }

    const fl::data::MonolithicPoint<double> &delta_solution() const {
      return delta_solution_;
    }

    fl::data::MonolithicPoint<double> &delta_solution() {
      return delta_solution_;
    }

    const fl::data::MonolithicPoint<double> &temp_gradient_info() const {
      return temp_gradient_info_;
    }

    fl::data::MonolithicPoint<double> &temp_gradient_info() {
      return temp_gradient_info_;
    }

    const std::vector<int> &inactive_set() const {
      return inactive_set_;
    }

    std::vector<int> &inactive_set() {
      return inactive_set_;
    }

    const std::vector<int> &active_set() const {
      return active_set_;
    }

    std::vector<int> &active_set() {
      return active_set_;
    }

    template < typename Matrix, typename KernelType,
    typename double >
    void Update(const Matrix &table, const KernelType &kernel,
                int new_index, double self_kernel_value,
                const fl::data::MonolithicPoint<double> &gradient) {

      fl::data::MonolithicPoint<double> beta;
      double eta = 0;
      beta.Init(active_set_.size());
      beta.SetZero();

      for (int i = 0; i < active_set_.size(); i++) {

        // Compute the kernel value with the current active point.
        int active_point_id = active_set_[i];
        kernel_matrix_.set(
          i, active_set_.size(),
          fl::ml::KernelValue::Compute(
            table, kernel, active_point_id, new_index,
            true));

        // Symmetric setting.
        kernel_matrix_.set(active_set_.size(), i,
                           kernel_matrix_.get(i, active_set_.size()));

        // Compute the beta by multiplying the old inverse by
        // the the column we are adding into the kernel matrix.
        for (int j = 0; j < active_set_.size(); j++) {
          beta[j] += kernel_matrix_.get(i, active_set_.size()) *
                     inverse_.get(j, i);
        }
      }

      // Set the self kernel value at the diagonal position.
      kernel_matrix_.set(active_set_.size(), active_set_.size(),
                         self_kernel_value);

      // Compute eta.
      for (int i = 0; i < active_set_.size(); i++) {
        eta += kernel_matrix_.get(active_set_.size(), i) * beta[i];
      }
      eta = 1.0 / (self_kernel_value - eta);

      // Update the inverse.
      for (int j = 0; j < active_set_.size() + 1; j++) {
        for (int i = 0; i < active_set_.size() + 1; i++) {
          double increment = eta;

          if (i < active_set_.size()) {
            increment *= beta[i];
          }
          else {
            increment *= (-1);
          }
          if (j < active_set_.size()) {
            increment *= beta[j];
          }
          else {
            increment *= (-1);
          }

          inverse_.set(i, j, inverse_.get(i, j) + increment);
        }
      }

      // Update the delta solution by computing the dot product
      // between beta and the current gradient.
      double factor = 0;
      for (int i = 0; i < active_set_.size(); i++) {
        factor += beta[i] * gradient[active_set_[i]];
      }
      factor -= gradient[new_index];
      factor *= eta;
      for (int i = 0; i < active_set_.size(); i++) {
        delta_solution_[active_set_[i]] -= factor * beta[i];
      }
      delta_solution_[new_index] += factor;
    }

    void Init(int num_points, int active_set_size) {
      num_points_ = num_points;
      delta_solution_.Init(num_points);
      temp_gradient_info_.Init(num_points);
      inactive_set_.resize(num_points);
      active_set_.reserve(active_set_size);
      active_set_.resize(0);
      inverse_.Init(active_set_size, active_set_size);
      kernel_matrix_.Init(std::min(active_set_size, num_points),
                          std::min(active_set_size, num_points));
      Reset();
    }

    void Reset() {
      delta_solution_.SetZero();
      temp_gradient_info_.SetZero();
      inactive_set_.resize(num_points_);
      active_set_.resize(0);
      for (int i = 0; i < num_points_; i++) {
        inactive_set_[i] = i;
      }
      inverse_.SetZero();
      kernel_matrix_.SetZero();
    }

    void Reset(const fl::data::MonolithicPoint<double> &gradient_in) {
      Reset();
      temp_gradient_info_.CopyValues(gradient_in);
    }
};

class SolveSubProblem {
  private:
    fl::data::MonolithicPoint<double> tmp_vector_;

  public:

    SolveSubProblem() {
    }

    void Init(int active_set_size, int num_points) {
      tmp_vector_.Init(std::min(active_set_size, num_points));
    }

    template < typename Matrix, typename KernelType, typename double >
    void Compute(
      int active_set_size,
      int max_random_set_size,
      const Matrix &table,
      const KernelType &kernel,
      const fl::data::MonolithicPoint<double> &right_hand_side,
      const fl::ml::BlockCoordDescentResult<double> &result,
      fl::ml::BlockCoordDescentDelta<double> &delta) {

      // Form the kernel submatrix.
      const std::vector<int> &active_set = delta.active_set();
      fl::dense::Matrix<double, false> &kernel_matrix = delta.kernel_matrix();

      for (int j = 0; j < active_set.size(); j++) {
        int column_index = active_set[j];
        for (int i = 0; i < active_set.size(); i++) {
          int row_index = active_set[i];
          kernel_matrix.set(
            i, j, fl::ml::KernelValue::Compute(
              table, kernel, row_index, column_index, true));
        }
      }

      // The curent solution.
      const fl::data::MonolithicPoint<double> &current_solution =
        result.solution();

      // Form the residual.
      fl::data::MonolithicPoint<double> residual;
      residual.Alias(delta.temp_gradient_info().ptr(), kernel_matrix.n_rows());

      for (int i = 0; i < active_set.size(); i++) {
        int row_index = active_set[i];
        residual[i] = right_hand_side[row_index];
        for (int j = 0; j < active_set.size(); j++) {
          int col_index = active_set[j];
          residual[i] -= kernel_matrix.get(i, j) * current_solution[col_index];
        }
      }

      // Compute the required update to the solution.
      success_t flag;
      fl::dense::ops::Solve<fl::la::Overwrite>(
        kernel_matrix, residual, &tmp_vector_, &flag);
      fl::data::MonolithicPoint<double> &delta_solution =
        delta.delta_solution();
      delta_solution.SetZero();
      for (int i = 0; i < kernel_matrix.n_rows(); i++) {
        int row_index = active_set[i];
        delta_solution[row_index] = -tmp_vector_[i];
      }
    }
};

template<enum fl::ml::GpRegressionComputation::Type>
class SelectActiveSetTrait {
  public:
    SelectActiveSetTrait(int active_set_size, int num_points);

    template < typename Matrix, typename KernelType, typename double >
    void Select(
      int active_set_size,
      int max_random_set_size,
      const Matrix &table,
      const KernelType &kernel,
      const fl::data::MonolithicPoint<double> &right_hand_side,
      const fl::ml::BlockCoordDescentResult<double> &result,
      fl::ml::BlockCoordDescentDelta<double> &delta);
};

template<>
class SelectActiveSetTrait<fl::ml::GpRegressionComputation::GREEDY_BC> {
  private:
    template<typename double>
    void ChooseSubset(
      const fl::ml::BlockCoordDescentResult<double> &result,
      std::vector<int> &inactive_set,
      int max_random_set_size, int *random_set_size) {

      // The maximum you can choose is bounded by the minimum of the
      // maximum random set size and the current inactive set size.
      *random_set_size =
        std::min(
          max_random_set_size,
          (int)inactive_set.size());

      for (int i = ((int) inactive_set.size()) - 1;
           i >= ((int) inactive_set.size()) - (*random_set_size); i--) {

        int random_index = fl::math::Random(0, i + 1);
        std::swap(inactive_set[random_index], inactive_set[i]);
      }
    }

  public:

    SelectActiveSetTrait(int active_set_size, int num_points) {
    }

    template < typename Matrix, typename KernelType, typename double >
    void Select(
      int active_set_size,
      int max_random_set_size,
      const Matrix &table,
      const KernelType &kernel,
      const fl::data::MonolithicPoint<double> &right_hand_side,
      const fl::ml::BlockCoordDescentResult<double> &result,
      fl::ml::BlockCoordDescentDelta<double> &delta) {

      // Initialize the gradient information.
      delta.Reset(result.gradient());

      // The references to the inactive set and the active
      // set. For the inactive set, we maintain such that the elements near
      // the tail form the random subset which we use to update
      // the gradient.
      std::vector<int> &active_set = delta.active_set();
      std::vector<int> &inactive_set = delta.inactive_set();

      // The reference to the gradient and the accumulated inverse
      // and accumulated kernel matrix.
      const fl::data::MonolithicPoint<double> &gradient =
        result.gradient();
      fl::dense::Matrix<double, false> &inverse = delta.inverse();
      fl::dense::Matrix<double, false> &kernel_matrix =
        delta.kernel_matrix();

      // The reference to the temporary gradient information for
      // selecting the active set and the delta solution.
      fl::data::MonolithicPoint<double> &temp_gradient_info =
        delta.temp_gradient_info();
      fl::data::MonolithicPoint<double> &delta_solution =
        delta.delta_solution();

      // The current random set size. Initially, it is equal to
      // the entire dataset.
      int random_set_size = inactive_set.size();

      do {

        // Select the next variable to add by solving the
        // one-dimensional optimization problem from the current
        // inactive set.
        int selected_index = 0;
        double minimum_value =
          std::numeric_limits<double>::max();
        double cached_self_kernel_value = -1;
        for (int i = inactive_set.size() - random_set_size;
             i < inactive_set.size(); i++) {
          double kernel_value =
            fl::ml::KernelValue::Compute(
              table, kernel, inactive_set[i], inactive_set[i], true);
          double current_value =
            - fl::math::Sqr(temp_gradient_info[inactive_set[i]]) /
            (2.0 * kernel_value);
          if (current_value <= minimum_value) {
            minimum_value = current_value;
            selected_index = i;
            cached_self_kernel_value = kernel_value;
          }
        }

        if (delta.current_active_set_size() == 0) {
          inverse.set(0, 0, 1.0 / cached_self_kernel_value);
          kernel_matrix.set(0, 0, cached_self_kernel_value);
          delta_solution[inactive_set[selected_index]] =
            -temp_gradient_info[inactive_set[selected_index]] /
            cached_self_kernel_value;
        }
        else {
          delta.Update(table, kernel, inactive_set[selected_index],
                       cached_self_kernel_value, gradient);
        }

        // Add the point to the active set.
        active_set.push_back(inactive_set[selected_index]);
        inactive_set[selected_index] =
          inactive_set[inactive_set.size() - 1];

        // Decrement the inactive set.
        inactive_set.resize(inactive_set.size() - 1);

        // Randomly choose a subset from the inactive set and
        // update the gradient component.
        ChooseSubset(
          result, inactive_set, max_random_set_size, &random_set_size);

        for (int i = inactive_set.size() - random_set_size;
             i < inactive_set.size(); i++) {

          // The index of the randomly chosen point.
          int random_point_index = inactive_set[i];
          double dot_product = 0;

          // Loop over each point in the current active set.
          for (int j = 0; j < active_set.size(); j++) {
            int active_point_index = active_set[j];
            double kernel_value =
              fl::ml::KernelValue::Compute(
                table, kernel, random_point_index, active_point_index, true);
            dot_product += kernel_value *
                           delta_solution[active_point_index];
          }
          temp_gradient_info[random_point_index] = dot_product +
              gradient[random_point_index];
        }

      }
      while (delta.current_active_set_size() < active_set_size &&
             inactive_set.size() > 0);
    }
};

template<>
class SelectActiveSetTrait<fl::ml::GpRegressionComputation::CYCLIC_BC> {
  private:
    int starting_index_;

    SolveSubProblem subproblem_;

  public:

    SelectActiveSetTrait(int active_set_size, int num_points) {
      starting_index_ = 0;
      subproblem_.Init(active_set_size, num_points);
    }

    template < typename Matrix, typename KernelType, typename double >
    void Select(
      int active_set_size,
      int max_random_set_size,
      const TableType &table,
      const KernelType &kernel,
      const fl::data::MonolithicPoint<double> &right_hand_side,
      const fl::ml::BlockCoordDescentResult<double> &result,
      fl::ml::BlockCoordDescentDelta<double> &delta) {

      // Set the active and inactive sets.
      std::vector<int> &active_set = delta.active_set();
      std::vector<int> &inactive_set = delta.inactive_set();
      active_set.resize(0);
      inactive_set.resize(0);
      for (int i = 0; i < std::min(active_set_size, table.n_entries()); i++) {
        int active_index = (i + starting_index_) % table.n_entries();
        active_set.push_back(active_index);
      }
      for (int i = std::min(active_set_size, table.n_entries());
           i < table.n_entries(); i++) {
        int inactive_index = (i + starting_index_) % table.n_entries();
        inactive_set.push_back(inactive_index);
      }

      // Solve the subproblem.
      subproblem_.Compute(
        active_set_size, max_random_set_size, table, kernel, right_hand_side,
        result,	delta);

      // Update the starting index for the next iteration.
      starting_index_ =
        (starting_index_ + std::min(active_set_size, table.n_entries())) %
        table.n_entries();
    }
};

template<>
class SelectActiveSetTrait<fl::ml::GpRegressionComputation::GRADIENT_BC> {
  private:
    std::vector<int> sorted_indices_;

    SolveSubProblem subproblem_;

    class Comparator {
      private:
        const fl::data::MonolithicPoint<double> *gradient_;

      public:
        void Init(const fl::data::MonolithicPoint<double> &gradient_in) {
          gradient_ = &gradient_in;
        }

        bool operator()(int first_point_index, int second_point_index) {
          double absolute_value_first_point_gradient =
            fabs((*gradient_)[first_point_index]);
          double absolute_value_second_point_gradient =
            fabs((*gradient_)[second_point_index]);
          return (absolute_value_first_point_gradient >
                  absolute_value_second_point_gradient) ||
                 (absolute_value_first_point_gradient ==
                  absolute_value_second_point_gradient &&
                  first_point_index > second_point_index);
        }
    } comp;

  public:

    SelectActiveSetTrait(int active_set_size, int num_points) {
      sorted_indices_.resize(num_points);
      subproblem_.Init(active_set_size, num_points);
    }

    template < typename TableType, typename KernelType, typename double >
    void Select(
      int active_set_size,
      int max_random_set_size,
      const TableType &table,
      const KernelType &kernel,
      const fl::data::MonolithicPoint<double> &right_hand_side,
      const fl::ml::BlockCoordDescentResult<double> &result,
      fl::ml::BlockCoordDescentDelta<double> &delta) {

      // Sort the current gradient component by absolute magnitude and
      // select the indices with the greatest ones.
      for (int i = 0; i < sorted_indices_.size(); i++) {
        sorted_indices_[i] = i;
      }

      comp.Init(result.gradient());
      std::sort(sorted_indices_.begin(), sorted_indices_.end(), comp);

      // Set the active and inactive sets.
      std::vector<int> &active_set = delta.active_set();
      std::vector<int> &inactive_set = delta.inactive_set();
      active_set.resize(0);
      inactive_set.resize(0);
      for (int i = 0; i < std::min(active_set_size, table.n_entries()); i++) {
        active_set.push_back(sorted_indices_[i]);
      }
      for (int i = std::min(active_set_size, table.n_entries());
           i < table.n_entries(); i++) {
        inactive_set.push_back(sorted_indices_[i]);
      }

      // Solve the subproblem.
      subproblem_.Compute(
        active_set_size, max_random_set_size, table, kernel, right_hand_side,
        result,	delta);
    }
};

template<enum fl::ml::GpRegressionComputation::Type ComputationType>
class BlockCoordDescent: private boost::noncopyable {
  private:

    template<typename double>
    static void Update_(const BlockCoordDescentDelta<double> &delta,
                        BlockCoordDescentResult<double> &result) {

      // The selected active set.
      const std::vector<int> &active_set = delta.active_set();

      // The accumulated kernel matrix.
      const fl::dense::Matrix<double, false> &kernel_matrix =
        delta.kernel_matrix();

      // Delta change and the final accumulated result.
      const fl::data::MonolithicPoint<double> &delta_solution =
        delta.delta_solution();
      fl::data::MonolithicPoint<double> &solution = result.solution();
      fl::data::MonolithicPoint<double> &gradient = result.gradient();

      for (int i = 0; i < active_set.size(); i++) {
        solution[active_set[i]] -= delta_solution[active_set[i]];

        double dot_product = 0;
        for (int j = 0; j < active_set.size(); j++) {
          dot_product += kernel_matrix.get(i, j) *
                         delta_solution[active_set[j]];
        }
        gradient[active_set[i]] += dot_product;
      }
    }

  public:

    template<typename TableType, typename KernelType, typename double>
    static void Compute(
      int active_set_size,
      int max_random_set_size,
      const TableType &table, const KernelType &kernel,
      const fl::data::MonolithicPoint<double> &right_hand,
      bool initialize_result,
      fl::ml::BlockCoordDescentResult<double> *result) {

      // Initialize the result.
      result->Init(initialize_result, right_hand);

      // Make a delta object.
      BlockCoordDescentDelta<double> delta;
      delta.Init(table.n_entries(), active_set_size);

      // A trait for selecting the active set.
      fl::ml::SelectActiveSetTrait<ComputationType> active_set_selection_trait(
        active_set_size, table.n_entries());

      // The main loop.
      while (!result->IsConverged()) {

        active_set_selection_trait.Select(
          active_set_size, max_random_set_size, table, kernel, right_hand,
          *result, delta);
        Update_(delta, *result);
      }
    }
};
};
};

#endif
