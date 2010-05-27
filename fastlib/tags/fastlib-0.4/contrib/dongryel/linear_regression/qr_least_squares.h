#ifndef FL_LITE_MLPACK_REGRESSION_QR_LEAST_SQUARES_H
#define FL_LITE_MLPACK_REGRESSION_QR_LEAST_SQUARES_H

#include <deque>
#include <list>
#include <vector>
#include "fastlib/la/matrix.h"

class QRLeastSquares {

  public:

    class ResizableMatrix {
      private:

        std::deque< Vector * > matrix_;

      public:

        ~ResizableMatrix() {
          for (int i = 0; i < (int) matrix_.size(); i++) {
            delete matrix_[i];
          }
        }

        void PrintDebug() const {
          printf("----- RESIZABLE MATRIX  -----\n");
          for (int i = 0; i < n_rows(); i++) {
            for (int j = 0; j < n_cols(); j++) {
              printf("%+3.3f ", (*matrix_[i])[j]);
            }
            printf("\n");
          }
        }

        int n_rows() const {
          return matrix_.size();
        }

        int n_cols() const {
          if (matrix_.size() == 0) {
            return 0;
          }
          else {
            return matrix_[0]->length();
          }
        }

        void Init(const Matrix &table_in,
                  const std::deque<int> &initial_active_column_indices,
                  int initial_right_hand_side_column_index,
                  bool include_bias_term_in) {

          int dimension = initial_active_column_indices.size();
          if (include_bias_term_in) {
            dimension++;
          }
          Vector predictions;
          Matrix q_transform, full_r_factor;
          q_transform.Init(table_in.n_cols(), dimension);
          predictions.Init(table_in.n_cols());
          for (int i = 0; i < table_in.n_cols(); i++) {
            Vector point;
            table_in.MakeColumnVector(i, &point);
            int j = 0;
            for (std::deque<int>::const_iterator it =
                   initial_active_column_indices.begin();
                 it != initial_active_column_indices.end(); it++, j++) {
              q_transform.set(i, j, point[ *it]);
            }
            if (include_bias_term_in) {
              q_transform.set(i, q_transform.n_cols() - 1, 1.0);
            }
            predictions[i] = point[initial_right_hand_side_column_index];
          }
          full_r_factor.Init(std::min(q_transform.n_rows(),
                                      q_transform.n_cols()),
                             q_transform.n_cols());
          la::QRExpert(&q_transform, &full_r_factor);

          // Project the prediction vector onto the Q factor.
          Matrix q_factor_alias;
          Vector transformed_predictions;
          q_factor_alias.Alias(
            q_transform.ptr(), q_transform.n_rows(),
            std::min(q_transform.n_rows(), q_transform.n_cols()));
          la::MulInit(
            predictions, q_factor_alias, &transformed_predictions);

          // Transfer the batch result to the R factor.
          int r_factor_dimension = table_in.n_rows();
          if (include_bias_term_in) {
            r_factor_dimension++;
          }
          for (int i = 0; i < full_r_factor.n_rows(); i++) {
            matrix_.push_front(new Vector());
            matrix_[0]->Init(r_factor_dimension);
          }

          for (int i = 0; i < full_r_factor.n_rows(); i++) {

            // Copy for each active indices.
            int j = 0;
            for (std::deque<int>::const_iterator it =
                   initial_active_column_indices.begin();
                 it != initial_active_column_indices.end(); it++, j++) {
              (*matrix_[i])[*it] = full_r_factor.get(i, j);
            }

            // Copy the intercept transformed, if available.
            if (include_bias_term_in) {
              (*matrix_[i])[n_cols() - 1] =
                full_r_factor.get(i, full_r_factor.n_cols() - 1);
            }

            // Copy the right hand side transformed.
            (*matrix_[i])[initial_right_hand_side_column_index] =
              transformed_predictions[i];
          }

          printf("Finished initializing the batch factor.\n");
        }

        void Compactify(std::deque<int> &active_column_indices) {

          bool row_is_all_zero = true;
          do {
            int last_row = n_rows() - 1;
            for (std::deque<int>::iterator active_column_indices_it =
                   active_column_indices.begin();
                 active_column_indices_it != active_column_indices.end();
                 active_column_indices_it++) {
              if (fabs((*matrix_[last_row])[*active_column_indices_it]) >
                  std::numeric_limits<double>::min()) {
                row_is_all_zero = false;
                break;
              }
            }
            if (row_is_all_zero) {
              delete matrix_[last_row];
              matrix_.pop_back();
            }
          }
          while (row_is_all_zero && n_rows() > 0);
        }

        template<typename PointType>
        void push_front(const PointType &point, bool include_bias_term) {

          matrix_.push_front(new Vector());

          int dimension = point.length();
          if (include_bias_term) {
            dimension++;
          }
          matrix_[0]->Init(dimension);
          for (int i = 0; i < point.length(); i++) {
            (*matrix_[0])[i] = point[i];
          }
          if (include_bias_term) {
            (*matrix_[0])[point.length()] = 1.0;
          }
        }

        double get(int row, int col) const {
          return (*matrix_[row])[col];
        }

        void set(int row, int col, double val_in) {
          (*matrix_[row])[col] = val_in;
        }
    };

  private:

    bool include_bias_term_;

    ResizableMatrix r_factor_;

    std::deque<int> active_column_indices_;

    std::deque<int> inactive_column_indices_;

    int active_right_hand_side_column_index_;

  private:

    std::deque<int>::iterator FindPosition_(
      std::deque<int> &list, int find_val,
      int *index_position);

    void MaintainTriangularity_(bool loop_all_active_columns,
                                bool loop_all_rows,
                                std::deque<int>::iterator &starting_column_it,
                                int starting_column_counter);

    void Sort_(std::deque<int> &sort_deque);

  public:

    int n_rows() const;

    std::deque<int> &inactive_column_indices();

    std::deque<int> &active_column_indices();

    const std::deque<int> &active_column_indices() const;

    int active_right_hand_side_column_index() const;

    void set_active_right_hand_side_column_index(
      int active_right_hand_side_column_index_in);

    void Init(
      const Matrix &table_in,
      const std::deque<int> &initial_active_column_indices,
      int initial_right_hand_side_index,
      bool include_bias_term_in);

    template<typename PointType>
    void PushRow(const PointType &row_in);

    template<typename PointType>
    void InsertColumnIntoActiveSet(int column_index);

    void MakeColumnActive(int column_index);
    void MakeColumnInactive(int column_index);

    template<typename PointType>
    void TransposeSolve(const PointType &right_hand_side,
                        PointType *solution_out) const;

    template<typename PointType>
    void Solve(PointType *right_hand_side, PointType *solution_out) const;

    void ClearInactiveColumns();
};

#endif
