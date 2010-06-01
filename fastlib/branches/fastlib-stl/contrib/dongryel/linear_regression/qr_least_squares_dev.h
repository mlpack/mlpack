#ifndef FL_LITE_MLPACK_REGRESSION_QR_LEAST_SQUARES_DEV_H
#define FL_LITE_MLPACK_REGRESSION_QR_LEAST_SQUARES_DEV_H

#include "qr_least_squares.h"
#include "givens_rotate_dev.h"

#include <list>

int QRLeastSquares::n_rows() const {
  return r_factor_.n_rows();
}

std::deque<int> &QRLeastSquares::inactive_column_indices() {
  return inactive_column_indices_;
}

std::deque<int> &QRLeastSquares::active_column_indices() {
  return active_column_indices_;
}

const std::deque<int> &QRLeastSquares::active_column_indices() const {
  return active_column_indices_;
}

int QRLeastSquares::active_right_hand_side_column_index() const {
  return active_right_hand_side_column_index_;
}

void QRLeastSquares::set_active_right_hand_side_column_index(
  int right_hand_side_column_index_in) {

  // If it is already the active as the right hand side, then do
  // nothing.
  if (active_right_hand_side_column_index_ ==
      right_hand_side_column_index_in) {
    return;
  }

  // Try removing the column from the active set.
  MakeColumnInactive(right_hand_side_column_index_in);

  // Here, it is guaranteed that right_hand_side_column_index_in is
  // among the inactive set. We need to swap it with the current right
  // hand side index.
  int index_position;
  std::deque<int>::iterator inactive_it =
    FindPosition_(inactive_column_indices_, right_hand_side_column_index_in,
                  &index_position);
  inactive_column_indices_.erase(inactive_it);
  inactive_column_indices_.push_back(active_right_hand_side_column_index_);
  Sort_(inactive_column_indices_);

  active_right_hand_side_column_index_ = right_hand_side_column_index_in;
}

std::deque<int>::iterator QRLeastSquares::FindPosition_(
  std::deque<int> &list, int find_val, int *index_position) {

  std::deque<int>::iterator it = list.begin();
  *index_position = -1;
  int i = 0;
  for (; it != list.end(); it++, i++) {
    if (*it == find_val) {
      *index_position = i;
      break;
    }
  }
  return it;
}

void QRLeastSquares::Sort_(std::deque<int> &sort_deque) {
  std::list<int> tmp_list;
  for (int i = 0; i < (int) sort_deque.size(); i++) {
    tmp_list.push_back(sort_deque[i]);
  }
  tmp_list.sort();
  sort_deque.resize(0);
  for (std::list<int>::iterator it = tmp_list.begin(); it != tmp_list.end();
       it++) {
    sort_deque.push_back(*it);
  }
}

void QRLeastSquares::MaintainTriangularity_(
  bool loop_all_active_columns,
  bool loop_all_rows,
  std::deque<int>::iterator &starting_column_it,
  int starting_column_counter) {

  // Perform the Givens transform so that triangularity is maintained
  // in the current active indices.
  int active_column_index_counter = starting_column_counter;
  std::deque<int>::iterator column_limit = active_column_indices_.end();

  if (loop_all_active_columns == false) {
    column_limit = starting_column_it;
    column_limit++;
  }
  for (std::deque<int>::iterator &active_index_it = starting_column_it;
       active_index_it != column_limit;
       active_index_it++, active_column_index_counter++) {

    // Get the current active index.
    int active_index = *active_index_it;

    // Loop backwards row-wise.
    int start_row_counter = r_factor_.n_rows() - 1;
    if (loop_all_rows == false) {
      start_row_counter = active_column_index_counter + 1;
    }

    for (int row_counter = start_row_counter;
         row_counter > active_column_index_counter; row_counter--) {

      // Rotate the row_counter-th row and the (row_counter - 1)-th row.
      double magnitude, cosine_value, sine_value;

      if (row_counter > 0 && row_counter < r_factor_.n_rows()) {
        GivensRotate::Compute(r_factor_.get(row_counter - 1,
                                            active_index),
                              r_factor_.get(row_counter, active_index),
                              &magnitude, &cosine_value,
                              &sine_value);

        // This applies the transformation to both inactive and active
        // columns plus the prediction index.
        GivensRotate::ApplyToRow(cosine_value, sine_value,
                                 row_counter - 1, row_counter,
                                 r_factor_);
        // Set the rotated row entry to zero.
        r_factor_.set(row_counter, active_index, 0.0);
      }
    }
  }
}

void QRLeastSquares::Init(
  const Matrix &table_in,
  const std::deque<int> &initial_active_column_indices,
  int initial_right_hand_side_column_index,
  bool include_bias_term_in) {

  // Set the flag for determining whether to use the bias term or not.
  include_bias_term_ = include_bias_term_in;

  // Set the initially active left hand side indices.
  active_column_indices_ = initial_active_column_indices;

  // If the bias term is requested, then the last active index is the
  // bias term with the index of $D$, i.e. the dimensionality of the
  // table.
  if (include_bias_term_) {
    active_column_indices_.push_back(table_in.n_rows());
  }

  // Sort the indices.
  Sort_(active_column_indices_);

  // Set the initially active right hand side index.
  active_right_hand_side_column_index_ = initial_right_hand_side_column_index;

  // Allocate space for factors.
  r_factor_.Init(table_in, initial_active_column_indices,
                 initial_right_hand_side_column_index, include_bias_term_);
  /*
  for (int i = table_in.n_entries() - 1; i >= 0; i--) {
    typename TableType::Dataset_t::Point_t point;
    fl::logger->Message() << "Processing " << i << "-th row.";
    table_in.get(i, &point);
    PushRow(point);
  }
  */
}

template<typename PointType>
void QRLeastSquares::PushRow(const PointType &row_in) {

  // Grow the R factor by one row.
  int index_position;
  r_factor_.push_front(row_in, include_bias_term_);
  std::deque<int>::iterator starting_column_it =
    FindPosition_(active_column_indices_, active_column_indices_[0],
                  &index_position);
  MaintainTriangularity_(true, false, starting_column_it, index_position);
}

void QRLeastSquares::MakeColumnActive(int column_index) {

  // Find the column index from the inactive set, and remove it and
  // put it into the active set.
  int index_position = -1;
  std::deque<int>::iterator inactive_it =
    FindPosition_(inactive_column_indices_, column_index, &index_position);

  // If the following statement holds, then the column_index is the
  // right hand side, which means it should not be removed from the
  // inactive set.
  if (inactive_it != inactive_column_indices_.end()) {
    inactive_column_indices_.erase(inactive_it);
  }
  active_column_indices_.push_back(column_index);
  Sort_(active_column_indices_);

  // Find the column that was just inserted from the active set.
  std::deque<int>::iterator starting_column_it =
    FindPosition_(active_column_indices_, column_index, &index_position);
  MaintainTriangularity_(false, true, starting_column_it, index_position);
}

void QRLeastSquares::MakeColumnInactive(int column_index) {

  // Find the column index from the active set, and remove it and put
  // it into the inactive set.
  int index_position = -1;
  std::deque<int>::iterator active_it =
    FindPosition_(active_column_indices_, column_index, &index_position);
  if (active_it == active_column_indices_.end()) {
    return;
  }

  // Save the iterator pointed by the next element in the active set.
  std::deque<int>::iterator next_it = active_it;
  next_it++;
  int find_val = *next_it;

  active_column_indices_.erase(active_it);
  inactive_column_indices_.push_back(column_index);
  Sort_(inactive_column_indices_);

  // Find the next element in the list.
  next_it = FindPosition_(active_column_indices_, find_val, &index_position);

  // Downdate the R factor.
  MaintainTriangularity_(true, false, next_it, index_position);
}

template<typename PointType>
void QRLeastSquares::TransposeSolve(const PointType &right_hand_side,
                                    PointType *solution) const {

  bool rank_deficient = false;
  solution->SetZero();

  int i = 0;
  for (std::deque<int>::const_iterator outer_begin =
         active_column_indices_.begin();
       outer_begin != active_column_indices_.end(); i++, outer_begin++) {

    double temp = right_hand_side[i];
    int column_index = *outer_begin;

    if (column_index >= r_factor_.n_cols()) {
      continue;
    }

    int j = 0;
    for (std::deque<int>::const_iterator inner_begin =
           active_column_indices_.begin();
         j < std::min(i, r_factor_.n_rows()); j++, inner_begin++) {
      temp -= r_factor_.get(j, column_index) * ((*solution)[j]);
    }
    if (i < r_factor_.n_rows() && fabs(r_factor_.get(i, column_index))
        > std::numeric_limits<double>::min()) {
      (*solution)[i] = temp / r_factor_.get(i, column_index);
    }
    else {
      rank_deficient = true;
    }
  }

  //if (rank_deficient) {
  //fl::logger->Warning() << "Rank deficiency detected in transpose solve.";
  //}
}

void QRLeastSquares::ClearInactiveColumns() {
  inactive_column_indices_.resize(0);
}

template<typename PointType>
void QRLeastSquares::Solve(PointType *right_hand_side,
                           PointType *solution) const {

  bool rank_deficient = false;

  // Do a triangular solve.
  solution->SetZero();

  // The starting index for the row.
  int j = active_column_indices_.size() - 1;
  for (std::deque<int>::const_reverse_iterator outer_begin =
         active_column_indices_.rbegin(); j >= 0; j--, ++outer_begin) {

    if (j >= r_factor_.n_rows()) {
      continue;
    }
    double temp;
    if (right_hand_side == NULL) {
      temp = r_factor_.get(j, active_right_hand_side_column_index_);
    }
    else {
      temp = (*right_hand_side)[j];
    }
    int row_index = *outer_begin;

    int i = active_column_indices_.size() - 1;
    for (std::deque<int>::const_reverse_iterator inner_begin =
           active_column_indices_.rbegin(); i > j; i--, ++inner_begin) {
      int column_index = *inner_begin;
      temp -= r_factor_.get(j, column_index) * ((*solution)[i]);
    }
    if (fabs(r_factor_.get(j, row_index)) >
        std::numeric_limits<double>::min()) {
      (*solution)[j] = temp / r_factor_.get(j, row_index);
    }
    else {
      rank_deficient = true;
    }
  }

  //if (rank_deficient) {
  //fl::logger->Warning() << "Encountered rank deficient case in linear "
  //"system solving!";
  //}
}

#endif
