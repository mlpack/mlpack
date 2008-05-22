#ifndef CUR_DECOMPOSITION_H
#define CUR_DECOMPOSITION_H

#include "fastlib/fastlib.h"
#include "fastlib/la/uselapack.h"

class CURDecomposition {

 private:

  static const double epsilon_ = 0.01;

  ////////// Private Member Functions //////////
  
  static success_t QRExpert(Matrix *A_in_Q_out, ArrayList<f77_integer> *pivots,
			    Matrix *R) {
    f77_integer info;
    f77_integer m = A_in_Q_out->n_rows();
    f77_integer n = A_in_Q_out->n_cols();
    f77_integer k = std::min(m, n);
    f77_integer lwork = n * la::dgeqrf_dorgqr_block_size;
    double tau[k + lwork];
    double *work = tau + k;
    pivots->Init(n);
    
    // Obtain both Q and R in A_in_Q_out
    F77_FUNC(la::dgeqp3)(m, n, A_in_Q_out->ptr(), m, pivots->begin(),
			 tau, work, lwork, &info);
    
    if (info != 0) {
      return SUCCESS_FROM_LAPACK(info);
    }
    
    // Extract R
    for (index_t j = 0; j < n; j++) {
      double *r_col = R->GetColumnPtr(j);
      double *q_col = A_in_Q_out->GetColumnPtr(j);
      int i = std::min(j + 1, k);
      mem::Copy(r_col, q_col, i);
      mem::Zero(r_col + i, k - i);
    }
    
    // Fix Q
    F77_FUNC(la::dorgqr)(m, k, k, A_in_Q_out->ptr(), m,
			 tau, work, lwork, &info);
    
    return SUCCESS_FROM_LAPACK(info);
  }
  
  static success_t QRInit(const Matrix &A, ArrayList<f77_integer> *pivots, 
			  Matrix *Q, Matrix *R) {

    index_t k = std::min(A.n_rows(), A.n_cols());
    Q->Copy(A);
    R->Init(k, A.n_cols());
    success_t success = QRExpert(Q, pivots, R);
    Q->ResizeNoalias(k);
    
    return success;
  }
  
  /** @brief The comparison function used for quick sort
   */
  static int qsort_compar_(const void *a, const void *b) {
    
    index_t a_dereferenced = *((index_t *) a);
    index_t b_dereferenced = *((index_t *) b);
    
    if(a_dereferenced < b_dereferenced) {
      return -1;
    }
    else if(a_dereferenced > b_dereferenced) {
      return 1;
    }
    else {
      return 0;
    }
  }

  /** @brief Removes duplicate elements in a sorted array.
   */
  static void remove_duplicates_in_sorted_array_(ArrayList<index_t> &array) {

    index_t i, k = 0;

    for(i = 1; i < array.size(); i++) {
      if(array[k] != array[i]) {
        array[k + 1] = array[i];
        k++;
      }
    }
    array.ShrinkTo(k + 1);
  }

  static index_t FindBinNumber_(const Vector &cumulative_distribution,
				double random_number) {
        
    for(index_t i = 0; i < cumulative_distribution.length(); i++) {
      if(random_number < cumulative_distribution[i]) {
        return i;
      }
    }    
    return cumulative_distribution.length() - 1;
  }

 public:

  static void ExactCompute(const Matrix &a_mat, Matrix *projection_operator,
			   ArrayList<index_t> *column_indices) {

    ArrayList<f77_integer> pivots;
    Matrix q_factor_mat, r_factor_mat;
    success_t flag = QRInit(a_mat, &pivots, &q_factor_mat, &r_factor_mat);

    // Check to make sure that QR decomposition has succeeded.
    DEBUG_ASSERT(flag);

    // Compute the Frobenius norm of each row of R matrix.
    Vector frobenius_norm_row_r_factor_mat;
    frobenius_norm_row_r_factor_mat.Init(r_factor_mat.n_rows());
    frobenius_norm_row_r_factor_mat.SetZero();
    for(index_t c = 0; c < r_factor_mat.n_cols(); c++) {
      for(index_t r = 0; r < r_factor_mat.n_rows(); r++) {
	frobenius_norm_row_r_factor_mat[r] +=
	   r_factor_mat.get(r, c) * r_factor_mat.get(r, c);
      }
    }

    // Compute the reverse cumulative distribution.
    for(index_t i = frobenius_norm_row_r_factor_mat.length() - 1; i >= 1; 
	i--) {      
      frobenius_norm_row_r_factor_mat[i - 1] += 
	frobenius_norm_row_r_factor_mat[i];
    }

    // Determine how many columns to keep. Let k be the number of
    // columns to keep.
    index_t last_column = frobenius_norm_row_r_factor_mat.length() - 1;
    for(index_t i = frobenius_norm_row_r_factor_mat.length() - 1; i >= 1;
	i--) {
      last_column = i;
      if(frobenius_norm_row_r_factor_mat[i] >= epsilon_ *
	 frobenius_norm_row_r_factor_mat[0]) {
	break;
      }
    }
    index_t number_of_columns = last_column + 1;

    // Get the upper left k by k matrix and the upper right k by (n -
    // k) matrix.
    Matrix upper_left, upper_right;
    upper_left.Init(number_of_columns, number_of_columns);
    upper_right.Init(number_of_columns, a_mat.n_cols() - number_of_columns);
    for(index_t c = 0; c < a_mat.n_cols(); c++) {
      for(index_t r = 0; r < number_of_columns; r++) {
	if(c < number_of_columns) {
	  upper_left.set(r, c, r_factor_mat.get(r, c));
	}
	else {
	  upper_right.set(r, c - number_of_columns, r_factor_mat.get(r, c));
	}
      }
    }

    // Compute the SVD of the upper left matrix and solve the least
    // squares problem.
    Vector tmp_singular_values;
    Matrix tmp_left_singular_vectors, tmp_right_singular_vectors_transposed;
    Matrix intermediate, transformation_matrix;
    la::SVDInit(upper_left, &tmp_singular_values, &tmp_left_singular_vectors,
		&tmp_right_singular_vectors_transposed);
    la::MulTransAInit(tmp_left_singular_vectors, upper_right,
		      &intermediate);
    for(index_t i = 0; i < tmp_singular_values.length(); i++) {
      if(tmp_singular_values[i] > 0.0001) {
	for(index_t j = 0; j < intermediate.n_cols(); j++) {
	  intermediate.set(i, j, intermediate.get(i, j) /
			   tmp_singular_values[i]);
	}
      }
      else {
	for(index_t j = 0; j < intermediate.n_cols(); j++) {
	  intermediate.set(i, j, 0);
	}
      }
    }
    la::MulTransAInit(tmp_right_singular_vectors_transposed, intermediate,
		      &transformation_matrix);

    // Retreive the column indices.
    column_indices->Init(number_of_columns);
    for(index_t i = 0; i < number_of_columns; i++) {
      (*column_indices)[i] = pivots[i] - 1;
    }

    // Return the projection operator.
    projection_operator->Init(number_of_columns, a_mat.n_cols());
    projection_operator->SetZero();
    for(index_t i = 0; i < number_of_columns; i++) {
      projection_operator->set(i, i, 1);
    }
    for(index_t i = 0; i < transformation_matrix.n_cols(); i++) {
      la::ScaleOverwrite(transformation_matrix.n_rows(), 1, 
			 transformation_matrix.GetColumnPtr(i),
			 projection_operator->GetColumnPtr
			 (i + number_of_columns));
    }
  }

  static void Compute(const Matrix &a_mat, Matrix *c_mat, Matrix *u_mat, 
		      Matrix *r_mat, ArrayList<index_t> *column_indices,
		      ArrayList<index_t> *row_indices) {
    
    Vector column_length_square_distribution;
    column_length_square_distribution.Init(a_mat.n_cols());
    
    // Compute the column length square distributions.
    for(index_t i = 0; i < a_mat.n_cols(); i++) {
      const double *i_th_column = a_mat.GetColumnPtr(i);
      double squared_length = 
	la::Dot(a_mat.n_rows(), i_th_column, i_th_column);
      column_length_square_distribution[i] = (i == 0) ? 
	squared_length:squared_length + 
	column_length_square_distribution[i - 1];
    }

    // If the all entries are close to zero, then we return.
    if(column_length_square_distribution[a_mat.n_cols() - 1] < 0.01) {
      c_mat->Init(a_mat.n_rows(), 1);
      c_mat->SetZero();
      u_mat->Init(1, 1);
      u_mat->set(0, 0, 1);
      r_mat->Init(1, a_mat.n_cols());
      r_mat->SetZero();
      column_indices->Init(1);
      (*column_indices)[0] = 0;
      row_indices->Init(1);
      (*row_indices)[0] = 0;
      return;
    }

    // Pick samples from the column distribution to form the matrix C.
    int num_column_samples = ((int) sqrt(a_mat.n_cols()));
    column_indices->Init(num_column_samples);
    for(index_t s = 0; s < num_column_samples; s++) {
      double random_number = 
	math::Random(0, column_length_square_distribution[a_mat.n_cols() - 1]);
      (*column_indices)[s] = 
	FindBinNumber_(column_length_square_distribution, random_number);
    }
    qsort(column_indices->begin(), column_indices->size(),
	  sizeof(index_t), &qsort_compar_);
    remove_duplicates_in_sorted_array_(*column_indices);
    num_column_samples = column_indices->size();
    c_mat->Init(a_mat.n_rows(), num_column_samples);    
    
    for(index_t s = 0; s < num_column_samples; s++) {
      index_t sample_column_number = (*column_indices)[s];
      double probability = 
	((sample_column_number == 0) ?
	 column_length_square_distribution[sample_column_number]:
	 column_length_square_distribution[sample_column_number] -
	 column_length_square_distribution[sample_column_number - 1]) /
	column_length_square_distribution[a_mat.n_cols() - 1];

      la::ScaleOverwrite(a_mat.n_rows(), 
			 1.0 / sqrt(num_column_samples * probability),
			 a_mat.GetColumnPtr(sample_column_number),
			 c_mat->GetColumnPtr(s));
    }

    // Form C^T C and compute its SVD.
    Matrix outer_product, left_singular_vectors,
      right_singular_vectors_transposed;
    Vector singular_values;
    la::MulTransAInit(*c_mat, *c_mat, &outer_product);
    la::SVDInit(outer_product, &singular_values, &left_singular_vectors,
		&right_singular_vectors_transposed);

    // Get the squared length distribution the row vectors.
    Vector row_length_square_distribution;
    row_length_square_distribution.Init(a_mat.n_rows());
    row_length_square_distribution.SetZero();
    for(index_t c = 0; c < a_mat.n_cols(); c++) {
      for(index_t r = 0; r < a_mat.n_rows(); r++) {
	row_length_square_distribution[r] += a_mat.get(r, c) * a_mat.get(r, c);
      }
    }
    for(index_t r = 1; r < a_mat.n_rows(); r++) {
      row_length_square_distribution[r] += 
	row_length_square_distribution[r - 1];
    }

    // Sample the row vector according to its distribution.
    int num_row_samples = ((int) sqrt(a_mat.n_rows()));    
    row_indices->Init(num_row_samples);
    for(index_t s = 0; s < num_row_samples; s++) {
      double random_number = 
	math::Random(0, row_length_square_distribution[a_mat.n_rows() - 1]);
      (*row_indices)[s] = 
	FindBinNumber_(row_length_square_distribution, random_number);
    }
    qsort(row_indices->begin(), row_indices->size(), sizeof(index_t), 
	  &qsort_compar_);
    remove_duplicates_in_sorted_array_(*row_indices);

    // Recompute the required row sample numbers and allocate the R
    // factor and Psi matrix based on it.
    num_row_samples = row_indices->size();
    r_mat->Init(num_row_samples, a_mat.n_cols());
    Matrix psi_mat;
    psi_mat.Init(num_row_samples, num_column_samples);

    for(index_t s = 0; s < num_row_samples; s++) {
      index_t sample_row_number = (*row_indices)[s];
      double probability = 
	((sample_row_number == 0) ?
	 row_length_square_distribution[sample_row_number]:
	 row_length_square_distribution[sample_row_number] -
	 row_length_square_distribution[sample_row_number - 1]) /
	row_length_square_distribution[a_mat.n_rows() - 1];

      for(index_t c = 0; c < a_mat.n_cols(); c++) {
	r_mat->set(s, c, a_mat.get(sample_row_number, c) /
		   sqrt(num_row_samples * probability));
      }
      for(index_t c = 0; c < num_column_samples; c++) {
	psi_mat.set(s, c, c_mat->get(sample_row_number, c) /
		    sqrt(num_row_samples * probability));
      }
    }

    // Compute the pseudo-inverse of the outer-product using its SVD
    // expansion.
    Matrix phi_mat;
    phi_mat.Init(outer_product.n_rows(), outer_product.n_cols());
    phi_mat.SetZero();
    for(index_t s = 0; s < singular_values.length(); s++) {
      if(singular_values[s] > 0.01) {
	for(index_t c = 0; c < phi_mat.n_cols(); c++) {
	  for(index_t r = 0; r < phi_mat.n_rows(); r++) {
	    phi_mat.set(r, c, phi_mat.get(r, c) +
			1.0 / singular_values[s] *
			left_singular_vectors.get(r, s) *
			left_singular_vectors.get(c, s));
	  }
	}
      }
    } // end of iterating over each singular values...
    
    // Compute U, the rotation nmatrix that relates the row and column
    // subsets.
    la::MulTransBInit(phi_mat, psi_mat, u_mat);
  }

};

#endif
