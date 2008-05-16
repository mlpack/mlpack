#ifndef CUR_DECOMPOSITION_H
#define CUR_DECOMPOSITION_H

#include "fastlib/fastlib.h"

class CURDecomposition {

 private:
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
  static void Compute(const Matrix &a_mat, Matrix *c_mat, Matrix *u_mat, 
		      Matrix *r_mat) {
    
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

    // Pick samples from the column distribution to form the matrix C.
    int num_column_samples = 2 * ((int) sqrt(a_mat.n_cols()));
    c_mat->Init(a_mat.n_rows(), num_column_samples);
    for(index_t s = 0; s < num_column_samples; s++) {
      double random_number = 
	math::Random(0, column_length_square_distribution[a_mat.n_cols() - 1]);
      index_t sample_column_number = 
	FindBinNumber_(column_length_square_distribution, random_number);
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
    int num_row_samples = 2 * ((int) sqrt(a_mat.n_rows()));
    r_mat->Init(num_row_samples, a_mat.n_cols());
    Matrix psi_mat;
    psi_mat.Init(num_row_samples, num_column_samples);
    
    for(index_t s = 0; s < num_row_samples; s++) {
      double random_number = 
	math::Random(0, row_length_square_distribution[a_mat.n_rows() - 1]);
      index_t sample_row_number = 
	FindBinNumber_(row_length_square_distribution, random_number);
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
