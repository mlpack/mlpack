#ifndef MATRIX_UTIL_H
#define MATRIX_UTIL_H

#include "fastlib/fastlib.h"

class MatrixUtil {

  public:

    static double L1Norm(const Matrix &m_mat) {

      double l1_norm = 0;
      for(index_t j = 0; j < m_mat.n_cols(); j++) {
	const double *m_mat_column = m_mat.GetColumnPtr(j);
	double tmp_l1_norm = 0;

	for(index_t i = 0; i < m_mat.n_rows(); i++) {
	  tmp_l1_norm += fabs(m_mat_column[i]);
	}
	l1_norm = std::max(l1_norm, tmp_l1_norm);
      }
      return l1_norm;
    }

    static double L1Norm(const Vector &v_vec) {
      
      double l1_norm = 0;
      for(index_t d = 0; d < v_vec.length(); d++) {
	l1_norm += fabs(v_vec[d]);
      }
      
      return l1_norm;
    }

    static double EntrywiseLpNorm(const Matrix &m_mat, int p) {

      double lp_norm = 0;
      for(index_t j = 0; j < m_mat.n_cols(); j++) {
	const double *m_mat_column = m_mat.GetColumnPtr(j);

	for(index_t i = 0; i < m_mat.n_rows(); i++) {
	  lp_norm += pow(fabs(m_mat_column[i]), p);
	}
      }
      return lp_norm;
    }

    static double EntrywiseLpNorm(int length, const double *v_arr, int p) {      
      double lp_norm = 0;
      for(index_t d = 0; d < length; d++) {
	lp_norm += pow(fabs(v_arr[d]), p);
      }
      
      return lp_norm;
    }

    static double EntrywiseLpNorm(const Vector &v_vec, int p) {
      
      double lp_norm = 0;
      for(index_t d = 0; d < v_vec.length(); d++) {
	lp_norm += pow(fabs(v_vec[d]), p);
      }
      
      return lp_norm;
    }

};

#endif
