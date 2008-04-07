/*
 * =====================================================================================
 * 
 *       Filename:  optimization_utils.h
 * 
 *    Description
 * 
 *        Version:  1.0
 *        Created:  03/12/2008 06:29:51 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
#include "fastlib/fastlib.h"

class OptUtils {
 public:
  static void RemoveMean(Matrix *data) {
    index_t dimension=data->n_rows();
    index_t num_of_points=data->n_cols();
    Vector mean;
    mean.Init(dimension);
    mean.SetAll(0);
    for(index_t i=0; i<num_of_points; i++){
      la::AddTo(dimension, data->GetColumnPtr(i), mean.ptr());
    }
    la::Scale(-1.0/num_of_points, &mean);
    for(index_t i=0; i<num_of_points; i++){
      la::AddTo(dimension, mean.ptr(), data->GetColumnPtr(i));
    }
  }
  
  static void NonNegativeProjection(Matrix *data) {
    double *ptr=data->ptr();
    for(index_t i=0; i<(index_t)data->n_elements(); i++) {
      if (ptr[i]<0) {
        ptr[i]=0;
      }
    }
  }
 
  static success_t SVDTransform(Matrix &input_mat, Matrix *output_mat, 
      index_t components_to_keep) {
    Matrix temp;
    temp.Copy(input_mat);
    RemoveMean(&temp);
    Vector s;
    Matrix U, VT;
    success_t success=la::SVDInit(temp, &s, &U, &VT);
    if (success==SUCCESS_PASS) {
      NOTIFY("PCA successful !! Printing requested %i eigenvalues...",
          components_to_keep);
      for(index_t i=0; i<components_to_keep; i++) {
        printf("%lg ", s[i]);
      }
      printf("\n");
    }
    output_mat->Init(components_to_keep, input_mat.n_cols());
    Matrix temp_reconstructed;
    Matrix temp_S;
    temp_S.Init(input_mat.n_rows(), input_mat.n_rows());
    temp_S.SetAll(0.0);
    for(index_t i=0; i<components_to_keep; i++) {
      temp_S.set(i, i, s[i]);
    }
    Matrix temp_U;
    la::MulInit(U, temp_S, &temp_U);
    la::MulInit(temp_U, VT, &temp_reconstructed);
    for(index_t i=0; i<output_mat->n_cols(); i++) {
      memcpy(output_mat->GetColumnPtr(i), 
          temp_reconstructed.GetColumnPtr(i), components_to_keep*sizeof(double));  
    }
    double error=0;
    for(index_t i=0; i<temp_reconstructed.n_rows(); i++) {
      for(index_t j=0; j<temp_reconstructed.n_cols(); j++) {
        error+=math::Sqr(temp_reconstructed.get(i,j)-temp.get(i,j));
         error+=math::Sqr(input_mat.get(i,j)-temp.get(i,j));
      }
    }
    NOTIFY("Reconstruction error : %lg", error);
    return success;
  }
  
  static void SparseProjection(Matrix *data, double sparse_factor) {
    DEBUG_ASSERT(sparse_factor<=1);
    DEBUG_ASSERT(sparse_factor>=0);
    index_t dimension=data->n_rows();
    ArrayList<index_t> zero_coeff;
    zero_coeff.Init();
    Vector w_vector;
    w_vector.Init(dimension);
    Vector v_vector;
    v_vector.Init(dimension);  
    Vector a_vector;
    Vector ones;
    ones.Init(dimension);
    ones.SetAll(1.0);
    // This part of the sparsity constraint function formula can be
    // precomputed and it is the same for every iteration
    double precomputed_sparse_factor=-sparse_factor*(math::Pow<1,2>(dimension)-1)+
          math::Pow<1,2>(dimension);

    for (index_t i=0; i<data->n_cols(); i++) {
      double *point=data->GetColumnPtr(i);
      double l2_norm=la::LMetric<2>(dimension,
         point, point);
      double l1_norm=precomputed_sparse_factor*l2_norm;
      // (L1-\sum x_i)/dimension
      double factor1=l1_norm;
      for(index_t j=0; j<dimension; j++) {
        factor1-=point[j];
      }
      factor1/=dimension;
      for(index_t j=0; j<dimension; j++) {
        v_vector[j]+=factor1;
      }
      zero_coeff.Clear();
      Vector midpoint;
      midpoint.Init(dimension);
      while(true) {
       midpoint.SetAll(l1_norm/(dimension-zero_coeff.size()));
        for(index_t j=0; j<zero_coeff.size(); j++) {
          midpoint[zero_coeff[j]]=0.0;
        }
        la::SubOverwrite(midpoint, v_vector, &w_vector);
        double w_norm=la::LengthEuclidean(w_vector);
        double w_times_v = 2*la::Dot(v_vector, w_vector);
        double v_norm_minus_l2=la::LengthEuclidean(v_vector)-l2_norm;
        double alpha = (-w_times_v+math::Pow<1,2>(w_times_v*w_times_v
            -4*w_norm*v_norm_minus_l2))/(2*w_norm);
        la::AddExpert(alpha, w_vector, &v_vector);
        bool all_positive=true;
        zero_coeff.Clear();
        double v_sum=0;
        for(index_t j=0; j<dimension; j++) {
          if (v_vector[j]<0) {
            all_positive=false;
            zero_coeff.AddBack(j);
            v_vector[j]=0;
          } else {
            v_sum+=v_vector[j];
          } 
        }
        if (all_positive==true) {
          break;
        }
        double temp=(l1_norm-v_sum)/(dimension-zero_coeff.size());
        la::AddExpert(temp, ones, &v_vector);
        for(index_t j=0; j<zero_coeff.size(); j++) {
          v_vector[zero_coeff[j]]=0;
        }
      }
    }
  }

};
