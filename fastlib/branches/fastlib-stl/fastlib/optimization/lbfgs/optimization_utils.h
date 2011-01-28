/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
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

#ifndef OPTIMIZATION_UTILS_H_
#define OPTIMIZATION_UTILS_H_

#include "fastlib.h"

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
  static void BoundProjection(Matrix *data, double lo, double hi) {
    double *ptr=data->ptr();
    for(index_t i=0; i<(index_t)data->n_elements(); i++) {
      if (ptr[i]>hi) {
        ptr[i]=hi;
        continue;
      }
      if (ptr[i]<lo) {
        ptr[i]=lo;
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
      double energy_kept=0;
      double total_energy=0;
      for(index_t i=0; i<components_to_keep; i++) {
       NOTIFY("%lg ", s[i]);
        energy_kept+=s[i];
      }
      printf("\n");
      for(index_t i=0; i<s.length(); i++) {
        total_energy+=s[i];
      }
      NOTIFY("Kept %lg%% of the energy", energy_kept*100/total_energy);
    }
    
    Vector s_chopped;
    s.MakeSubvector(0, components_to_keep, &s_chopped);
//  Matrix temp_U(components_to_keep, U.components_to_keep);
    Matrix temp_VT(components_to_keep, VT.n_cols());
    for(index_t i=0; i<temp_VT.n_cols(); i++) {
      memcpy(temp_VT.GetColumnPtr(i), 
          VT.GetColumnPtr(i), components_to_keep*sizeof(double));  
    }
   
    la::ScaleRows(s_chopped, &temp_VT);
    output_mat->Own(&temp_VT);
//    la::MulInit(temp_U, temp_VT, output_mat);
    /*
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
    for(index_t i=0; i<output_mat->n_rows(); i++) {
      for(index_t j=0; j<output_mat->n_cols(); j++) {
        error+=math::Sqr(output_mat->get(i,j)-input_mat.get(i,j));
      }
    }
    NOTIFY("Reconstruction error : %lg", error);
    */
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
    double precomputed_sparse_factor=-sparse_factor*(std::sqrt(dimension)-1)+
          std::sqrt(dimension);

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
        double alpha = (-w_times_v+std::sqrt(w_times_v*w_times_v
            -4*w_norm*v_norm_minus_l2))/(2*w_norm);
        la::AddExpert(alpha, w_vector, &v_vector);
        bool all_positive=true;
        zero_coeff.Clear();
        double v_sum=0;
        for(index_t j=0; j<dimension; j++) {
          if (v_vector[j]<0) {
            all_positive=false;
            zero_coeff.PushBackCopy(j);
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

#endif // OPTIMIZATION_UTILS_H_

