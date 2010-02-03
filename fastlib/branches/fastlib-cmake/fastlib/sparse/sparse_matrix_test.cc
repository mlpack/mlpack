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
 *       Filename:  sparse_matrix_test.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  12/08/2007 03:50:44 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <stdio.h>
#include <limits>
#include <vector>
#include <map>
#include "fastlib/fastlib.h"
#include "fastlib/base/test.h"
#include "sparse_matrix.h"

class SparseMatrixTest {
 public:
  SparseMatrixTest() {
  }
  ~SparseMatrixTest() {
  }
  void Init() {
    for(index_t i=0; i<num_of_cols_; i++) {
      for(index_t j=0; j<num_of_rows_; j++) {
        mat_[i][j] = (i+j) * ((i+j) % 2);
      }
    }
  }
  
  void Destruct() {
    delete smat_;    
  }
  
  void TestInit1() {
    smat_ = new SparseMatrix(num_of_rows_, 
                             num_of_cols_, 
                             num_of_nnz_);
    smat_->StartLoadingRows();
    std::vector<index_t> ind;
    std::vector<double>  val;
    for(index_t i=0; i<num_of_cols_; i++) {
      ind.clear();
      val.clear();
      for(index_t j=0; j<num_of_cols_; j++) {
        if (mat_[i][j] != 0) {
          ind.push_back(j);
          val.push_back(mat_[i][j]);
        }
      }
      smat_->LoadRow(i, ind, val);
    }

    for(index_t i=0; i<num_of_rows_; i++) {
      for(index_t j=0; j<num_of_cols_; j++) {
        TEST_DOUBLE_APPROX(smat_->get(i,j), 
                           mat_[i][j], 
                           std::numeric_limits<double>::epsilon());
      }
    }
    NOTIFY("TestInit1 sucess!!\n");
  }
  
  void TestInit2() {
    std::vector<index_t> rows;
    std::vector<index_t> cols;
    std::vector<double>  vals;
    std::vector<index_t> nnz(num_of_rows_);
    for(index_t i=0; i<num_of_rows_; i++) {
      for(index_t j=0; j<num_of_cols_; j++) {
        if (mat_[i][j] != 0) {
          rows.push_back(i);
          cols.push_back(j);
          vals.push_back(mat_[i][j]);
          nnz[i]++;
        }
      }
    }
    /*for(index_t i=0; i<(index_t)rows.size(); i++) {
      printf("%i %i %lg\n",  
              rows[i],
             cols[i],
             vals[i]);
    }*/
    smat_ = new SparseMatrix();
    smat_->Init(rows, cols, vals, 
                *(std::max_element(nnz.begin(), nnz.end())), num_of_rows_);
    // printf("%s\n", smat_->Print().c_str());
    for(index_t i=0; i<num_of_rows_; i++) {
      for(index_t j=0; j<num_of_cols_; j++) {
        TEST_DOUBLE_APPROX(smat_->get(i,j), mat_[i][j],
                           std::numeric_limits<double>::epsilon());
      }
    }
    NOTIFY("TestInit2 success!!\n");
  }
  
  void TestInit3() {
    FILE *fp = fopen("temp.txt", "w");
    if (fp==NULL) {
      FATAL("Cannot open temp.txt error %s", strerror(errno));
    }
    for(index_t i=0; i<num_of_cols_; i++) {
      for(index_t j=0; j<num_of_cols_; j++) {
        if (mat_[i][j] != 0) {
          fprintf(fp, "%i %i %g\n", i, j, mat_[i][j]);
        }
      }
    }
    fclose(fp);
    smat_ = new SparseMatrix();
     smat_->Init("temp.txt");
    unlink("temp.txt");
    for(index_t i=0; i<num_of_rows_; i++) {
      for(index_t j=0; j<num_of_cols_; j++) {
        TEST_DOUBLE_APPROX(smat_->get(i,j), 
                           mat_[i][j], 
                           std::numeric_limits<double>::epsilon());
      }
    }
    NOTIFY("TestInit3 success!!");
  }
  
  void TestCopyConstructor() {
    smat_ = new SparseMatrix();
     NOTIFY("TestCopyConstructor success!!\n");
  }
  
  void TestMakeSymmetric() {
    TestInit1();
    smat_->set(2, 3, 1.44);
    smat_->set(3, 2, 0.74);
    smat_->set(7, 8, 4.33);
    smat_->set(8, 7, 0.22);
    smat_->MakeSymmetric();
    for(index_t i=0; i<num_of_rows_; i++) {
      for(index_t j=0; j<num_of_cols_; j++) {
        TEST_DOUBLE_APPROX(smat_->get(i,j), 
                           smat_->get(j, i),
                           std::numeric_limits<double>::epsilon());
      }
    }
    NOTIFY("Test MakeSymmetric success!!\n");
  }
  
  void TestNegate() {
    TestInit1();
    smat_->Negate();
    for(index_t i=0; i<num_of_rows_; i++) {
      for(index_t j=0; j<num_of_cols_; j++) {
        TEST_DOUBLE_APPROX(smat_->get(i,j), -mat_[i][j],
                           std::numeric_limits<double>::epsilon());
      }
    }
    NOTIFY("Test Negate success!!");
  }
  
  void TestColumnScale() {
    TestInit1();
    Vector scale;
    scale.Init(num_of_cols_);
    for(index_t i=0; i<num_of_cols_; i++) {
      scale[i]=i;
    }
    smat_->EndLoading();
    smat_->ColumnScale(scale);
    for(index_t i=0; i<num_of_rows_; i++) {
      for(index_t j=0; j<num_of_cols_; j++) {
        TEST_DOUBLE_APPROX(smat_->get(i,j), j*mat_[i][j],
                           std::numeric_limits<double>::epsilon());
      }
    }
    NOTIFY("Test ColumnScale success!!");
  }

  void TestRowScale() {
    TestInit1();
    Vector scale;
    scale.Init(num_of_cols_);
    for(index_t i=0; i<num_of_cols_; i++) {
      scale[i]=i;
    }
    smat_->EndLoading();
    smat_->RowScale(scale);
    for(index_t i=0; i<num_of_rows_; i++) {
      for(index_t j=0; j<num_of_cols_; j++) {
        TEST_DOUBLE_APPROX(smat_->get(i,j), i*mat_[i][j],
                           std::numeric_limits<double>::epsilon());
      }
    }
    NOTIFY("Test RowScale success!!");
  }
  
  void TestRowSums() {
    TestInit1();
    Vector row_sums;
    smat_->EndLoading();
    smat_->RowSums(&row_sums);
    for(index_t i=0; i<num_of_rows_; i++) {
      double row_sum=0;
      for(index_t j=0; j<num_of_cols_; j++) {
        row_sum+=mat_[i][j];
      }
      TEST_DOUBLE_APPROX(row_sums[i], row_sum, 0.001);
    } 
    NOTIFY("Test RowSums success!!");
  }
  
  void TestInvRowSums() {
    TestInit1();
    Vector row_sums;
    smat_->EndLoading();
    smat_->InvRowSums(&row_sums);
    for(index_t i=0; i<num_of_rows_; i++) {
      double row_sum=0;
      for(index_t j=0; j<num_of_cols_; j++) {
        row_sum+=mat_[i][j];
      }
      TEST_DOUBLE_APPROX(row_sums[i], 1.0/row_sum ,
                           std::numeric_limits<double>::epsilon());
    } 
    NOTIFY("Test InvRowSums success!!");
  }
  
  void TestInvColMaxs() {
    TestInit1();
    Vector col_maxs;
    smat_->EndLoading();
    smat_->InvColMaxs(&col_maxs);
    for(index_t i=0; i<num_of_cols_; i++) {
      double col_max=0;
      for(index_t j=0; j<num_of_rows_; j++) {
        col_max=max(col_max, mat_[j][i]);
      }
      TEST_DOUBLE_APPROX(col_maxs[i], 1.0/col_max,
          std::numeric_limits<double>::epsilon());
    } 
    NOTIFY("Test InvColMaxs success!!");
  
  }
  
  void TestEig() {
    TestInit1();
    smat_->EndLoading();
    Vector eigvalues_real;
    Vector eigvalues_imag;
    Matrix eigvectors;
    smat_->Eig(1, "LM", &eigvectors, &eigvalues_real, &eigvalues_imag);
    // eigvectors.PrintDebug();
    NOTIFY("Test Eigenvector success!!\n");
  }
  
  void TestLinSolve() {
    TestInit1();
    Vector b,x;
    b.Init(num_of_cols_);
    b.SetZero();
    x.Init(num_of_cols_);
    x.SetAll(1);
    smat_->MakeSymmetric();
    smat_->EndLoading();
    smat_->LinSolve(b, &x);
    x.PrintDebug();
    NOTIFY("Test Linear Solve success!!\n");
  }
  void TestBasicOperations() {
    SparseMatrix a("A.txt");
    a.EndLoading();
    SparseMatrix b("B.txt");
    b.EndLoading();
    SparseMatrix a_plus_b("AplusB.txt");
    SparseMatrix a_minus_b("AminusB.txt");
    SparseMatrix a_times_b("AtimesB.txt");
    SparseMatrix a_dot_times_b("AdottimesB.txt");

    SparseMatrix temp;
    Sparsem::Add(a, b, &temp);
    temp.EndLoading();  
//    a_plus_b.EndLoading();
//    printf("%s\n", temp.Print().c_str());
//    printf("%s\n", a_plus_b.Print().c_str());
    for(index_t i=0; i<20; i++) {
      for(index_t j=0; j<20; j++) {
        TEST_DOUBLE_APPROX(a_plus_b.get(i,j), temp.get(i,j), 0.01);
      }
    }
    temp.Destruct();
    NOTIFY("Matrix addition sucess!!\n");
    
    Sparsem::Subtract(a, b, &temp);
    temp.EndLoading();
    a_minus_b.EndLoading();
    for(index_t i=0; i<21; i++) {
      for(index_t j=0; j<21; j++) {
        TEST_DOUBLE_APPROX(a_minus_b.get(i,j), temp.get(i,j), 0.01);
      }
    }
    temp.Destruct();
    NOTIFY("Matrix subtraction success!!\n");
    
    Sparsem::Multiply(a, b, &temp);
//    printf("%s\n", temp.Print().c_str());
//    printf("%s\n", a_times_b.Print().c_str());
    temp.EndLoading();
    a_times_b.EndLoading();
    for(index_t i=0; i<21; i++) {
      for(index_t j=0; j<21; j++) {
        TEST_DOUBLE_APPROX(a_times_b.get(i,j), temp.get(i,j), 0.01);
      }
    }
    temp.Destruct();
    NOTIFY("Matrix multiplication success!!\n");
    
    SparseMatrix i_w("I-W.txt");
    SparseMatrix i_w_i_w_trans("I-W_time_I-W_trans.txt");
    i_w.SortIndices();
    Sparsem::MultiplyT(i_w, &temp);
//  printf("%s\n", temp.Print().c_str());
//  printf("%s\n", a_times_a_trans.Print().c_str());
    temp.EndLoading();
    i_w_i_w_trans.EndLoading();
//  printf("%s\n", i_w_i_w_trans.Print().c_str());
    for(index_t i=0; i<i_w.num_of_rows(); i++) {
      for(index_t j=0; j<i_w.num_of_columns(); j++) {
        TEST_DOUBLE_APPROX(i_w_i_w_trans.get(i,j), temp.get(i,j), 0.01);
      }
    }
    temp.Destruct();
    NOTIFY("Matrix multiplication success!!\n");

    
    Sparsem::DotMultiply(a, b, &temp);
//    printf("%s\n", temp.Print().c_str());
//    printf("%s\n", a_dot_times_b.Print().c_str());
    temp.EndLoading();
    a_dot_times_b.EndLoading();

    for(index_t i=0; i<a_dot_times_b.num_of_rows(); i++) {
      for(index_t j=0; j<a_dot_times_b.num_of_columns(); j++) {
        TEST_DOUBLE_APPROX(a_dot_times_b.get(i,j), temp.get(i,j), 0.01);
      }
    }
    temp.Destruct();
    NOTIFY("Matrix dot multiplication success!!\n");
    
    Sparsem::Multiply(a, 3.45, &temp);
    for(index_t i=0; i<21; i++) {
      for(index_t j=0; j<21; j++) {
        TEST_DOUBLE_APPROX(3.45 * a.get(i,j), temp.get(i,j), 
                           std::numeric_limits<double>::epsilon());
      }
    }
    temp.Destruct();
    NOTIFY("Matrix scalar multiplication success!!\n");
    
  }
  void TestAll() {
    Init();
    TestInit1();
    Destruct();
    Init();
    TestInit2();
    Destruct();
    Init();
    TestInit3();
    Destruct();
    Init();
    TestCopyConstructor();
    Destruct();
    Init();
    TestMakeSymmetric();
    Destruct();
    Init();
    TestNegate();
    Destruct();
    Init();
    TestColumnScale();
    Destruct();
    Init();
    TestRowScale();
    Destruct();
    Init();
    TestRowSums();
    Destruct();
    Init();
    TestInvRowSums(); 
    Destruct();
    Init();
    TestInvColMaxs();
    Destruct();
    Init();
    TestEig();
    Destruct();
    Init();
    TestLinSolve();
    Destruct();
    Init();
    TestBasicOperations();
  }
    
 private:
  SparseMatrix *smat_;
  static const index_t num_of_cols_  = 80;
  static const index_t num_of_rows_  = 80;
  static const index_t num_of_nnz_   = 4;
  double               mat_[num_of_rows_][num_of_cols_];
  std::vector<index_t>  indices_;
  std::vector<index_t>  rows_;
};

int main() {
  SparseMatrixTest test;
  test.TestAll();
}
