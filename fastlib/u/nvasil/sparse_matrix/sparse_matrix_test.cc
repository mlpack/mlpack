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
#include "u/nvasil/test/test.h"
#include "u/nvasil/sparse_matrix/sparse_matrix.h"

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
		NONFATAL("TestInit1 sucess!!\n");
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
		NONFATAL("TestInit2 success!!\n");
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
		NONFATAL("TestInit3 success!!");
	}
	void TestCopyConstructor() {
	  smat_ = new SparseMatrix();
	 	NONFATAL("TestCopyConstructor success!!\n");
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
		NONFATAL("Test MakeSymmetric success!!\n");
	}
  void TestEig() {
	  TestInit1();
		smat_->EndLoading();
		std::vector<double> eigvalues_real;
	  std::vector<double> eigvalues_imag;
		Matrix eigvectors;
		smat_->Eig(1, "LM", &eigvectors, &eigvalues_real, &eigvalues_imag);
    eigvectors.PrintDebug();
		NONFATAL("Test Eigenvector success!!\n");
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
    NONFATAL("Test Linear Solve success!!\n");
	}
	void TestBasicOperations() {
		SparseMatrix a("A.txt");
		SparseMatrix b("B.txt");
		SparseMatrix a_plus_b("AplusB.txt");
		SparseMatrix a_minus_b("AminusB.txt");
		SparseMatrix a_times_b("AtimesB.txt");
		SparseMatrix a_dot_times_b("AdottimesB.txt");
		SparseMatrix temp;
		temp.Init(20,20, 3);
		Sparsem::Add(a, b, &temp);
		for(index_t i=0; i<20; i++) {
		  for(index_t j=0; j<20; j++) {
			  TEST_DOUBLE_APPROX(a_plus_b.get(i,j), temp.get(i,j), 
						               std::numeric_limits<double>::epsilon());
			}
		}
		temp.Destruct();
		NONFATAL("Matrix addition sucess!!\n");
		temp.Init(20,20, 3);
		Sparsem::Subtract(a, b, &temp);
		for(index_t i=0; i<20; i++) {
		  for(index_t j=0; j<20; j++) {
			  TEST_DOUBLE_APPROX(a_minus_b.get(i,j), temp.get(i,j), 
						               std::numeric_limits<double>::epsilon());
			}
		}
		temp.Destruct();
		NONFATAL("Matrix subtraction success!!\n");
    temp.Init(20,20, 3);
		Sparsem::Multiply(a, b, &temp);
		for(index_t i=0; i<20; i++) {
		  for(index_t j=0; j<20; j++) {
			  TEST_DOUBLE_APPROX(a_times_b.get(i,j), temp.get(i,j), 
						               std::numeric_limits<double>::epsilon());
			}
		}
		temp.Destruct();
		NONFATAL("Matrix multiplicationn success!!\n");
    temp.Init(20,20, 3);
		Sparsem::DotMultiply(a, b, &temp);
		for(index_t i=0; i<20; i++) {
		  for(index_t j=0; j<20; j++) {
			  TEST_DOUBLE_APPROX(a_dot_times_b.get(i,j), temp.get(i,j), 
						               std::numeric_limits<double>::epsilon());
			}
		}
		temp.Destruct();
		NONFATAL("Matrix dot multiplicationn success!!\n");
    temp.Init(20,20, 3);
		Sparsem::Multiply(a, 3.45, &temp);
		for(index_t i=0; i<20; i++) {
		  for(index_t j=0; j<20; j++) {
			  TEST_DOUBLE_APPROX(a.get(i,j), 3.45*temp.get(i,j), 
						               std::numeric_limits<double>::epsilon());
			}
		}
		temp.Destruct();
		NONFATAL("Matrix scalar multiplicationn success!!\n");
		
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
		TestEig();
		Destruct();
		Init();
		TestLinSolve();
		Destruct();
		Init();
		TestBasicOperations();
		Destruct();
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
