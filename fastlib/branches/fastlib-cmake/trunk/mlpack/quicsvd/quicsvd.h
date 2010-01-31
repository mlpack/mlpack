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
/**
 * @file quicsvd.h
 *
 * This file implements the interface of QuicSVD class
 * It approximate the original matrix by another matrix
 * with smaller dimension to a certain degree specified by the 
 * user and then make SVD decomposition in the projected supspace.
 *
 * Run with --help for more usage.
 * 
 * @see quicsvd.cc
 */

#ifndef QUICSVD_QUICSVD_H
#define QUICSVD_QUICSVD_H
#include <fastlib/fastlib.h>
#include "cosine_tree.h"
#include <vector>
#include <queue>

class QuicSVD {

  // An alias for the priority queue of CosineNode
  typedef std::priority_queue<CosineNode*, std::vector<CosineNode*>,
    CompareCosineNode> CosineNodeQueue;

  /** Alias matrix to the original matrix */
  Matrix A_;

  /** Frobenius norm square of A_ */
  double dataNorm2_;

  /** Root of the cosine tree */
  CosineNode root_;

  /** Priority queue of cosine node waiting to be split
   *  in the decreasing order L2 error norm with respect to
   *  the original matrix A_
   */
  CosineNodeQueue leaves_;


  /** Orthonomal basis of the subspace */
  ArrayList<Vector> basis_;

  /** Projection coordinates of columns of A_ onto the subspace */
  ArrayList<Vector> UTA_;
  
  /** Projection magnitude of columns of A_ in the subspace */
  ArrayList<double> projMagSq_;

  /** Frobenius norm of A_ being projected to the subspace */
  double sumProjMagSq_;

  /** Target relative error |A-A'|/|A| */
  double targetRelErr_;

 public:
  /** Constructor with original matrix and target relative error */
  QuicSVD(const Matrix& A, double targetRelErr);
  
  /** Get the number of columns of A
   *  The method is implelemented in column-wise manner
   *  as in FASTLib, this is more convenient and efficient.
   */
  index_t n_cols() {
    return A_.n_cols();
  }

  /** Compute and initialize the vector of singular values s,
   *  orthogonal matrices U, VT such that A ~ U diag(s) VT
   *  while achieving the target relative error
   */
  void ComputeSVD(Vector* s, Matrix* U, Matrix* VT);

  /** Static method to compute the QUIC-SVD directly from
   *  the original matrix and a specified relative error.
   *  It mimics the same syntax of la::SVDInit(A, s, U, VT)
   *
   *  Return: actual relative norm difference
   */
  static double SVDInit(const Matrix& A, double targetRelErr,
		      Vector* s, Matrix* U, Matrix* VT);

 private:
  // Helper private functions

  /** Add a vector representing the cosine node 
   *  into the basis and orthogolize it using modifed 
   *  modified Gram-Schmidt method.
   */
  void addBasisFrom(const CosineNode& node);

  /** Return current relative error |A'-A|/|A| */
  double curRelErr();

  /** Compute the L2 error and add a cosine node to the priority queue */
  void addToQueue(CosineNode* node);

  /** Calculate L2 error of a cosine node with respect to the
   *  the original matrix A 
   */
  double calL2Err(const CosineNode& node);

  /** Extract SVD in the subspace of basis after the target 
   *  relative error has been achieved */
  void extractSVD(Vector* s, Matrix* U, Matrix* VT);

  /** Friend unit test class */
  friend class QuicSVDTest;
};

class QuicSVDTest {
  FILE * logfile;

  void test_QuicSVD() {
    Matrix A;
    printf("Load data from input1.txt.\n");
    data::Load("input1.txt", &A);
    QuicSVD svd(A, 0.1);
    Vector s;
    Matrix U, VT, S;
    svd.ComputeSVD(&s, &U, &VT);
    data::Save("U.txt", U);
    S.InitDiagonal(s);
    data::Save("S.txt", S);
    data::Save("VT.txt", VT);
    printf("U,s,VT are saved to U.txt, S.txt, VT.txt, respectively.\n");
  }
public:
  QuicSVDTest() {
    logfile = fopen("LOG", "w");
  }
  ~QuicSVDTest() {
    fclose(logfile);
  }

  void run_tests() {
    test_QuicSVD();
  }

};

#endif
