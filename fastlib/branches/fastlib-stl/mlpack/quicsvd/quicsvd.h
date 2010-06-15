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
  std::vector<Vector> basis_;

  /** Projection coordinates of columns of A_ onto the subspace */
  std::vector<Vector> UTA_;
  
  /** Projection magnitude of columns of A_ in the subspace */
  std::vector<double> projMagSq_;

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
