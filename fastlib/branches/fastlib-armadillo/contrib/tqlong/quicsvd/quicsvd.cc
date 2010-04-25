/** 
 *  @file quicsvd.cc
 *
 *  This file implements QuicSVD class
 * 
 *  @see quicsvd.h
 */
#include "quicsvd.h"
#include <cmath>

QuicSVD::QuicSVD(const Matrix& A, double targetRelErr) : root_(A) {
  A_.Alias(A);

  basis_.Init(); // init empty basis
  UTA_.Init();   // and empty projection, too
  projMagSq_.InitRepeat(0, n_cols()); // projected magnitude squares are zero's
  sumProjMagSq_ = 0;
  targetRelErr_ = targetRelErr;
  dataNorm2_ = root_.getSumL2(); // keep the Frobenius norm of A
  //printf("Done Init ncols = %d projMagSq_ = %d\n", n_cols(), projMagSq_.size());
}

// Modified Gram-Schmidt method, calculate the orthogonalized
// new basis vector
bool MGS(const ArrayList<Vector>& basis, const Vector& newVec, 
	 Vector* newBasisVec) {
  //ot::Print(basis, "basis", stdout);
  //ot::Print(newVec, "newVec", stdout);
  newBasisVec->Copy(newVec);
  for (index_t i = 0; i < basis.size(); i++) {
    double prod = - la::Dot(basis[i], *newBasisVec);
    la::AddExpert(prod, basis[i], newBasisVec);
  }
  //ot::Print(*newBasisVec, "--newBasisVec--", stdout);

  // check of the remaning vector is zero vector (dependent)
  // if not, normalize it. Otherwise, signal the caller
  double L2 = la::LengthEuclidean(*newBasisVec);
  if (L2 > 1e-12) {
    la::Scale(1.0/L2, newBasisVec);
    return true;
  }
  else
    return false;
}

void QuicSVD::addBasisFrom(const CosineNode& node) {
  Vector nodeBasis;
  // check if new vector are independent and orthogonalize it
  if (MGS(basis_, node.getMean(), &nodeBasis)) {
    Vector av;
    basis_.PushBackCopy(nodeBasis);  // add to the basis
    la::MulInit(nodeBasis, A_, &av); // calculate projection of A
    UTA_.PushBackCopy(av);           // on the new basis vector and save
    for (index_t i_col = 0; i_col < n_cols(); i_col++) {
      double magSq = math::Sqr(av[i_col]);  // magnitude square of the i-th
      projMagSq_[i_col] += magSq;           // column of A in the new subspace
      sumProjMagSq_ += magSq;               // spanned by the basis is increased
    }
  }
}

// Calculate relative error |A|-|A'|/|A|
double QuicSVD::curRelErr() {
  return sqrt((dataNorm2_ - sumProjMagSq_) / dataNorm2_);
}

// Add a node to the priority queue after calculating its Frobenius error
// when projected on the subspace span by the basis
void QuicSVD::addToQueue(CosineNode* node) {
  if (node == NULL) return;
  node->setL2Err(calL2Err(*node));
  leaves_.push(node);
}

// Calculate Frobenius error of a node when projected on the subspace
// spanned by the current basis
double QuicSVD::calL2Err(const CosineNode& node) {
  double nodeSumL2 = node.getSumL2();
  double nodeSumProjL2 = 0;
  for (index_t i_col = 0; i_col < node.n_cols(); i_col++)
    nodeSumProjL2 += projMagSq_[node.getOrigIndex(i_col)];
  return nodeSumL2 - nodeSumProjL2;
}

// Compute a subspace such that when the Frobenius relative error 
// of matrix A when projected onto this subspace is less than the
// target relative error
void QuicSVD::ComputeSVD(Vector* s, Matrix* U, Matrix* VT) {
  leaves_.push(&root_);  // the root is the first element of the queue
  addBasisFrom(root_);   // at begining, the basis has at most 1 vector

  //printf("Add Root\n");

  //ot::Print(curRelErr(), "curRelErr = ", stdout);
  while (curRelErr() > targetRelErr_) {
    CosineNode* node = leaves_.top(); 
    leaves_.pop();  // pop the top of the queue

    node->Split();  // split it

    // only add the basis from the right node as the left node 
    // mean vector is well represented by the parent
    if (node->hasRight()) addBasisFrom(*(node->getRight()));
    
    // add both left child and right child to the queue
    addToQueue(node->getLeft());
    addToQueue(node->getRight());
    
    //ot::Print(curRelErr(), "curRelErr = ", stdout);
  } 
  //ot::Print(basis_, "basis", stdout);

  // after achieve the desire subspace, SVD on this subspace
  extractSVD(s, U, VT);
}

// Compute a square matrix UTA2 = UTA*UTA' that has the same
// left singular vectors with UTA and has singluar values that are 
// squares of singular values of UTA.
// SVD on UTA2 is more efficient as it is a square matrix 
// with smaller dimension
void createUTA2(const ArrayList<Vector>& UTA, Matrix* UTA2) {
  UTA2->Init(UTA.size(), UTA.size());
  for (int i = 0; i < UTA.size(); i++)
    for (int j = 0; j < UTA.size(); j++)
      UTA2->ref(i, j) = la::Dot(UTA[i], UTA[j]);
}

// Matrix multiplication of a list of vector and a matrix
// C = AB
void MulInit(const ArrayList<Vector>& A, const Matrix& B,
	     Matrix* C) {
  index_t m = A[0].length();
  index_t n = A.size();
  index_t p = B.n_cols();
  C->Init(m, p);
  for (index_t i = 0; i < m; i++)
    for (index_t j = 0; j < p; j++) {
      C->ref(i, j) = 0;
      for (index_t k = 0; k < n; k++) 
	C->ref(i, j) += A[k][i] * B.get(k, j);
    }
}

// Matrix multiplication C = B' A'
void MulTransCInit(const ArrayList<Vector>& A, const Matrix& B,
		   Matrix* C) {
  index_t m = A[0].length();
  index_t n = A.size();
  index_t p = B.n_cols();
  C->Init(p, m);
  for (index_t i = 0; i < m; i++)
    for (index_t j = 0; j < p; j++) {
      C->ref(j, i) = 0;
      for (index_t k = 0; k < n; k++) 
	C->ref(j, i) += A[k][i] * B.get(k, j);
    }
}

// Inverse scale the rows of a matrix
void InverseRowScale(const Vector& s, Matrix* A) {
  for (index_t i = 0; i < A->n_rows(); i++)
    for (index_t j = 0; j < A->n_cols(); j++)
      A->ref(i, j) /= s[i];
}

// Extract SVD of a matrix when projected to a subspace
// spanned by the basis
void QuicSVD::extractSVD(Vector* s, Matrix* U, Matrix* VT) {
  //ot::Print(UTA_, "UTA", stdout);

  Matrix UTA2;
  Matrix Uprime, VprimeT;
  Vector sprime;

  // as we only need the left singular vectors,
  // we will SVD on UTA2 = UTA_ UTA_'
  createUTA2(UTA_, &UTA2);  
  
  la::SVDInit(UTA2, &sprime, &Uprime, &VprimeT);
  
  s->Init(sprime.length());
  for (index_t i = 0; i < sprime.length(); i++)
    (*s)[i] = sqrt(sprime[i]); // the original singular values are square roots
                               // of that of UTA2
  
  MulInit(basis_, Uprime, U);  // transform back to the original space
  MulTransCInit(UTA_, Uprime, VT);
  InverseRowScale(*s, VT);

  //ot::Print(*U, "U=", stdout);
  //ot::Print(*s, "s=", stdout);
  //ot::Print(*VT, "VT=", stdout);
}


// Init matrices by Singular Value Decomposition, 
// mimic la::SVDInit(A, s, U, VT) interface
double QuicSVD::SVDInit(const Matrix& A, double targetRelErr,
		      Vector* s, Matrix* U, Matrix* VT) {
  // check if we need to transpose A to save some computation
  bool transpose = A.n_rows() > A.n_cols() * 1.1;

  if (!transpose) {
    QuicSVD svd(A, targetRelErr);
    svd.ComputeSVD(s, U, VT);
    return svd.curRelErr();
  }
  else {
    Matrix AT, UT, V;
    la::TransposeInit(A, &AT);
    QuicSVD svd(AT, targetRelErr);
    svd.ComputeSVD(s, &V, &UT);
    la::TransposeInit(UT, U);
    la::TransposeInit(V, VT);
    return svd.curRelErr();
  }
}
