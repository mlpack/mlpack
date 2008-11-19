
#include "quicsvd.h"
#include <cmath>

QuicSVD::QuicSVD(const Matrix& A, double targetRelErr) : root_(A) {
  A_.Alias(A);
  
  basis_.Init();
  UTA_.Init();
  projMagSq_.InitRepeat(0, n_cols());
  sumProjMagSq_ = 0;
  targetRelErr_ = targetRelErr;
  dataNorm2_ = root_.getSumL2();
  //printf("Done Init ncols = %d projMagSq_ = %d\n", n_cols(), projMagSq_.size());
}

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
  if (MGS(basis_, node.getMean(), &nodeBasis)) {
    Vector av;
    basis_.PushBackCopy(nodeBasis);
    la::MulInit(nodeBasis, A_, &av);
    UTA_.PushBackCopy(av);
    for (index_t i_col = 0; i_col < n_cols(); i_col++) {
      double magSq = math::Sqr(av[i_col]);
      projMagSq_[i_col] += magSq;
      sumProjMagSq_ += magSq;
    }
  }
}

double QuicSVD::curRelErr() {
  return sqrt((dataNorm2_ - sumProjMagSq_) / dataNorm2_);
}

void QuicSVD::addToQueue(CosineNode* node) {
  if (node == NULL) return;
  node->setL2Err(calL2Err(*node));
  leaves_.push(node);
}

double QuicSVD::calL2Err(const CosineNode& node) {
  double nodeSumL2 = node.getSumL2();
  double nodeSumProjL2 = 0;
  for (index_t i_col = 0; i_col < node.n_cols(); i_col++)
    nodeSumProjL2 += projMagSq_[node.getOrigIndex(i_col)];
  return nodeSumL2 - nodeSumProjL2;
}

void QuicSVD::ComputeSVD(Vector* s, Matrix* U, Matrix* VT) {
  leaves_.push(&root_);
  addBasisFrom(root_);

  //printf("Add Root\n");

  //ot::Print(curRelErr(), "curRelErr = ", stdout);
  while (curRelErr() > targetRelErr_) {
    CosineNode* node = leaves_.top(); leaves_.pop();
    node->Split();

    if (node->hasRight()) addBasisFrom(*(node->getRight()));
    
    addToQueue(node->getLeft());
    addToQueue(node->getRight());
    
    //ot::Print(curRelErr(), "curRelErr = ", stdout);
  }
  ot::Print(basis_, "basis", stdout);
  extractSVD(s, U, VT);
}


void createUTA2(const ArrayList<Vector>& UTA, Matrix* UTA2) {
  UTA2->Init(UTA.size(), UTA.size());
  for (int i = 0; i < UTA.size(); i++)
    for (int j = 0; j < UTA.size(); j++)
      UTA2->ref(i, j) = la::Dot(UTA[i], UTA[j]);
}

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

void InverseRowScale(const Vector& s, Matrix* A) {
  for (index_t i = 0; i < A->n_rows(); i++)
    for (index_t j = 0; j < A->n_cols(); j++)
      A->ref(i, j) /= s[i];
}

void QuicSVD::extractSVD(Vector* s, Matrix* U, Matrix* VT) {
  ot::Print(UTA_, "UTA", stdout);

  Matrix UTA2;
  Matrix Uprime, VprimeT;
  Vector sprime;
  createUTA2(UTA_, &UTA2);
  
  la::SVDInit(UTA2, &sprime, &Uprime, &VprimeT);
  
  s->Init(sprime.length());
  for (index_t i = 0; i < sprime.length(); i++)
    (*s)[i] = sqrt(sprime[i]);
  
  MulInit(basis_, Uprime, U);
  MulTransCInit(UTA_, Uprime, VT);
  InverseRowScale(*s, VT);

  ot::Print(*U, "U=", stdout);
  ot::Print(*s, "s=", stdout);
  ot::Print(*VT, "VT=", stdout);
}
