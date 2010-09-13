/** @file: anmf.h
 *  The main header file for affine NMF
 *  including following functionalities:
 *    + Graph Matching
 *    + Nonegative Matrix Factorization
 *    + etc.
 */

#ifndef TQLONG_ANMF_H
#define TQLONG_ANMF_H

#ifndef BEGIN_ANMF_NAMESPACE
#define BEGIN_ANMF_NAMESPACE namespace anmf {
#endif
#ifndef END_ANMF_NAMESPACE
#define END_ANMF_NAMESPACE }
#endif

BEGIN_ANMF_NAMESPACE;

/** Integer Projected Fixed Point graph matching
 *  Solving max_x x'Mx s.t Ax = 1, x \in {0,1}^{n^2}
 *  Input:  x is a vectorized matrix, x_ia = 1 iff feature i is mapped to feature a
 *          M is a similarity measure matrix, M_{ia; jb} is the score if edge (i,j) is
 *            mapped to edge (a,b)
 *          A is the one-to-one constraint on x
 *  Output: sol[i*n+a] == 1 iff feature i is mapped to feature a 
 */
void ipfpGraphMatching(fx_module* module, const Matrix& M, Vector& solution);

/** Project a vectorized matrix to the one-to-one discrete constraint 
 *  Basically, it is equivalent to the max-weight matching problem
 *  which can be solved by the Hungarian method
 *  Input:  a - weight[i,j] = a[i*n+j]
 *  Output: b - matching[i, j] = b[i*n+j] \in {0, 1}
 */
class ProjectOneToOne
{
  const Vector& a_;
  Vector& b_;
  Vector label_;
  ArrayList<int> match_, prev_;
  int n_;
 public:
  ProjectOneToOne(const Vector& a, Vector* b);
 private:
  void initLabel();
  int findAugmentingPath(int u);
  void augmentPath(int u, int v);
  inline static bool equal(double a, double b) { return a-b < 1e-16 && b-a < 1e-16; }
};

END_ANMF_NAMESPACE;

#endif
