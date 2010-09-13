#include <fastlib/fastlib.h>
#include <limits>

#include "anmf.h"

BEGIN_ANMF_NAMESPACE;

double xAy(const Vector& x, const Matrix& A, const Vector& y)
{
  DEBUG_ASSERT(x.length() == A.n_rows() && A.n_cols() == y.length());
  double s = 0;
  for (int i = 0; i < x.length(); i++)
    for (int j = 0; j < y.length(); j++)
      s += x[i]*A.get(i,j)*y[j];
  return s;
}

double euclideanDistance(const Vector& x, const Vector& y)
{
  DEBUG_ASSERT(x.length() == y.length());
  double s = 0;
  for (int i = 0; i < x.length(); i++)
      s += math::Sqr(x[i]-y[i]);
  return sqrt(s);
}

void ipfpGraphMatching(fx_module* module, const Matrix& M, Vector& solution)
{
  int n = (int) sqrt(M.n_rows());
  int n2 = n*n;
  DEBUG_ASSERT(M.n_rows() == M.n_cols() && n2 == M.n_rows());

  Vector x_k;
  x_k.Init(n2); 
  
  // start with the identity match: i <--> i, i = 0, ... n-1
  x_k.SetZero();
  for (int i = 0; i < n; i++) 
    x_k[i*n+i] = 1;

  // sol[i*n+a] == 1 iff feature i is mapped to feature a
  solution.Copy(x_k);
  double OPT = xAy(x_k, M, x_k);

  int maxIter = fx_param_int(module, "iter", 10);
  double tol = fx_param_double(module, "tol", 1e-12);
  bool stop = false;
  for (int iter = 0; iter < maxIter && !stop; iter++)
  {
    Vector Mxk, b_k, d_k;
    la::MulInit(M, x_k, &Mxk);
    ProjectOneToOne(Mxk, &b_k);

    // check if new value better than OPT
    double value = xAy(b_k, M, b_k);
    if (value > OPT)
    {
      solution.CopyValues(b_k);
      OPT = value;
    }

    // prepare the next search point
    // by finding the best point on the line joining x_k and b_k
    la::SubInit(b_k, x_k, &d_k);
    double C = xAy(x_k, M, d_k);
    double D = xAy(d_k, M, d_k);
    double r = 1;
    if (D < 0)
      r = (-C/D < r) ? -C/D : r;
    if (r*la::LengthEuclidean(d_k) < tol) stop = true;
    
    la::AddExpert(r, d_k, &x_k);
  }
}

// Output a one-to-one matching that have maximum total weight
// A classic problem solved by the Hungarian method (for bipartie graph)
// Input:  a - weight[i,j] = a[i*n+j]
// Output: b - matching[i, j] = b[i*n+j] \in {0, 1}
ProjectOneToOne::ProjectOneToOne(const Vector& a, Vector* pb) : a_(a), b_(*pb)
{
  n_ = (int) sqrt(a_.length());
  int n2 = n_*n_;
  DEBUG_ASSERT(n2 == a_.length());
  b_.Init(n2);
  initLabel();
  for (int n_match = 0; n_match < n_; n_match++)
  {
    // pick a free vertex
    int u = 0;
    for (; u < n_ && match_[u] == -1; u++);
    int v = findAugmentingPath(u);
    augmentPath(u, v);
  }
  // left vertex i is mapped to right vertex match_[i]-n
  b_.SetZero();
  for (int i = 0; i < n_; i++)
    b_[i*n_+match_[i]-n_] = 1;
}

void ProjectOneToOne::initLabel()
{
  // init label such that label[i] + label[j] >= a[i,j]
  // where the i-th right vertices is the i+n-th element
  label_.Init(n_*2);
  for (int i = 0; i < n_; i++)
  {
    // init the left vertices
    label_[i] = 0;
    // init the right vertices
    label_[i+n_] = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < n_; j++)
      if (a_[j*n_+i] > label_[i+n_]) label_[i+n_] = a_[j*n_+i];
  }
  // at first, there is no match
  match_.InitRepeat(-1, n_*2);
  prev_.InitRepeat(-1, n_*2);
}

int ProjectOneToOne::findAugmentingPath(int u)
{
  for (int i = 0; i < n_*2; i++) prev_[i] = -1;
  prev[u] = u;
  
}

void ProjectOneToOne::augmentPath(int u, int v)
{
}

END_ANMF_NAMESPACE;
