#include <fastlib/fastlib.h>
#include <limits>
#include <queue>
#include <iostream>
#include <sstream>

#include "anmf.h"

using namespace std;

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

std::string toString (const Vector& v)
{
  std::stringstream s;
  s << "(" << v[0];
  for (int i = 1; i < v.length(); i++)
    s << "," << v[i];
  s << ")";
  return s.str();
}

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
  Vector label_, slack_;
  ArrayList<int> match_, prev_;
  int n_;
 public:
  ProjectOneToOne(const Vector& a, Vector* b);
 private:
  void initLabel();
  int findAugmentingPath(int u);
  void augmentPath(int u, int v);
  inline static bool equal(double a, double b);
  inline double weight(int u, int v);
};

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

  int maxIter = fx_param_int(module, "iter", 10), iter;
  double tol = fx_param_double(module, "ctol", 1e-12);
  bool stop = false;
  for (iter = 0; iter < maxIter && !stop; iter++)
  {
    cout << "iter = " << iter << endl;
    //    ot::Print(x_k, "x_k");
    Vector Mxk, b_k, d_k;
    la::MulInit(M, x_k, &Mxk);
    ProjectOneToOne(Mxk, &b_k);

    // check if new value better than OPT
    double value = xAy(b_k, M, b_k);
    cout << "value = " << value << " OPT = " << OPT << endl;
    if (value > OPT)
    {
      solution.CopyValues(b_k);
      OPT = value;
    }

    // prepare the next search point
    // by finding the best point on the line joining x_k and b_k
    la::SubInit(x_k, b_k, &d_k);
    double C = xAy(x_k, M, d_k);
    double D = xAy(d_k, M, d_k);
    double r = 1;
    if (D < 0)
      r = (-C/D < r) ? -C/D : r;
    if (r*la::LengthEuclidean(d_k) < tol) stop = true;
    
    la::AddExpert(r, d_k, &x_k);
  }
  cout << "Stop at iter = " << iter << " OPT = " << OPT << endl;
}

// Output a one-to-one matching that have maximum total weight
// A classic problem solved by the Hungarian method (for bipartie graph)
// Input:  a - weight[i,j] = a[i*n+j]
// Output: b - matching[i, j] = b[i*n+j] \in {0, 1}
ProjectOneToOne::ProjectOneToOne(const Vector& a, Vector* pb) : a_(a), b_(*pb)
{
  // ot::Print(a, "a");
  n_ = (int) sqrt(a_.length());
  int n2 = n_*n_;
  DEBUG_ASSERT(n2 == a_.length());
  initLabel();
  for (int n_match = 0; n_match < n_; n_match++)
  {
    // cout << "n_match = " << n_match << endl;
    // pick a free vertex
    int u = 0;
    for (; u < n_ && match_[u] != -1; u++);
    DEBUG_ASSERT(u < n_);
    int v = findAugmentingPath(u);
    augmentPath(u, v);
    // ot::Print(match_, "match");
  }
  // left vertex i is mapped to right vertex match_[i]-n
  b_.Init(n2);
  b_.SetZero();
  for (int i = 0; i < n_; i++)
    b_[i*n_+match_[i]-n_] = 1;
}

void ProjectOneToOne::initLabel()
{
  // init label such that label[i] + label[j] >= a[i,j]
  // where the i-th right vertices is the i+n-th element
  label_.Init(n_*2);
  slack_.Init(n_*2);
  for (int i = 0; i < n_; i++)
  {
    // init the left vertices
    label_[i] = 0;
    slack_[i] = 0;
    // init the right vertices
    label_[i+n_] = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < n_; j++)
      if (weight(j,i) > label_[i+n_]) label_[i+n_] = weight(j,i);
  }
  // at first, there is no match
  match_.InitRepeat(-1, n_*2);
  prev_.Init(n_*2);
}

bool ProjectOneToOne::equal(double a, double b) 
{ 
  static const double tol = 1e-12;
  return a-b < tol && b-a < tol; 
}

double ProjectOneToOne::weight(int u, int v)
{
  if (u < n_ && v < n_)
    return a_[u*n_+v];
  else if (u >= n_)
    return a_[v*n_+u-n_];
  else 
    return a_[u*n_+v-n_];
}

// find an augmenting path from a free left vertex
// to a free right vertex in the equality bipartie graph
// (i,j) \in E iff label[i] + label[j] == weight[i,j]
// if no path is found, change the labels to include more edges
int ProjectOneToOne::findAugmentingPath(int u)
{
  std::queue<int> vertexQueue;
  vertexQueue.push(u);
  for (int i = 0; i < n_*2; i++)
    prev_[i] = -1;
  prev_[u] = u;
  // initialize the slacks
  for (int i = n_; i < 2*n_; i++)
    slack_[i] = label_[i]+label_[u]-weight(u, i);
  for (;;)
  {
    DEBUG_ASSERT(!vertexQueue.empty());
    int i = vertexQueue.front();
    // cout << "i = " << i << endl;
    vertexQueue.pop();
    if (i < n_)
    {
      for (int j = n_; j < n_*2; j++)
      {
	// cout << "j = " << j << " s = " << label_[i]+label_[j]-weight(i,j) << endl;
	if (equal(label_[i]+label_[j], weight(i,j)) && prev_[j] == -1) // if there is an edge (i,j) in the equality graph
	{                                                            // and j is not marked (visited)
	  // cout << "j = " << j << endl;
	  vertexQueue.push(j);
	  prev_[j] = i;
	  if (j >= n_ && match_[j] == -1) // found a free right vertex
	    return j;
	}
      }
    }
    else // i is a right vertex then i should be matched (not free)
    {
      int j = match_[i];
      if (prev_[j] == -1)
      {
	// cout << "j = " << j << endl;
	vertexQueue.push(j);
	prev_[j] = i;
	// update slacks when new left vertex is visited
	for (int k = n_; k < n_*2; k++)
	  if (prev_[k] == -1 && slack_[k] > label_[k] + label_[j] - weight(j, k)) 
	    slack_[k] = label_[k] + label_[j] - weight(j, k);
      }
    }
    if (!vertexQueue.empty()) continue;
    // cout << "Change labels" << endl;
    // if the queue is empty (i.e. no path found under current equality graph)
    // change the labels to include more edges
    // for visited left vertex, label -= min(slacks of not visited right vertices)
    // for visited right vertex, label += min(slacks of not visited right vertices)
    // for non-visited right vertex, slack -= min(slacks of not visited right vertices)
    // ot::Print(slack_, "slack before");
    double minSlack = std::numeric_limits<double>::infinity();
    for (int i = n_; i < n_*2; i++)
      if (prev_[i] == -1 && minSlack > slack_[i]) minSlack = slack_[i];
    // cout << "min slack = " << minSlack << endl;
    DEBUG_ASSERT(minSlack > 0);
    for (int i = 0; i < n_; i++)
      if (prev_[i] != -1) label_[i] -= minSlack;
    for (int i = n_; i < n_*2; i++)
      if (prev_[i] != -1) label_[i] += minSlack;
      else 
      {
	slack_[i] -= minSlack;
	// if (slack_[i] == 0) cout << "slack[" << i << "] = 0" << endl;
      }
    // push visited vertices back to the queue
    for (int i = 0; i < n_*2; i++)
      if (prev_[i] != -1) vertexQueue.push(i);
    // ot::Print(label_, "label");
    // ot::Print(slack_, "slack after");
  }
  return -1; // failure
}

void ProjectOneToOne::augmentPath(int u, int v)
{
  DEBUG_ASSERT(v >= n_);
  // cout << "augmenting ... ";
  while (v != u)
  {
    int i = prev_[v];
    int j = prev_[i];
    match_[i] = v;
    match_[v] = i;
    v = j;
  }
  // cout << "done" << endl;
}

END_ANMF_NAMESPACE;
