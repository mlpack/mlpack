/* test.cc */
#include <boost/mpi.hpp>
#include <iostream>
#include <fstream>
#include <armadillo>
#include <functional>
#include <boost/serialization/string.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
using namespace boost;
using namespace std;
using namespace arma;
using namespace boost::posix_time;

namespace boost { namespace serialization {
  template<class Archive>
  inline void save(Archive & ar, const vec &t, const unsigned int file_version)
  {
    ar & t.n_elem;
    for (unsigned int i = 0; i < t.n_elem; i++)
      ar & *(t.memptr()+i);
  }
  template<class Archive>
  inline void load(Archive & ar, vec &t, const unsigned int file_version)
  {
    unsigned int n_elem;
    ar & n_elem;
    t = vec(n_elem);
    for (unsigned int i = 0; i < t.n_elem; i++)
      ar & *(t.memptr()+i);
  }
  template<class Archive>
  inline void serialize(Archive & ar, vec &t, const unsigned int file_version)
  {
    boost::serialization::split_free(ar, t, file_version);
  }

  template<class Archive>
  inline void save(Archive & ar, const time_duration &t, const unsigned int file_version)
  {
    std::string s = to_simple_string(t);
    ar & s;
  }
  template<class Archive>
  inline void load(Archive & ar, time_duration &t, const unsigned int file_version)
  {
    std::string s;
    ar & s;
    t = duration_from_string(s);
  }
  template<class Archive>
  inline void serialize(Archive & ar, time_duration &t, const unsigned int file_version)
  {
    boost::serialization::split_free(ar, t, file_version);
  }

} } // namespace boost::serialization

namespace boost { namespace mpi {
  template<>
  struct is_commutative<std::plus<vec>, vec>
    : mpl::true_ { };
  template<>
  struct is_commutative<std::plus<double>, double>
    : mpl::true_ { };
} } // end namespace boost::mpi

namespace std
{
  template <class T> struct max2 : binary_function <T,T,T> {
    T operator() (const T& x, const T& y) const
    {return x < y ? y : x;}
  };
}

void testBroadcast(mpi::communicator &world);
void testReduce(mpi::communicator &world);
void testCG(mpi::communicator &world);

void readData(string filename, int &n_points, int &dim, std::vector<vec> &samples, std::vector<double> &labels);
void sendSamples(mpi::communicator &world, int &n_points, int dim, std::vector<vec> &samples, std::vector<double> &labels, const std::vector<vec> &all_samples, const std::vector<double> &all_labels);
vec calGrad(const vec& w, const std::vector<vec> &samples, const std::vector<double> &labels);
double calFunc(const vec& w, const std::vector<vec> &samples, const std::vector<double> &labels);
double errorCount(const vec& w, const std::vector<vec> &samples, const std::vector<double> &labels);

vec parGrad(mpi::communicator &world, vec& w, const std::vector<vec> &samples, const std::vector<double> &labels, bool broadcasted);
double parFunc(mpi::communicator &world, vec& w, const std::vector<vec> &samples, const std::vector<double> &labels, bool broadcasted);
vec parLineSearch(mpi::communicator &world, const vec& w, const std::vector<vec> &samples, const std::vector<double> &labels,
                  const vec &grad, const vec& dir, double f0, double &alpha);
vec parConjGrad(mpi::communicator &world, vec& w, const std::vector<vec> &samples, const std::vector<double> &labels, const vec &b, bool broadcasted);
double parErrorCount(mpi::communicator &world, vec& w, const std::vector<vec> &samples, const std::vector<double> &labels, bool broadcasted);

int main(int argc, char* argv[])
{
  mpi::environment env(argc, argv);
  mpi::communicator world;
  
  int n_points, dim, n_total = 0;
  ifstream in;
  std::vector<vec> samples, all_samples;
  std::vector<double> labels, all_labels;

  // only the first/master process read the data file
  if (world.rank() == 0)
  {
    readData("test.txt", n_points, dim, all_samples, all_labels);
    n_total = (int) all_samples.size();
    cout << "Read n = " << n_total << "\n";
  }
  mpi::broadcast(world, dim, 0);
  sendSamples(world, n_points, dim, samples, labels, all_samples, all_labels);
  all_samples.clear();
  all_labels.clear();
  // at this point, the data set is partitioned and
  // each process (starting from 1) has a part (subset of samples and labels)

  ptime time_start(microsec_clock::local_time());

  vec w, grad, dir;
  double f = 0, f0 = 0, g = 0, g0 = 0;
  bool ok = true;
  double atol = 1e-6, rtol = 1e-4,
         gatol = 1e-6, grtol = 1e-6;

  if (world.rank() == 0)
  {
    w = randu<vec>(dim+1)*16; // [w, b]
  }

  int iter;
  for (iter = 0; iter < 100000 && ok; iter++)
  {
    grad = parGrad(world, w, samples, labels, false);  // only process 0 has gradient
    f = parFunc(world, w, samples, labels, true);     // only process 0 has function value
    if (world.rank() == 0)
    {
      g = norm(grad, 2);
      if (iter == 0)
      {
        f0 = f;
        g0 = g;
      }
      if (g <= g0*grtol+gatol)
      {
        cout << "Gradient small ... terminate.\n";
        ok = false;
      }
      if (f <= f0*rtol+atol)
      {
        cout << "Function small ... terminate.\n";
        ok = false;
      }
    }
    dir = parConjGrad(world, w, samples, labels, grad, true);  // only process 0 has search direction
    //dir = grad;
    double alpha;
    w = parLineSearch(world, w, samples, labels, grad, -dir, f, alpha);  // only process 0 has new search point
    if (world.rank() == 0 && alpha == 0)
      ok = false;
    mpi::broadcast(world, ok, 0);               // broadcast termination condition
    if (world.rank() == 0 && (iter == 0 || (iter+1) % 10 == 0))
    {
      cout << "iter " << iter << " f = " << f/n_total
           << " norm = " << g/n_total
           << " alpha = " << alpha << "\n";
    }
  }
  f = parFunc(world, w, samples, labels, false);
  double error = parErrorCount(world, w, samples, labels, true);
  ptime time_end(microsec_clock::local_time());
  time_duration duration(time_end - time_start), max_duration, total_duration;
  mpi::reduce(world, duration, max_duration, std::max2<time_duration>(), 0);
  mpi::reduce(world, duration, total_duration, std::plus<time_duration>(), 0);
  if (world.rank() == 0)
  {
    cout << "iter = " << iter << " final f = " << f/n_total << " w = \n" << w
         << "error = " << error/n_total
         << " duration: " << duration << "\n"
         << " max_duration: " << max_duration << "\n"
         << " total_duration: " << total_duration << "\n";
  }


  //  testBroadcast(world);
  //  testReduce(world);
  //  testCG(world);
    return 0;
}

template <typename M, typename V>
V conjgrad(const M& A, const V& b, int maxiter = 10)
{
  V x = zeros<V>(b.n_elem);
  V r = b, p = b, Ap;
  double alpha, rdotr = dot(r,r), rdotr_next;
  p = r = b;
  for (int iter = 0; iter < maxiter && sqrt(rdotr) >= 1e-10 ; iter++)
  {
    Ap = A*p;
    alpha = rdotr / dot(p, Ap);
    x = x + alpha*p;
    r = r - alpha*Ap;
    rdotr_next = dot(r, r);
    p = r + (rdotr_next / rdotr) *p;
    rdotr = rdotr_next;
  }
  return x;
}

vec prodMatVec(const vec& w, const std::vector<vec> &samples, const std::vector<double> &labels, const vec& p)
{
  int dim = w.n_elem-1;
  vec Ap = zeros<vec>(dim+1);
  for (unsigned int i = 0; i < samples.size(); i++)
  {
    double z = labels[i]* (dot(w.rows(0, dim-1), samples[i])+w[dim]);
    double xp = dot(p.rows(0, dim-1), samples[i])+p[dim];
    double ez = exp(-z);
    double coef = (ez) / (1+ez) / (1+ez) * xp;
    Ap.rows(0,dim-1) += (coef*samples[i]);
    Ap[dim] += coef;
  }
  return Ap;
}

vec parConjGrad(mpi::communicator &world, vec& w, const std::vector<vec> &samples, const std::vector<double> &labels, const vec &b, bool broadcasted)
{
  if (!broadcasted)
    mpi::broadcast(world, w, 0);
  vec x, r, p, Ap, Ap_part;
  double alpha = 0, rdotr = 0, rdotr_next = 0;
  bool ok = true;
  int maxiter = 10;

  if (world.rank() == 0)
  {
    x = zeros<vec>(b.n_elem);
    r = p = b;
    rdotr = dot(r,r);
    ok = sqrt(rdotr) > 1e-10;
  }
  mpi::broadcast(world, ok, 0);
  for (int iter = 0; iter < maxiter && ok; iter++)
  {
    mpi::broadcast(world, p, 0);
//    if (world.rank() == 0)
//      Ap_part = zeros<vec>(w.n_elem);
//    else
    Ap_part = prodMatVec(w, samples, labels, p);
    mpi::reduce(world, Ap_part, Ap, std::plus<vec>(), 0);

    if (world.rank() == 0)
    {
      alpha = rdotr / dot(p, Ap);
      x = x + alpha*p;
      r = r - alpha*Ap;
      rdotr_next = dot(r, r);
      p = r + (rdotr_next / rdotr) *p;
      rdotr = rdotr_next;
      ok = sqrt(rdotr) > 1e-10;
    }
    mpi::broadcast(world, ok, 0);
  }
  return x;
}

double errorCount(const vec& w, const std::vector<vec> &samples, const std::vector<double> &labels)
{
  int dim = w.n_elem-1;
  double n_error = 0;
  for (unsigned int i = 0; i < samples.size(); i++)
  {
    double z = labels[i]*(dot(w.rows(0, dim-1), samples[i])+w[dim]);  // w[0..dim-1] is the weight, w[dim] is the bias
    n_error += (z <= 0);
  }
  return n_error;
}

double parErrorCount(mpi::communicator &world, vec& w, const std::vector<vec> &samples, const std::vector<double> &labels, bool broadcasted)
{
  double f, result;
  if (!broadcasted)
    mpi::broadcast(world, w, 0);
  f = errorCount(w, samples, labels);
  mpi::reduce(world, f, result, std::plus<double>(), 0);
  return result;
}


vec calGrad(const vec& w, const std::vector<vec> &samples, const std::vector<double> &labels)
{
  int dim = w.n_elem-1;
  vec grad = zeros<vec>(dim+1);
  for (unsigned int i = 0; i < samples.size(); i++)
  {
    double z = labels[i]*(dot(w.rows(0, dim-1), samples[i])+w[dim]);
    double ez = exp(-z);
    double coef = (-labels[i]*ez) / (1+ez);
    grad.rows(0,dim-1) += (coef*samples[i]);
    grad[dim] += coef;
  }
  return grad;
}

vec parGrad(mpi::communicator &world, vec& w, const std::vector<vec> &samples, const std::vector<double> &labels, bool broadcasted)
{
  vec grad, result;
  if (!broadcasted)
    mpi::broadcast(world, w, 0);
  grad = calGrad(w, samples, labels);
  mpi::reduce(world, grad, result, std::plus<vec>(), 0);
  return result;
}

double calFunc(const vec& w, const std::vector<vec> &samples, const std::vector<double> &labels)
{
  int dim = w.n_elem-1;
  double func = 0;
  for (unsigned int i = 0; i < samples.size(); i++)
  {
    double z = labels[i]*(dot(w.rows(0, dim-1), samples[i])+w[dim]);  // w[0..dim-1] is the weight, w[dim] is the bias
    func += log1p(exp(-z));
  }
  return func;
}

double parFunc(mpi::communicator &world, vec& w, const std::vector<vec> &samples, const std::vector<double> &labels, bool broadcasted)
{
  double f, result;
  if (!broadcasted)
    mpi::broadcast(world, w, 0);
  f = calFunc(w, samples, labels);
  mpi::reduce(world, f, result, std::plus<double>(), 0);
  return result;
}

vec parLineSearch(mpi::communicator &world, const vec& w, const std::vector<vec> &samples, const std::vector<double> &labels,
                  const vec &grad, const vec& dir, double f0, double &alpha)
{
  double tau = 5e-1;
  double c1 = 1e-4;
  double gdotd = 0.0;
  bool ok = true;
  vec search_point;
  if (world.rank() == 0)
    gdotd = dot(grad, dir);
  alpha = 1.0/tau;
  while (alpha >= 1e-10 && ok)
  {
    alpha = tau*alpha;
    if (world.rank() == 0)
      search_point = w + alpha*dir;
    double f = parFunc(world, search_point, samples, labels, false);
    if (world.rank() == 0)
    {
      if (f <= f0 + c1*alpha*gdotd)
        ok = false;
    }
    mpi::broadcast(world, ok, 0);
  }
  if (ok)
  {
    alpha = 0;
    return w;
  }
  else
    return search_point;
}

void sendSamples(mpi::communicator &world, int &n_points, int dim, std::vector<vec> &samples, std::vector<double> &labels, const std::vector<vec> &all_samples, const std::vector<double> &all_labels)
{
  if (world.rank() == 0)
  {
    std::vector<mpi::request> reqs;
    int size = world.size();
    int sample_per_process = all_samples.size() / size;
    int current_process = -1;
    for (unsigned int i = 0; i < all_samples.size(); i++)
    {
      if (i % sample_per_process == 0 && current_process < size-1)
      {
        current_process++;
        int n = current_process < size-1 ? sample_per_process : all_samples.size()-i;
//        cout << "r = " << current_process << " n = " << n << "\n";
        if (current_process > 0)
          reqs.push_back(world.isend(current_process, 0, n));
        else
          n_points = n;
      }
      if (current_process > 0)
      {
        reqs.push_back(world.isend(current_process, 1, all_samples[i]));
        reqs.push_back(world.isend(current_process, 2,  all_labels[i]));
      }
      else
      {
        samples.push_back(all_samples[i]);
        labels.push_back(all_labels[i]);
      }
    }
    mpi::wait_all(reqs.begin(), reqs.end());
  }
  else
  {
    world.recv(0, 0, n_points);
    for (int i = 0; i < n_points; i++)
    {
      vec v(dim);
      double label;
      world.recv(0, 1, v);
      world.recv(0, 2, label);
      samples.push_back(v);
      labels.push_back(label);
    }
  }
//  cout << "rank = " << world.rank() << " n_points = " << n_points << "\n";
}

void readData(string filename, int &n_points, int &dim, std::vector<vec> &samples, std::vector<double> &labels)
{
//  ifstream in(filename.c_str());
//  in >> n_points >> dim;
//  samples.clear();
//  labels.clear();
//  for (int i = 0; i < n_points; i++)
//  {
//    vec v(dim);
//    double label;
//    for (int j = 0; j < dim; j++)
//      in >> v[j];
//    in >> label;
//    samples.push_back(v);
//    labels.push_back(label);
//  }
  n_points = 1000;
  dim = 2;
  vec w = randu<vec>(dim) * 2 - 1;
  w = w / norm(w,2);
  double b = randu<vec>(1)[0] * 2 - 1;
  double margin = 0.01;
  cout << "w = \n" << w << "b = " << b << "\n";
  samples.clear();
  labels.clear();
  for (int i = 0; i < n_points; i++)
  {
    vec v = randu<vec>(dim) * 2 - 1;
    double m = dot(w,v)+b, l = 0;
    if (m > margin) l = 1;
    if (m < -margin) l = -1;
    if (l == 0) continue;
    samples.push_back(v);
    labels.push_back(l);
  }
  n_points = samples.size();
}

void testBroadcast(mpi::communicator &world)
{
  srand(time(NULL)+world.rank()*1013);
  vec g;
  vec result;

  if (world.rank() == 0)
    g = randu<vec>(5);
  mpi::broadcast(world, g, 0);
  g = g*world.rank();

  mpi::reduce(world, g, result, std::plus<vec>(), 0);
  cout << "Process " << world.rank() << " of " << world.size()
            << "." << "\ng = \n" << g << "\n";
  if (world.rank() == 0)
    cout << "Sum = " << result << "\n";
}

void testReduce(mpi::communicator &world)
{
  srand(time(NULL)+world.rank()*1013);
  vec g = randu<vec>(5);
  vec result;

  cout << "Process " << world.rank() << " of " << world.size()
            << "." << "\ng = \n" << g << "\n";

  mpi::reduce(world, g, result, std::plus<vec>(), 0);
  if (world.rank() == 0)
    cout << "Sum = " << result << "\n";
}

void testCG(mpi::communicator &world)
{
  srand(time(NULL)+world.rank()*1013);
  mat A = randu<mat>(5,5);
  A = A+trans(A);
//  A.eye();
  vec b = randn<vec>(5);
  //NaiveMatVec nA(A);
  
  cout << "I am process " << world.rank() << " of " << world.size()
            << "." << endl;
  cout << "A = \n" << A << "\n"
       << "b = " << b << "\n";
  vec x = conjgrad<mat, vec>(A, b, A.n_rows);
  cout << "My result: x=" << x
       << " residual = " << norm(A*x-b, 2) << endl;
}
