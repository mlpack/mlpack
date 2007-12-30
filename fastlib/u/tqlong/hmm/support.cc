#include "fastlib/fastlib.h"
#include "support.h"

void print_matrix(const Matrix& a, const char* msg) {
  printf("%s - Matrix (%d x %d) = \n", msg, a.n_rows(), a.n_cols());
  for (int i = 0; i < a.n_rows(); i++) {
    for (int j = 0; j < a.n_cols(); j++)
      printf("%8.4f", a.get(i, j));
    printf("\n");
  }
}

void print_vector(const Vector& a, const char* msg) {
  printf("%s - Vector (%d) = \n", msg, a.length());
  for (int i = 0; i < a.length(); i++)
    printf("%8.4f", a[i]);
  printf("\n");
}

double RAND_NORMAL_01() {
  double r = 2, u, v;
  while (r > 1) {
    u = RAND_UNIFORM(-1, 1);
    v = RAND_UNIFORM(-1, 1);
    r = u*u+v*v;
  }
  return sqrt(-2*log(r)/r)*u;
}

void RAND_NORMAL_01_INIT(int N, Vector* v) {
  double r, u, t;
  Vector& v_ = *v;
  v_.Init(N);
  for (int i = 0; i < N; i+=2) {
    r = 2;
    while (r > 1) {
      u = RAND_UNIFORM(-1, 1);
      t = RAND_UNIFORM(-1, 1);
      r = u*u+t*t;
    }
    v_[i] = sqrt(-2*log(r)/r)*u;
    if (i+1 < N) v_[i+1] = sqrt(-2*log(r)/r)*t;
  }
}

void RAND_NORMAL_INIT(const Vector& mean, const Matrix& cov, Vector* v) {
  int N = mean.length();
  Vector v01;
  RAND_NORMAL_01_INIT(N, &v01);
  la::MulInit(cov, v01, v);
  la::AddTo(mean, v);
}

// return x'Ay
double MyMulExpert(const Vector& x, const Matrix& A, const Vector& y) {
  int M = A.n_rows();
  int N = A.n_cols();
  DEBUG_ASSERT_MSG((M==x.length() && N==y.length()), "MyMulExpert: sizes do not match");

  double s = 0;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      s += x[i] * A.get(i, j) * y[j];
  return s;
}

double NORMAL_DENSITY(const Vector& x, const Vector& mean, const Matrix& inv_cov, double det_cov) {
  Vector d;
  la::SubInit(x, mean, &d);
  return det_cov * exp(-0.5*MyMulExpert(d, inv_cov, d));
}

bool kmeans(const ArrayList<Matrix>& data, int num_clusters, 
	    ArrayList<int> *labels_, ArrayList<Vector> *cetroids_, 
	    int max_iter, double error_thresh)
{
  ArrayList<int> counts; //number of points in each cluster
  ArrayList<Vector> tmp_centroids;
  int num_points, num_dims;
  int i, j, num_iter=0;
  double error, old_error;
  
  //Assign pointers to references to avoid repeated dereferencing.
  ArrayList<int> &labels = *labels_;
  ArrayList<Vector> &centroids = *cetroids_;

  num_points = 0;
  for (int i = 0; i < data.size(); i++)
    num_points+=data[i].n_cols();
  if (num_points < num_clusters) 
    return false;

  num_dims = data[0].n_rows();

  centroids.Init(num_clusters);
  tmp_centroids.Init(num_clusters);

  counts.Init(num_clusters);
  labels.Init(num_points);

  //Initialize the clusters to k points
  for (j=0; j < num_clusters; j++) {
    Vector temp_vector;
    int i = (int) math::Random(0, data.size() - 0.5);
    int k = (int) math::Random(0, data[i].n_cols()-0.5);
    data[i].MakeColumnVector(k, &temp_vector);
    centroids[j].Copy(temp_vector);
    tmp_centroids[j].Init(num_dims);
  }

  error = DBL_MAX;

  do {

    old_error = error; error = 0;

    for (i=0; i < num_clusters; i++) {
      tmp_centroids[i].SetZero();
          counts[i] = 0;
    }
    i = 0;
    for (int t=0, i = 0; t<data.size(); t++) 
    for (int k = 0; k < data[t].n_cols(); k++, i++) {
      
      // Find the cluster closest to this point and update its label
      double min_distance = DBL_MAX;
      Vector data_i_Vec;
      data[t].MakeColumnVector(k, &data_i_Vec);
      
      for (j=0; j<num_clusters; j++ ) {
	double distance = la::DistanceSqEuclidean(data_i_Vec, centroids[j]);
	if (distance < min_distance) {
	  labels[i] = j;
	  min_distance = distance;
	}
      }

      // Accumulate the stats for the new centroid of the target cluster
      la::AddTo(data_i_Vec, &(tmp_centroids[labels[i]]));
      counts[labels[i]]++;
      error += min_distance;
    }

    // Now update all the centroids
    for (int j=0; j < num_clusters; j++) {
      if (counts[j] > 0)
	la::ScaleOverwrite((1/(double)counts[j]), 
			   tmp_centroids[j], &(centroids[j]));
    }
    num_iter++;

  } while ((fabs(error - old_error) > error_thresh)
	   && (num_iter < max_iter));

  return true;
  
}

bool kmeans(Matrix const &data, int num_clusters, 
	    ArrayList<int> *labels_, ArrayList<Vector> *cetroids_, 
	    int max_iter, double error_thresh)
{
  ArrayList<int> counts; //number of points in each cluster
  ArrayList<Vector> tmp_centroids;
  int num_points, num_dims;
  int i, j, num_iter=0;
  double error, old_error;
  
  //Assign pointers to references to avoid repeated dereferencing.
  ArrayList<int> &labels = *labels_;
  ArrayList<Vector> &centroids = *cetroids_;

  if (data.n_cols() < num_clusters) 
    return false;

  num_points = data.n_cols();
  num_dims = data.n_rows();

  centroids.Init(num_clusters);
  tmp_centroids.Init(num_clusters);

  counts.Init(num_clusters);
  labels.Init(num_points);

  //Initialize the clusters to k points
  for (i=0, j=0; j < num_clusters; i+=num_points/num_clusters, j++) {
    Vector temp_vector;
    data.MakeColumnVector(i, &temp_vector);
    centroids[j].Copy(temp_vector);
    tmp_centroids[j].Init(num_dims);
  }

  error = DBL_MAX;

  do {

    old_error = error; error = 0;

    for (i=0; i < num_clusters; i++) {
      tmp_centroids[i].SetZero();
          counts[i] = 0;
    }

    for (i=0; i<num_points; i++) {

      // Find the cluster closest to this point and update its label
      double min_distance = DBL_MAX;
      Vector data_i_Vec;
      data.MakeColumnVector(i, &data_i_Vec);
      
      for (j=0; j<num_clusters; j++ ) {
	double distance = la::DistanceSqEuclidean(data_i_Vec, centroids[j]);
	if (distance < min_distance) {
	  labels[i] = j;
	  min_distance = distance;
	}
      }

      // Accumulate the stats for the new centroid of the target cluster
      la::AddTo(data_i_Vec, &(tmp_centroids[labels[i]]));
      counts[labels[i]]++;
      error += min_distance;
    }

    // Now update all the centroids
    for (int j=0; j < num_clusters; j++) {
      if (counts[j] > 0)
	la::ScaleOverwrite((1/(double)counts[j]), 
			   tmp_centroids[j], &(centroids[j]));
    }
    num_iter++;

  } while ((fabs(error - old_error) > error_thresh)
	   && (num_iter < max_iter));

  return true;
  
}

void mat2arrlst(Matrix& a, ArrayList<Vector> * seqs) {
  int n = a.n_cols();
  ArrayList<Vector> & s_ = *seqs;
  s_.Init();
  for (int i = 0; i < n; i++) {
    Vector seq;
    a.MakeColumnVector(i, &seq);
    s_.AddBackItem(seq);
  }
}

void mat2arrlstmat(int N, Matrix& a, ArrayList<Matrix> * seqs) {
  int n = a.n_cols();
  ArrayList<Matrix>& s_ = *seqs;
  s_.Init();
  for (int i = 0; i < n; i+=N) {
    Matrix b;
    a.MakeColumnSlice(i, N, &b);
    s_.AddBackItem(b);
  }
}
