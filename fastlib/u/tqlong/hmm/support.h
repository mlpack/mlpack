#ifndef FASTLIB_HMM_SUPPORT_H
#define FASTLIB_HMM_SUPPORT_H

#define RAND_UNIFORM_01 ((double) rand() / (double)RAND_MAX)
#define RAND_UNIFORM(a,b) (RAND_UNIFORM_01 * ((b)-(a)) + (a))
double RAND_NORMAL_01();
void RAND_NORMAL_01_INIT(int N, Vector* v);
void RAND_NORMAL_INIT(const Vector& mean, const Matrix& cov, Vector* v);

// return x'Ay
double MyMulExpert(const Vector& x, const Matrix& A, const Vector& y);

// compute normal PDF
double NORMAL_DENSITY(const Vector& x, const Vector& mean, const Matrix& inv_cov, double det_cov);

void print_matrix(const Matrix& a, const char* msg);

void print_vector(const Vector& a, const char* msg);

bool kmeans(const ArrayList<Matrix>& data, int num_clusters, 
	    ArrayList<int> *labels_, ArrayList<Vector> *cetroids_, 
	    int max_iter = 1000, double error_thresh = 1e-3);

bool kmeans(Matrix const &data, int num_clusters, 
	    ArrayList<int> *labels_, ArrayList<Vector> *cetroids_, 
	    int max_iter=1000, double error_thresh=1e-04);

void mat2arrlst(Matrix& a, ArrayList<Vector> * seqs);

void mat2arrlstmat(int N, Matrix& a, ArrayList<Matrix> * seqs);

#endif

