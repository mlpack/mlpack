#include "fastlib/fastlib.h"

#define A 1

#define epsilon 1e-4


void save_correctly(const char *filename, Matrix a) {
  Matrix a_transpose;
  la::TransposeInit(a, &a_transpose);
  data::Save(filename, a_transpose);
}


void rand_vector(Vector &v) {

  index_t d = v.length();
  v.SetZero();
  
  for(index_t i = 0; i+1 < d; i+=2) {
    double a = drand48();
    double b = drand48();
    double first_term = sqrt(-2 * log(a));
    double second_term = 2 * M_PI * b;
    v[i] =   first_term * cos(second_term);
    v[i+1] = first_term * sin(second_term);
  }
  
  if((d % 2) == 1) {
    v[d - 1] = sqrt(-2 * log(drand48())) * cos(2 * M_PI * drand48());
  }

  la::Scale(1/sqrt(la::Dot(v, v)), &v);

}


void center(Matrix X, Matrix &X_centered) {
  Vector col_vector_sum;
  col_vector_sum.Init(X.n_rows());
  col_vector_sum.SetZero();
  
  index_t n = X.n_cols();
 
  for(index_t i = 0; i < n; i++) {
    Vector cur_col_vector;
    X.MakeColumnVector(i, &cur_col_vector);
    la::AddTo(cur_col_vector, &col_vector_sum);
  }

  la::Scale(1/(double)n, &col_vector_sum);

  X_centered.CopyValues(X);

  for(index_t i = 0; i < n; i++) {
    Vector cur_col_vector;
    X_centered.MakeColumnVector(i, &cur_col_vector);
    la::SubFrom(col_vector_sum, &cur_col_vector);
  }

}


void whiten(Matrix X, Matrix &whitening, Matrix &X_whitened) {
  Matrix X_transpose, X_cov, D, E, E_times_D;
  Vector D_vector;

  la::TransposeInit(X, &X_transpose);
  la::MulInit(X, X_transpose, &X_cov);

  la::Scale(1 / (double) (X.n_cols() - 1), &X_cov);

  X_cov.PrintDebug("X_cov");

  la::EigenvectorsInit(X_cov, &D_vector, &E);
  D.InitDiagonal(D_vector);
  
  index_t d = D.n_rows();
  for(index_t i = 0; i < d; i++) {
    D.set(i, i, pow(D.get(i, i), -.5));
  }

  la::MulInit(E, D, &E_times_D);
  la::MulTransBOverwrite(E_times_D, E, &whitening);
  
  la::MulOverwrite(whitening, X, &X_whitened);
}


void univariate_FastICA(Matrix X,
			double (*contrast_function)(double),
			Matrix fixed_subspace,
			index_t dim_num,
			double tolerance,
			Vector &w) {
  index_t n_fixed = dim_num;
  Vector w_old;
  index_t d = X.n_rows();
  index_t n = X.n_cols();
  rand_vector(w);

  

  w_old.Init(d);
  w_old.SetZero();



  bool converged = false;

  
  for(index_t epoch = 0; !converged; epoch++) {

    w.PrintDebug("w at beginning of epoch");

    printf("epoch %d\n", epoch);
    
    Vector first_sum;
    first_sum.Init(d);
    first_sum.SetZero();
    
    
    Vector tanh_dots;
    tanh_dots.Init(n);
    
    for(index_t i = 0; i < n; i++) {
      Vector x;
      X.MakeColumnVector(i, &x);
      tanh_dots[i] = contrast_function(la::Dot(w, x));
      //printf("%f\t", tanh_dots[i]);
      la::AddExpert(tanh_dots[i], x, &first_sum);
    }

    //first_sum.PrintDebug("first_sum");

    la::Scale((double)1/(double)n, &first_sum);

    printf("first_sum: %f %f\n", first_sum[0], first_sum[1]);
    
    double second_sum = 0;
    for(index_t i = 0; i < n; i++) {
      second_sum += A * (1 - (tanh_dots[i] * tanh_dots[i]));
    }
    la::Scale(-second_sum/(double)n, &w);

    printf("second_sum: %f\n", second_sum / (double)n);

    w.PrintDebug("w before adding first_sum");
    
    la::AddTo(first_sum, &w);

    w.PrintDebug("w after adding first_sum");

    // normalize
    la::Scale(1/sqrt(la::Dot(w, w)), &w);

    w.PrintDebug("before correction");
    
    // make orthogonal to fixed_subspace

    for(index_t i = 0; i < n_fixed; i++) {
      Vector w_i;
      fixed_subspace.MakeColumnVector(i, &w_i);
      
      la::AddExpert(-la::Dot(w_i, w), w_i, &w);

    }
    
    
    // normalize
    la::Scale(1/sqrt(la::Dot(w, w)), &w);
    

    w.PrintDebug("after correction");
    
    // check for convergence

    Vector w_diff;
    la::SubInit(w_old, w, &w_diff);

    if(la::Dot(w_diff, w_diff) < epsilon) {
      converged = true;
    }
    else {
      la::AddOverwrite(w_old, w, &w_diff);
    
      if(la::Dot(w_diff, w_diff) < epsilon) {
	converged = true;
      }
    }

    w_old.CopyValues(w);


  }

}


void deflationICA(Matrix X,
		  double (*contrast_function)(double),
		  double tolerance, Matrix &fixed_subspace) {
  index_t d = X.n_rows();

  printf("d = %d\n", d);
  
  for(index_t i = 0; i < d; i++) {
    printf("i = %d\n", i);
    Vector w;
    w.Init(d);
    univariate_FastICA(X, contrast_function, fixed_subspace, i, tolerance, w);
    Vector fixed_subspace_vector_i;
    fixed_subspace.MakeColumnVector(i, &fixed_subspace_vector_i);
    fixed_subspace_vector_i.CopyValues(w);
  }

  // fixed_subspace is the transpose of the postwhitening unmixing matrix W~
  // saving fixed_subspace really saves the transpose, so we simply save
  data::Save("fixed_subspace.dat", fixed_subspace);
  
}
  
double d_logcosh(double u) {
  return tanh(A * u);
}

double d_exp(double u) {
  return u * exp(-u*u/2);
}


int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  srand48(time(0));

  const char *data = fx_param_str(NULL, "data", NULL);

  Matrix X, X_centered, whitening, X_whitened;
  data::Load(data, &X);

  index_t d = X.n_rows(); // number of dimensions
  index_t n = X.n_cols(); // number of points

  X_centered.Init(d, n);
  X_whitened.Init(d, n);
  whitening.Init(d, d);

  printf("%d,%d\n", X.n_rows(), X.n_cols());

  printf("centering\n");
  center(X, X_centered);
 


  printf("whitening\n");

  whiten(X_centered, whitening, X_whitened);


  double tolerance = 1e-4;

  Matrix post_whitening_W_transpose;
  post_whitening_W_transpose.Init(d, d);
  

  printf("deflationICA\n");

  deflationICA(X_whitened, &d_logcosh, tolerance, post_whitening_W_transpose);
  
  Matrix post_whitening_W;
  la::TransposeInit(post_whitening_W_transpose, &post_whitening_W);




  Matrix W;
  la::MulInit(post_whitening_W, whitening, &W);


  Matrix Y;
  la::MulInit(post_whitening_W, X_whitened, &Y);

  save_correctly("post_whitening_W.dat", post_whitening_W);  
  save_correctly("whitening.dat", whitening);

  save_correctly("W.dat", W);

  save_correctly("X_centered.dat", X_centered);
  save_correctly("X_whitened.dat", X_whitened);
  save_correctly("Y.dat", Y);
  

  //  fx_done();

  return 0;
}
