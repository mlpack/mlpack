#include "fastlib/fastlib.h"

#define A 1

#define epsilon 1e-4


void rand_vector(index_t n_dims, Vector &v) {
  
  v.Init(n_dims);
  v.SetZero();
  
  for(index_t i = 0; i+1 < n_dims; i+=2) {
    double a = drand48();
    double b = drand48();
    double first_term = sqrt(-2 * log(a));
    double second_term = 2 * M_PI * b;
    v[i] =   first_term * cos(second_term);
    v[i+1] = first_term * sin(second_term);
  }
  
  if((n_dims % 2) == 1) {
    v[n_dims - 1] = sqrt(-2 * log(drand48())) * cos(2 * M_PI * drand48());
  }
}


void center(Matrix X, Matrix &X_centered) {
  Vector col_vector_sum;
  col_vector_sum.Init(X.n_rows());
  col_vector_sum.SetZero();
  
  index_t num_points = X.n_cols();
 
  for(index_t i = 0; i < num_points; i++) {
    Vector cur_col_vector;
    X.MakeColumnVector(i, &cur_col_vector);
    la::AddTo(cur_col_vector, &col_vector_sum);
  }

  la::Scale(1/(double)num_points, &col_vector_sum);

  X_centered.Copy(X);

  for(index_t i = 0; i < num_points; i++) {
    Vector cur_col_vector;
    X_centered.MakeColumnVector(i, &cur_col_vector);
    la::SubFrom(col_vector_sum, &cur_col_vector);
  }

}


void whiten(Matrix X, Matrix &X_whitened) {
  Matrix X_transpose, X_squared, D, E, E_transpose, X_temp, X_temp2;
  Vector D_vector;

  la::TransposeInit(X, &X_transpose);
  la::MulInit(X, X_transpose, &X_squared);

  la::EigenvectorsInit(X_squared, &D_vector, &E);
  la::TransposeInit(E, &E_transpose);
  D.InitDiagonal(D_vector);
  index_t n_dims = D.n_rows();
  for(index_t i = 0; i < n_dims; i++) {
    D.set(i, i, pow(D.get(i, i), -.5));
  }

  la::MulInit(E, D, &X_temp);
  la::MulInit(X_temp, E_transpose, &X_temp2);
  la::MulInit(X_temp2, X, &X_whitened);
}


void univariate_FastICA(Matrix X,
			double (*contrast_function)(double),
			Matrix fixed_subspace,
			index_t dim_num,
			double tolerance,
			Vector &w) {
  index_t n_fixed = dim_num;
  Vector w_old;
  index_t n_dims = X.n_rows();
  index_t n_points = X.n_cols();
  rand_vector(n_dims, w);

  w_old.Init(n_dims);
  w_old.SetZero();



  bool converged = false;

  
  for(index_t epoch = 0; !converged; epoch++) {
    
    Vector first_sum;
    first_sum.Init(n_dims);
    first_sum.SetZero();
    
    
    Vector tanh_dots;
    tanh_dots.Init(n_points);
    
    for(index_t i = 0; i < n_points; i++) {
      Vector x;
      X.MakeColumnVector(i, &x);
      tanh_dots[i] = contrast_function(la::Dot(w, x));
      la::AddExpert(tanh_dots[i], x, &first_sum);
    }
    la::Scale(1/(double)n_points, &first_sum);
    
    double second_sum = 0;
    for(index_t i = 0; i < n_points; i++) {
      second_sum += A * (1 - (tanh_dots[i] * tanh_dots[i]));
    }
    la::Scale(-second_sum/(double)n_points, &w);
    
    la::AddTo(first_sum, &w);
    
    // make orthogonal to fixed_subspace

    for(index_t i = 0; i < n_fixed; i++) {
      Vector w_i;
      fixed_subspace.MakeColumnVector(i, &w_i);
      
      la::AddExpert(-la::Dot(w_i, w), w_i, &w);
      
    }
    
    
    // normalize
    la::Scale(1/sqrt(la::Dot(w, w)), &w);
    
    
    
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
		  double tolerance) {
  Matrix fixed_subspace;
  index_t n_dims = X.n_rows();
  fixed_subspace.Init(n_dims, n_dims);
  
  for(index_t i = 0; i < n_dims; i++) {
    Vector w;
    univariate_FastICA(X, contrast_function, fixed_subspace, i, tolerance, w);
    Vector fixed_subspace_vector_i;
    fixed_subspace.MakeColumnVector(i, &fixed_subspace_vector_i);
    fixed_subspace_vector_i.CopyValues(w);
  }

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

  Matrix X, X_centered, X_whitened;
  data::Load(data, &X);

  //index_t n_dims = X.n_rows();
  //index_t n_points = X.n_cols();

  center(X, X_centered);

  data::Save("X_centered.dat", X_centered);

  whiten(X_centered, X_whitened);

  data::Save("X_whitened.dat", X_whitened);

  double tolerance = 1e-4;

  deflationICA(X_whitened, &d_logcosh, tolerance);




  //  fx_done();

  return 0;
}
