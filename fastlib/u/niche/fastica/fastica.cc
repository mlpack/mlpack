#include "fastlib/fastlib.h"

#define A 1

#define epsilon 1e-4

#define LOGCOSH 0
#define EXP 1

// n indicates number of points
// d indicates number of dimensions (number of components or variables)


namespace {


  void SaveCorrectly(const char *filename, Matrix a) {
    Matrix a_transpose;
    la::TransposeInit(a, &a_transpose);
    data::Save(filename, a_transpose);
  }
  

  void RandVector(Vector &v) {

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


  void Center(Matrix X, Matrix &X_centered) {
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


  void Whiten(Matrix X, Matrix &whitening, Matrix &X_whitened) {
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


  void UnivariateFastICA(Matrix X, int contrast_type, Matrix fixed_subspace,
			 index_t dim_num, double tolerance, Vector &w) {
    index_t d = X.n_rows();
    index_t n = X.n_cols();

    RandVector(w);

  
    Vector w_old;
    w_old.Init(d);
    w_old.SetZero();



    bool converged = false;

  
    for(index_t epoch = 0; !converged; epoch++) {

      printf("\nEPOCH %"LI"d\n", epoch);

      w.PrintDebug("w at beginning of epoch");

    


      Vector first_sum;
      first_sum.Init(d);
      first_sum.SetZero();

      double second_sum = 0;

      
      if(contrast_type == LOGCOSH) {

	Vector first_deriv_dots;
	first_deriv_dots.Init(n);
	
	for(index_t i = 0; i < n; i++) {
	  Vector x;
	  X.MakeColumnVector(i, &x);
	  first_deriv_dots[i] = tanh(A * la::Dot(w, x));
	  la::AddExpert(first_deriv_dots[i], x, &first_sum);
	}
	la::Scale(1/(double)n, &first_sum);
	
	
	for(index_t i = 0; i < n; i++) {
	  second_sum += first_deriv_dots[i] * first_deriv_dots[i];
	}
	
	second_sum *= A / (double) n;
	second_sum -= A;
	

      }
      else if(contrast_type == EXP) {

	Vector dots;
	Vector exp_dots;
	dots.Init(n);
	exp_dots.Init(n);

	for(index_t i = 0; i < n; i++) {
	  Vector x;
	  X.MakeColumnVector(i, &x);

	  double dot = la::Dot(w, x);
	  dots[i] = dot;
	  exp_dots[i] = exp(-dot * dot/2);

	  la::AddExpert(dot * exp_dots[i], x, &first_sum);
	}
	la::Scale(1/(double)n, &first_sum);
	
	
	for(index_t i = 0; i < n; i++) {
	  second_sum += exp_dots[i] * (dots[i] * dots[i] - 1);
	}
	
	second_sum /= (double) n;

      }
      else {
	printf("ERROR: invalid contrast function: contrast_type = %d\n",
	       contrast_type);
	exit(SUCCESS_FAIL);
      }

      
      la::Scale(second_sum, &w);
      la::AddTo(first_sum, &w);

      first_sum.PrintDebug("first_sum");
      printf("second_sum = %f\n", second_sum);



      // normalize
      la::Scale(1/sqrt(la::Dot(w, w)), &w);

      w.PrintDebug("before correction");
    
      // make orthogonal to fixed_subspace

      for(index_t i = 0; i < dim_num; i++) {
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


  void DeflationICA(Matrix X, int contrast_type,
		     double tolerance, Matrix &fixed_subspace) {
    index_t d = X.n_rows();

    printf("%"LI"d Components\n", d);
  
    for(index_t i = 0; i < d; i++) {
      printf("\n\nExtracting component %"LI"d\n", i);
      Vector w;
      w.Init(d);
      UnivariateFastICA(X, contrast_type, fixed_subspace, i, tolerance, w);
      Vector fixed_subspace_vector_i;
      fixed_subspace.MakeColumnVector(i, &fixed_subspace_vector_i);
      fixed_subspace_vector_i.CopyValues(w);
    }
  
  }
  
  double DLogCosh(double u) {
    return tanh(A * u);
  }

  double DExp(double u) {
    return u * exp(-u*u/2);
  }

}

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  srand48(time(0));

  const char *data = fx_param_str(NULL, "data", NULL);

  Matrix X, X_centered, whitening, X_whitened;
  data::Load(data, &X);

  index_t d = X.n_rows(); // number of dimensions
  index_t n = X.n_cols(); // number of points

  printf("d = %d, n = %d\n", d, n);


  
  X_centered.Init(d, n);
  X_whitened.Init(d, n);
  whitening.Init(d, d);
  

  printf("centering\n");
  Center(X, X_centered);
 


  printf("whitening\n");

  Whiten(X_centered, whitening, X_whitened);


  double tolerance = 1e-4;

  Matrix post_whitening_W_transpose;
  post_whitening_W_transpose.Init(d, d);
  

  printf("deflation ICA\n");

  DeflationICA(X_whitened, EXP, tolerance, post_whitening_W_transpose);
  
  Matrix post_whitening_W;
  la::TransposeInit(post_whitening_W_transpose, &post_whitening_W);




  Matrix W;
  la::MulInit(post_whitening_W, whitening, &W);


  Matrix Y;
  la::MulInit(post_whitening_W, X_whitened, &Y);

  SaveCorrectly("post_whitening_W.dat", post_whitening_W);  
  SaveCorrectly("whitening.dat", whitening);

  SaveCorrectly("W.dat", W);

  SaveCorrectly("X_centered.dat", X_centered);
  SaveCorrectly("X_whitened.dat", X_whitened);
  SaveCorrectly("Y.dat", Y);
  

  //  fx_done();
  
  return 0;
}
