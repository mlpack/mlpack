#include "fastlib/fastlib.h"
#include "lin_alg.h"

#define LOGCOSH 0
#define GAUSS 10
#define KURTOSIS 20
#define SKEW 30

#define SYMMETRIC 0
#define DEFLATION 1


/**
 * TODO: This is the raison d'être of this class
 *
 * Example use:
 *
 * @code
 * TODO: write an example use
 */

class FastICA {
  
 private:
  int approach_;
  int nonlinearity_;
  //const index_t first_eig;
  //const index_t last_eig;
  index_t num_of_IC_;
  bool fine_tune_;
  double a1_;
  double a2_;
  double mu_;
  bool stabilization_;
  double epsilon_;
  index_t max_num_iterations_;
  index_t max_fine_tune_;
  double percent_cut_;
  Matrix X_;




  void SymmetricLogCoshUpdate_(index_t n, Matrix X, Matrix *B) {
    Matrix hyp_tan, col_vector, sum, temp1, temp2;
    
    MapOverwrite(&TanhArg,
		 a1_,
		 MulTransAInit(&X, B, &hyp_tan));
    
    
    ColVector(d, a1_, &col_vector);
    
    Scale(1 / (double) n,
	  AddTo(MulInit(&X, &hyp_tan, &temp1),
		DotMultiplyOverwrite(MulInit(&col_vector,
					     MapOverwrite(&MinusArg,
							  n,
							  MatrixMapSum(&Square, 0, &hyp_tan, &sum)),
					     &temp2),
				     B)));
  }


  void SymmetricLogCoshFineTuningUpdate_(index_t n, Matrix X, Matrix *B) {
    Matrix Y, hyp_tan, Beta, Beta_Diag, D, sum, temp1, temp2, temp3;
	
    MulTransAInit(&X, B, &Y);
    MapInit(&TanhArg, a1_, &Y, &hyp_tan);
    DotMultiplySum(&Y, &hyp_tan, &Beta);
    VectorToDiag(MapOverwrite(&Inv,
			      0,
			      AddTo(&Beta,
				    Scale(a1_,
					  MapOverwrite(&MinusArg,
						       n,
						       MatrixMapSum(&Square, 0, &hyp_tan, &sum))))),
		 &D);
    
    AddExpert(mu_, 
	      MulInit(MulInit(B,
			      SubFrom(VectorToDiag(&Beta, &Beta_Diag),
				      MulTransAInit(&Y, &hyp_tan, &temp1)),
			      &temp2),
		      &D,
		      &temp3),
	      B);
  }


  void SymmetricGaussUpdate_(index_t n, Matrix X, Matrix* B) {
    Matrix U, U_squared, ex, col_vector, sum, temp1, temp2;
    
    MulTransAInit(&X, B, &U);
    
    MapInit(&Square, 0, &U, &U_squared);
    MapInit(&ExpArg, -a2_ / 2, &U_squared, &ex);
    DotMultiplyOverwrite(&ex, &U);
    //U is gauss
    AddTo(DotMultiplyOverwrite(&ex,
			       Scale(-a2_, &U_squared)),
	  &ex);
    //ex is dGauss
    
    ColVector(d, a2_, &col_vector);
    
    Scale(1 / (double) n,
	  SubOverwrite(MulInit(&X, &U, &temp1),
		       DotMultiplyOverwrite(B,
					    MulInit(&col_vector,
						    Sum(&ex, &sum),
						    &temp2)),
		       B));
  }


  void SymmetricGaussFineTuningUpdate_(index_t n, Matrix X, Matrix* B) {
    Matrix Y, Y_squared_a2, ex, gauss, D, Beta, temp1, temp2, temp3;
    Vector Beta_vector, sum_vector;
	
    MulTransAInit(&X, B, &Y);
    MapInit(&SquareArg, a2_, &Y, &Y_squared_a2);
    MapInit(&ExpArg, -.5, &Y_squared_a2, &ex);
    DotMultiplyInit(&Y, &ex, &gauss);
	
    Beta_vector.Init(d);
    double *Y_col_j;
    double *gauss_col_j;
    for(index_t j = 0; j < d; j++) {
      Y_col_j = Y.GetColumnPtr(j);
      gauss_col_j = gauss.GetColumnPtr(j);
      double sum = 0;
      for(index_t i = 0; i < n; i++) {
	sum += Y_col_j[i] * gauss_col_j[i];
      }
      Beta_vector[j] = sum;
    }

	
    sum_vector.Init(d);
    double *Y_squared_a2_col_j;
    double *ex_col_j;
    for(index_t j = 0; j < d; j++) {
      Y_squared_a2_col_j = Y_squared_a2.GetColumnPtr(j);
      ex_col_j = ex.GetColumnPtr(j);
      double sum = 0;
      for(index_t i = 0; i < n; i++) {
	sum += (Y_squared_a2_col_j[i] - 1) * ex_col_j[i];
      }
      sum_vector[j] = sum;
    }


    //D = diag(1 ./ (Beta + sum((Y_squared_a2 - 1) .* ex)))
    VectorToDiag(MapOverwrite(&Inv,
			      0,
			      AddTo(&Beta_vector, &sum_vector)),
		 &D);
	  
    //B = B + myy * B * (Y' * gauss - diag(Beta)) * D;
    AddExpert(mu_,
	      MulInit(MulInit(B,
			      SubFrom(VectorToDiag(&Beta_vector, &Beta),
				      MulTransAInit(&Y, &gauss, &temp1)),
			      &temp2),
		      &D,
		      &temp3),
	      B);
  }


  void SymmetricKurtosisUpdate_(index_t n, Matrix X, Matrix* B) {
    Matrix temp1, temp2;

    Scale(1 / (double) n,
	  MulInit(&X,
		  MapOverwrite(&pow,
			       3,
			       MulTransAInit(&X, B, &temp1)),
		  &temp2));
	
    AddTo(&temp2,
	  Scale(-3, B));
  }


  void SymmetricKurtosisFineTuningUpdate_(index_t n, Matrix X, Matrix* B) {
    Matrix Y, G_pow_3, Beta, Beta_Diag, D_vector, D, temp1, temp2, temp3;

    MulTransAInit(&X, B, &Y);
    MapInit(&pow, 3, &Y, &G_pow_3);

    DotMultiplySum(&Y, &G_pow_3, &Beta);

    VectorToDiag(MapOverwrite(&Inv,
			      0,
			      MapInit(&Plus,
				      -3 * n,
				      &Beta,
				      &D_vector)),
		 &D);

    AddExpert(mu_, 
	      MulInit(MulInit(B,
			      SubFrom(VectorToDiag(&Beta, &Beta_Diag),
				      MulTransAInit(&Y, &G_pow_3, &temp1)),
			      &temp2),
		      &D,
		      &temp3),
	      B);
  }


  void SymmetricSkewUpdate_(index_t n, Matrix X, Matrix* B) {
    Matrix temp1;

    Scale(1 / (double) n,
	  MulOverwrite(&X,
		       MapOverwrite(&Square,
				    0,
				    MulTransAInit(&X, B, &temp1)),
		       B));

  }


  void SymmetricSkewFineTuningUpdate_(index_t n, Matrix X, Matrix* B) {
    Matrix Y, G_skew, Beta, Beta_Diag, D_vector, D, temp1, temp2, temp3;
	
    MulTransAInit(&X, B, &Y);
    MapInit(&Square, 0, &Y, &G_skew);
    DotMultiplySum(&Y, &G_skew, &Beta);
    VectorToDiag(MapInit(&Inv, 0, &Beta, &D_vector),
		 &D);
	
    AddExpert(mu_, 
	      MulInit(MulInit(B,
			      SubFrom(VectorToDiag(&Beta, &Beta_Diag),
				      MulTransAInit(&Y, &G_skew, &temp1)),
			      &temp2),
		      &D,
		      &temp3),
	      B);
  }

  
  void DeflationLogCoshUpdate_(index_t n, Matrix X, Vector* w) {
    Vector hyp_tan, temp1;
    
    MapOverwrite(&TanhArg,
		 a1_,
		 MulInit(w, &X, &hyp_tan));
        
    Scale(1 / (double) n,
	  AddTo(MulInit(&X, &hyp_tan, &temp1),
		Scale(a1_ * (VectorMapSum(&Square, 0, &hyp_tan) - n),
		      w)));
  }


  void DeflationLogCoshFineTuningUpdate_(index_t n, Matrix X, Vector* w) {
    Vector hyp_tan, X_hyp_tan, Beta_w, temp1;
    
    MapOverwrite(&TanhArg,
		 a1_,
		 MulInit(w, &X, &hyp_tan));
    
    MulInit(&X, &hyp_tan, &X_hyp_tan);
    double Beta = la::Dot(X_hyp_tan, *w);
    
    AddExpert(mu_,
	      Scale(1 / (a1_ * (VectorMapSum(&Square, 0, &hyp_tan) - n) + Beta),
		    SubInit(&X_hyp_tan,
			    ScaleInit(Beta, w, &Beta_w),
			    &temp1)),
	      w);
  }


  void DeflationGaussUpdate_(index_t n, Matrix X, Vector* w) {
    Vector u, u_squared, ex, temp1;

    MulInit(w, &X, &u);
    MapInit(&Square, 0, &u, &u_squared);
    MapInit(&ExpArg, -.5 * a2_, &u_squared, &ex);
    DotMultiplyOverwrite(&ex, &u);
    //u is gauss
    AddTo(DotMultiplyOverwrite(&ex,
			       Scale(-a2_, &u_squared)),
	  &ex);
    //ex is dGauss
    Scale(1 / (double) n,
	  AddTo(MulInit(&X, &u, &temp1),
		Scale(-1 * Sum(&ex), w)));
  }
  
  
  void DeflationGaussFineTuningUpdate_(index_t n, Matrix X, Vector* w) {
    Vector u, u_squared, ex, X_gauss, Beta_w, temp1;

    MulInit(w, &X, &u);
    MapInit(&Square, 0, &u, &u_squared);
    MapInit(&ExpArg, -.5 * a2_, &u_squared, &ex);
    DotMultiplyOverwrite(&ex, &u);
    //u is gauss
    AddTo(DotMultiplyOverwrite(&ex,
			       Scale(-a2_, &u_squared)),
	  &ex);
    //ex is dGauss

    MulInit(&X, &u, &X_gauss);
    double Beta = la::Dot(X_gauss, *w);

    AddExpert(mu_,
	      Scale(1 / (Beta - Sum(&ex)),
		    SubInit(&X_gauss,
			    ScaleInit(Beta, w, &Beta_w),
			    &temp1)),
	      w);
  }

  
  void DeflationKurtosisUpdate_(index_t n, Matrix X, Vector* w) {
    Vector temp1, temp2;

    Scale(1 / (double) n,
	  MulInit(&X,
		  MapOverwrite(&pow,
			       3,
			       MulInit(w, &X, &temp1)),
		  &temp2));
	    
    AddTo(&temp2,
	  Scale(-3, w));
  }
  
  
  void DeflationKurtosisFineTuningUpdate_(index_t n, Matrix X, Vector* w) {
    Vector EXG_pow_3, Beta_w, temp1;
	    
    Scale(1 / (double) n,
	  MulInit(&X,
		  MapOverwrite(&pow,
			       3,
			       MulInit(w, &X, &temp1)),
		  &EXG_pow_3));

    double Beta = la::Dot(*w, EXG_pow_3);
	    
    AddExpert(mu_ / (Beta - 3),
	      SubFrom(ScaleInit(Beta, w, &Beta_w),
		      &EXG_pow_3),
	      w);
  }

  
  void DeflationSkewUpdate_(index_t n, Matrix X, Vector* w) {
    Vector temp1;
	    
    Scale(1 / (double) n,
	  MulInit(&X,
		  MapOverwrite(&Square,
			       0,
			       MulInit(w, &X, &temp1)),
		  w));
  }

  
  void DeflationSkewFineTuningUpdate_(index_t n, Matrix X, Vector* w) {
    Vector EXG_skew, Beta_w, temp1;
	    
    Scale(1 / (double) n,
	  MulInit(&X,
		  MapOverwrite(&Square,
			       0,
			       MulInit(w, &X, &temp1)),
		  &EXG_skew));

    double Beta = la::Dot(*w, EXG_skew);
	    
    AddExpert(mu_ / Beta,
	      SubFrom(ScaleInit(Beta, w, &Beta_w),
		      &EXG_skew),
	      w);
  }
  
  
  

    
    
  
  public:

  index_t d;
  index_t n;

  int approach() {
    return approach_;
  }

  int nonlinearity() {
    return nonlinearity;
  }

  //  index_t first_eig() {
  //    return first_eig_;
  //  }

  //  index_t last_eig() {
  //    return last_eig_;
  //  }

  index_t num_of_IC() {
    return num_of_IC_;
  }

  bool fine_tune() {
    return fine_tune_;
  }

  double a1() {
    return a1_;
  }

  double a2() {
    return a2_;
  }

  double mu() {
    return mu_;
  }

  bool stabilization() {
    return stabilization_;
  }

  double epsilon() {
    return epsilon_;
  }

  index_t max_num_iterations() {
    return max_num_iterations_;
  }

  index_t max_fine_tune() {
    return max_fine_tune_;
  }

  double percent_cut() {
    return percent_cut_;
  }

  double X() {
    return X_;
  }


  /** 
   * Default constructor does nothing special
   */
  FastICA() {
  }

  /**
   * Pass in the data matrix
   */
  void Init(Matrix X_in) {
    X_.Copy(X_in); // for some reason Alias makes this crash, so copy for now
    d = X_.n_rows();
    n = X_.n_cols();
  }


  // NOTE: these functions currently are public because some of them can
  //       serve as utilities that actually should be moved to lin_alg.h
 
 

  /**
   * Orthogonalize W and return the result in W, using Eigen Decomposition
   * @pre W and W_old store the same matrix in disjoint memory
   */
  void Orthogonalize(const Matrix W_old, Matrix *W) {
    Matrix W_squared, W_squared_inv_sqrt;
    
    la::MulTransAInit(W_old, W_old, &W_squared);
    
    Matrix D, E, E_times_D;
    Vector D_vector;
    
    la::EigenvectorsInit(W_squared, &D_vector, &E);
    D.InitDiagonal(D_vector);
    
    index_t d = D.n_rows();
    for(index_t i = 0; i < d; i++) {
      D.set(i, i, 1 / sqrt(D.get(i, i)));
    }
    
    la::MulInit(E, D, &E_times_D);
    la::MulTransBInit(E_times_D, E, &W_squared_inv_sqrt);
    
    // note that up until this point, W == W_old
    la::MulOverwrite(W_old, W_squared_inv_sqrt, W);
  }
  

  /**
   * Select indices < max according to probability equal to parameter
   * percentage, and return indices in a Vector
   * @pre selected_indices is an uninitialized Vector, percentage in [0 1]
   */
  index_t GetSamples(int max, double percentage, Vector *selected_indices) {   
    
    index_t num_selected = 0;
    Vector rand_nums;
    rand_nums.Init(max);
    for(index_t i = 0; i < max; i++) {
      double rand_num = drand48();
      rand_nums[i] = rand_num;
      if(rand_num <= percentage) {
	num_selected++;
      }
    }

    selected_indices -> Init(num_selected);
    
    int j = 0;
    for(index_t i = 0; i < max; i++) {
      if(rand_nums[i] <= percentage) {
	(*selected_indices)[j] = i;
	j++;
      }
    }

    return num_selected;
  }


  /**
   * Return Select indices < max according to probability equal to parameter
   * percentage, and return indices in a Vector
   * @pre selected_indices is an uninitialized Vector, percentage in [0 1]
   */
  index_t RandomSubMatrix(index_t n, double percent_cut, Matrix X, Matrix* X_sub) {
    Vector selected_indices;
    index_t num_selected = GetSamples(n, percent_cut, &selected_indices);
    MakeSubMatrixByColumns(selected_indices, X, X_sub);
    return num_selected;
  }


  /**
   * Run the fixed point iterative component of FastICA
   * @pre{ X is a d by n data matrix, for d dimensions and n samples}
   */
  int FixedPointICA(Matrix X, Matrix whitening_matrix, Matrix dewhitening_matrix,
		    Matrix* A, Matrix* W) {
    // ensure default values are passed into this function if the user doesn't care about certain parameters

    int g = nonlinearity_;
    
    if(d < num_of_IC_) {
      printf("ERROR: must have num_of_IC <= Dimension!\n");
      W -> Init(0,0);
      A -> Init(0,0);
      return SUCCESS_FAIL;
    }

    W -> Init(d, num_of_IC_);
    A -> Init(num_of_IC_, d);

    if((percent_cut_ > 1) || (percent_cut_ < 0)) {
      percent_cut_ = 1;
      printf("Setting percent_cut to 1\n");
    }
    else if(percent_cut_ < 1) {
      if((percent_cut_ * n) < 1000) {
	percent_cut_ = min(1000 / (double) n, (double) 1);
	printf("Warning: Setting percent_cut to %0.3f (%d samples).\n", percent_cut_,
	       (int) floor(percent_cut_ * n));
      }
    }
    
    int g_orig = g;

    if(percent_cut_ != 1) {
      g_orig += 2;
    }
    
    if(mu_ != 1) {
      g_orig += 1;
    }

    bool fine_tuning_enabled = true;
    int g_fine;

    if(fine_tune_) {
      g_fine = g + 1;
    }
    else {
      if(mu_ != 1) {
	g_fine = g_orig;
      }
      else {
	g_fine = g_orig + 1;
      }

      fine_tuning_enabled = false;
    }

    bool stabilization_enabled;
    if(stabilization_) {
      stabilization_enabled = true;
    }
    else {
      if(mu_ != 1) {
	stabilization_enabled = true;
      }
      else {
	stabilization_enabled = false;
      }
    }

    double mu_orig = mu_;
    double mu_k = 0.01;
    index_t failure_limit = 5;
    int used_nonlinearity = g_orig;
    double stroke = 0;
    bool not_fine = true;
    bool taking_long = false;

    // currently we don't allow for guesses for the initial unmixing matrix B
    int initial_state_mode = 0;

    Matrix B;
  
    if(approach_ == SYMMETRIC) {
      printf("using Symmetric approach\n");

      if(initial_state_mode == 0) {
	//generate random B
	B.Init(d, num_of_IC_);

	
	for(index_t i = 0; i < num_of_IC_; i++) {
	  Vector b;
	  B.MakeColumnVector(i, &b);
	  RandVector(b);
	}
	
	/*
	B.SetZero();
	for(index_t i = 0; i < d; i++) {
	  B.set(i, i, 1);
	}
	*/
	  
      }
      

      Matrix B_old, B_old2;

      B_old.Init(d, num_of_IC_);
      B_old2.Init(d, num_of_IC_);

      B_old.SetZero();
      B_old2.SetZero();


      for(index_t round = 1; round <= (max_num_iterations_ + 1); round++) {
	if(round == (max_num_iterations_ + 1)) {
	  printf("No convergence after %d steps\n", max_num_iterations_);
	
	
	  // orthogonalize B via: newB = B * (B' * B) ^ -.5;
	  Matrix temp;
	  temp.Copy(B);
	  Orthogonalize(temp, &B);

	  MulTransAOverwrite(&B, &whitening_matrix, W);
	  MulOverwrite(&dewhitening_matrix, &B, A);
	  return SUCCESS_PASS;
	}
	
	{
	  Matrix temp;
	  temp.Copy(B);
	  Orthogonalize(temp, &B);
	  B.PrintDebug("B");
	}

	Matrix B_delta_cov;
	MulTransAInit(&B, &B_old, &B_delta_cov);
	double min_abs_cos = DBL_MAX;
	for(index_t i = 0; i < d; i++) {
	  double current_cos = fabs(B_delta_cov.get(i, i));
	  if(current_cos < min_abs_cos) {
	    min_abs_cos = current_cos;
	  }
	}
      
	printf("min_abs_cos = %f\n", min_abs_cos);

	if(1 - min_abs_cos < epsilon_) {
	  if(fine_tuning_enabled && not_fine) {
	    not_fine = false;
	    used_nonlinearity = g_fine;
	    mu_ = mu_k * mu_orig;
	    B_old.SetZero();
	    B_old2.SetZero();
	  }
	  else {
	    MulOverwrite(&dewhitening_matrix, &B, A);
	    MulTransAOverwrite(&B, &whitening_matrix, W);
	    return SUCCESS_PASS;
	  }
	}
	else if(stabilization_enabled) {

	  Matrix B_delta_cov2;
	  MulTransAInit(&B, &B_old2, &B_delta_cov2);
	  double min_abs_cos2 = DBL_MAX;
	  for(index_t i = 0; i < d; i++) {
	    double current_cos2 = fabs(B_delta_cov2.get(i, i));
	    if(current_cos2 < min_abs_cos2) {
	      min_abs_cos2 = current_cos2;
	    }
	  }

	  if((stroke == 0) && (1 - min_abs_cos2 < epsilon_)) {
	    stroke = mu_;
	    mu_ *= .5;
	    if((used_nonlinearity % 2) == 0) {
	      used_nonlinearity += 1;
	    }
	  }
	  else if(stroke > 0) {
	    mu_ = stroke;
	    stroke = 0;

	    if((mu_ == 1) && ((used_nonlinearity % 2) != 0)) {
	      used_nonlinearity -= 1;
	    }
	  }
	  else if((!taking_long) &&
		  (round > ((double) max_num_iterations_ / 2))) {
	    taking_long = true;
	    mu_ *= .5;
	    if((used_nonlinearity % 2) == 0) {
	      used_nonlinearity += 1;
	    }
	  }
	}

	B_old2.CopyValues(B_old);
	B_old.CopyValues(B);

	// show progress here, (the lack of code means no progress shown for now)



	// use Newton-Raphson to update B

	printf("used_nonlinearity = %d\n", used_nonlinearity);

	switch(used_nonlinearity) {
	  
	case LOGCOSH: {
	  SymmetricLogCoshUpdate_(n, X, &B);
	  break;
	}
	
	case LOGCOSH + 1: {
	  SymmetricLogCoshFineTuningUpdate_(n, X, &B);
	  break;	
	}
	
	case LOGCOSH + 2: {
	  Matrix X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut_, X, &X_sub);
	  SymmetricLogCoshUpdate_(num_selected, X_sub, &B);
	  break;
	}
	
	case LOGCOSH + 3: {
	  Matrix X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut_, X, &X_sub);
	  SymmetricLogCoshFineTuningUpdate_(num_selected, X_sub, &B);
	  break;
	}

	case GAUSS: {
	  SymmetricGaussUpdate_(n, X, &B);
	  break;
	}
	
	case GAUSS + 1: {
	  SymmetricGaussFineTuningUpdate_(n, X, &B);
	  break;
	}

	case GAUSS + 2: {
	  Matrix X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut_, X, &X_sub);
	  SymmetricGaussUpdate_(num_selected, X_sub, &B);
	  break;
	}

	case GAUSS + 3: {
	  Matrix X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut_, X, &X_sub);
	  SymmetricGaussFineTuningUpdate_(num_selected, X_sub, &B);
	  break;
	}

	case KURTOSIS: {
	  SymmetricKurtosisUpdate_(n, X, &B);
	  break;
	}
	
	case KURTOSIS + 1: {
	  SymmetricKurtosisFineTuningUpdate_(n, X, &B);
	  break;
	}

	case KURTOSIS + 2: {
	  Matrix X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut_, X, &X_sub);
	  SymmetricKurtosisUpdate_(num_selected, X_sub, &B);
	  break;
	}

	case KURTOSIS + 3: {
	  Matrix X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut_, X, &X_sub);
	  SymmetricKurtosisFineTuningUpdate_(num_selected, X_sub, &B);
	  break;
	}

	case SKEW: {
	  SymmetricSkewUpdate_(n, X, &B);
	  break;
	}
	
	case SKEW + 1: {
	  SymmetricSkewFineTuningUpdate_(n, X, &B);
	  break;
	}

	case SKEW + 2: {
	  Matrix X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut_, X, &X_sub);
	  SymmetricSkewUpdate_(num_selected, X_sub, &B);
	  break;
	}
	  
	case SKEW + 3: {
	  Matrix X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut_, X, &X_sub);
	  SymmetricSkewFineTuningUpdate_(num_selected, X_sub, &B);
	  break;
	}
	  
	default:
	  printf("ERROR: invalid contrast function: used_nonlinearity = %d\n",
		 used_nonlinearity);
	  exit(SUCCESS_FAIL);
	  
	}
      }
    }
    else if(approach_ == DEFLATION) {
      printf("using Deflation approach\n");
      B.Init(d, d);
      B.SetZero();

      index_t round = 0;

      index_t num_failures = 0;

      while(round < num_of_IC_) {
	mu_ = mu_orig;
	used_nonlinearity = g_orig;
	stroke = 0;
	not_fine = true;
	taking_long = false;
	int end_fine_tuning = 0;
	
	Vector w;
	if(initial_state_mode == 0) {
	  w.Init(d);
	  RandVector(w);
	}

	for(index_t i = 0; i < round; i++) {
	  Vector b_i;
	  B.MakeColumnVector(i, &b_i);
	  la::AddExpert(-la::Dot(b_i, w), b_i, &w);
	}
	la::Scale(1/sqrt(la::Dot(w, w)), &w); // normalize

	Vector w_old, w_old2;
	w_old.Init(d);
	w_old.SetZero();
	w_old2.Init(d);
	w_old2.SetZero();

	index_t i = 1;
	index_t gabba = 1;
	while(i <= max_num_iterations_ + gabba) {

	  for(index_t j = 0; j < round; j++) {
	    Vector b_j;
	    B.MakeColumnVector(j, &b_j);
	    la::AddExpert(-la::Dot(b_j, w), b_j, &w);
	  }
	  la::Scale(1/sqrt(la::Dot(w, w)), &w); // normalize

	  if(not_fine) {
	    if(i == (max_num_iterations_ + 1)) {
	      round++;
	      num_failures++;
	      if(num_failures > failure_limit) {
		printf("Too many failures to converge (%d). Giving up.\n", num_failures);
		return SUCCESS_FAIL;
	      }
	      break;
	    }
	  }
	  else {
	    if(i >= end_fine_tuning) {
	      w_old.Copy(w);
	    }
	  }

	  // check for convergence
	  bool converged = false;
	  Vector w_diff;
	  la::SubInit(w_old, w, &w_diff);
	  
	  if(la::Dot(w_diff, w_diff) < epsilon_) {
	    converged = true;
	  }
	  else {
	    la::AddOverwrite(w_old, w, &w_diff);
	    
	    if(la::Dot(w_diff, w_diff) < epsilon_) {
	      converged = true;
	    }
	  }

	  if(converged) {
	    if(fine_tuning_enabled & not_fine) {
	      not_fine = false;
	      gabba = max_fine_tune_;
	      w_old.SetZero();
	      w_old2.SetZero();
	      used_nonlinearity = g_fine;
	      mu_ = mu_k * mu_orig;

	      end_fine_tuning = max_fine_tune_ + i;
	    }
	    else {
	      num_failures = 0;
	      Vector B_col_round, A_col_round, W_col_round;

	      B.MakeColumnVector(round, &B_col_round);
	      A -> MakeColumnVector(round, &A_col_round);
	      W -> MakeColumnVector(round, &W_col_round);

	      B_col_round.CopyValues(w);
	      la::MulOverwrite(dewhitening_matrix, w, &A_col_round);
	      la::MulOverwrite(w, whitening_matrix, &W_col_round);

	      break; // this line is intended to take us to the next IC
	    }
	  }
	  else if(stabilization_enabled) {
	    converged = false;
	    la::SubInit(w_old2, w, &w_diff);
	    
	    if(la::Dot(w_diff, w_diff) < epsilon_) {
	      converged = true;
	    }
	    else {
	      la::AddOverwrite(w_old2, w, &w_diff);
	      
	      if(la::Dot(w_diff, w_diff) < epsilon_) {
		converged = true;
	      }
	    }
	    
	    if((stroke == 0) && converged) {
	      stroke = mu_;
	      mu_ *= .5;
	      if((used_nonlinearity % 2) == 0) {
		used_nonlinearity++;
	      }
	    }
	    else if(stroke != 0) {
	      mu_ = stroke;
	      stroke = 0;
	      if((mu_ == 1) && ((used_nonlinearity % 2) != 0)) {
		used_nonlinearity--;
	      }
	    }
	    else if(not_fine && (!taking_long) && (i > ((double) max_num_iterations_ / 2))) {
	      taking_long = true;
	      mu_ *= .5;
	      if((used_nonlinearity % 2) == 0) {
		used_nonlinearity++;
	      }
	    }
	  }

	  w_old2.CopyValues(w_old);
	  w_old.CopyValues(w);
	  
	  printf("used_nonlinearity = %d\n", used_nonlinearity);

	  switch(used_nonlinearity) {

	  case LOGCOSH: {
	    DeflationLogCoshUpdate_(n, X, &w);
	    break;
	  }

	  case LOGCOSH + 1: {
	    DeflationLogCoshFineTuningUpdate_(n, X, &w);
	    break;
	  }

	  case LOGCOSH + 2: {
	    Matrix X_sub;
	    index_t num_selected = RandomSubMatrix(n, percent_cut_, X, &X_sub);
	    DeflationLogCoshUpdate_(num_selected, X_sub, &w);
	    break;
	  }

	  case LOGCOSH + 3: {
	    Matrix X_sub;
	    index_t num_selected = RandomSubMatrix(n, percent_cut_, X, &X_sub);
	    DeflationLogCoshFineTuningUpdate_(num_selected, X_sub, &w);
	    break;
	  }

	  case GAUSS: {
	    DeflationGaussUpdate_(n, X, &w);
	    break;
	  }

	  case GAUSS + 1: {
	    DeflationGaussFineTuningUpdate_(n, X, &w);
	    break;
	  }

	  case GAUSS + 2: {
	    Matrix X_sub;
	    index_t num_selected = RandomSubMatrix(n, percent_cut_, X, &X_sub);
	    DeflationGaussUpdate_(num_selected, X_sub, &w);
	    break;
	  }

	  case GAUSS + 3: {
	    Matrix X_sub;
	    index_t num_selected = RandomSubMatrix(n, percent_cut_, X, &X_sub);
	    DeflationGaussFineTuningUpdate_(num_selected, X_sub, &w);
	    break;
	  }

	  case KURTOSIS: {
	    DeflationKurtosisUpdate_(n, X, &w);
	    break;
	  }

	  case KURTOSIS + 1: {
	    DeflationKurtosisFineTuningUpdate_(n, X, &w);
	    break;
	  }

	  case KURTOSIS + 2: {
	    Matrix X_sub;
	    index_t num_selected = RandomSubMatrix(n, percent_cut_, X, &X_sub);
	    DeflationKurtosisUpdate_(num_selected, X_sub, &w);
	    break;
	  }

	  case KURTOSIS + 3: {
	    Matrix X_sub;
	    index_t num_selected = RandomSubMatrix(n, percent_cut_, X, &X_sub);
	    DeflationKurtosisFineTuningUpdate_(num_selected, X_sub, &w);
	    break;
	  }

	  case SKEW: {
	    DeflationSkewUpdate_(n, X, &w);
	    break;
	  }

	  case SKEW + 1: {
	    DeflationSkewFineTuningUpdate_(n, X, &w);
	    break;
	  }

	  case SKEW + 2: {
	    Matrix X_sub;
	    index_t num_selected = RandomSubMatrix(n, percent_cut_, X, &X_sub);
	    DeflationSkewUpdate_(num_selected, X_sub, &w);
	    break;
	  }

	  case SKEW + 3: {
	    Matrix X_sub;
	    index_t num_selected = RandomSubMatrix(n, percent_cut_, X, &X_sub);
	    DeflationSkewFineTuningUpdate_(num_selected, X_sub, &w);
	    break;
	  }
	    
	  default: 
	    printf("ERROR: invalid contrast function: used_nonlinearity = %d\n",
		   used_nonlinearity);
	    exit(SUCCESS_FAIL);
	    
	  }
	
	  la::Scale(1/sqrt(la::Dot(w, w)), &w); // normalize
	  i++;
	}
	round++;
      }
    }
    
    // this code should be unreachable
    MulTransAOverwrite(&B, &whitening_matrix, W);
    return SUCCESS_FAIL; 
  }


  /**
   * Runs FastICA Algorithm on matrix X and Inits W to unmixing matrix and Y to
   * independent components matrix, such that \f$ X = W * Y \f$
   */
  int DoFastICA(datanode *module, Matrix *W, Matrix *Y) {

    const char *string_approach =
      fx_param_str(NULL, "approach", "symmetric");
    if(strcasecmp(string_approach, "deflation") == 0) {
      approach_ = DEFLATION;
    }
    else if(strcasecmp(string_approach, "symmetric") == 0) {
      approach_ = SYMMETRIC;
    }
    else {
      printf("ERROR: approach must be 'deflation' or 'symmetric'\n");
      W -> Init(0,0);
      Y -> Init(0,0);
      return SUCCESS_FAIL;
    }
    
    const char *string_nonlinearity =
      fx_param_str(NULL, "nonlinearity", "logcosh");
    if(strcasecmp(string_nonlinearity, "logcosh") == 0) {
      nonlinearity_ = LOGCOSH;
    }
    else if(strcasecmp(string_nonlinearity, "gauss") == 0) {
      nonlinearity_ = GAUSS;
    }
    else if(strcasecmp(string_nonlinearity, "kurtosis") == 0) {
      nonlinearity_ = KURTOSIS;
    }
    else if(strcasecmp(string_nonlinearity, "skew") == 0) {
      nonlinearity_ = SKEW;
    }
    else {
      printf("ERROR: nonlinearity not in {'logcosh', 'gauss', 'kurtosis', 'skew'}\n");
      W -> Init(0,0);
      Y -> Init(0,0);
      return SUCCESS_FAIL;
    }

    //const index_t first_eig_ = fx_param_int(NULL, "first_eig", 1);
    // for now, the last eig must be d, and num_of IC must be d, until I have time to incorporate PCA into this code
    //const index_t last_eig_ = fx_param_int(NULL, "last_eig", d);
    num_of_IC_ = d; //fx_param_int(NULL, "num_of_IC", d);
    fine_tune_ = fx_param_bool(NULL, "fine_tune", 0);
    a1_ = fx_param_double(NULL, "a1", 1);
    a2_ = fx_param_double(NULL, "a2", 1);
    mu_ = fx_param_double(NULL, "mu", 1);
    stabilization_ = fx_param_bool(NULL, "stabilization", true);
    epsilon_ = fx_param_double(NULL, "epsilon", 0.0001);
  
    int int_max_num_iterations = fx_param_int(NULL, "max_num_iterations", 1000);
    if(int_max_num_iterations < 0) {
      printf("ERROR: max_num_iterations = %d must be >= 0\n",
	     int_max_num_iterations);
      W -> Init(0,0);
      Y -> Init(0,0);
      return SUCCESS_FAIL;
    }
    max_num_iterations_ = (index_t) int_max_num_iterations;

    int int_max_fine_tune = fx_param_int(NULL, "max_fine_tune", 5);
    if(int_max_fine_tune < 0) {
      printf("ERROR: max_fine_tune = %d must be >= 0\n",
	     int_max_fine_tune);
      W -> Init(0,0);
      Y -> Init(0,0);
      return SUCCESS_FAIL;
    }
    max_fine_tune_ = (index_t) int_max_fine_tune;

    percent_cut_ = fx_param_double(NULL, "percent_cut", 1);
    if((percent_cut_ < 0) || (percent_cut_ > 1)) {
      printf("ERROR: percent_cut = %f must be an element in [0,1]\n",
	     percent_cut_);
      W -> Init(0,0);
      Y -> Init(0,0);
      return SUCCESS_FAIL;
    }

    Matrix X_centered, X_whitened, whitening_matrix, dewhitening_matrix, A;

    Center(X_, &X_centered);

    WhitenUsingEig(X_centered, &X_whitened, &whitening_matrix, &dewhitening_matrix);
  
    int ret_val =
      FixedPointICA(X_whitened, whitening_matrix, dewhitening_matrix, &A, W);

    if(ret_val == SUCCESS_PASS) {
      W -> PrintDebug("W");
      la::MulInit(*W, X_, Y);
    }
    else {
      Y -> Init(0,0);
    }

    return ret_val;
  }
};
