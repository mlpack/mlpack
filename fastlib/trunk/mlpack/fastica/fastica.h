/**
 * @file fastica.h
 *
 * FastICA Algorithm
 *
 * Implements the FastICA Algorithm for Independent Component Analysis using
 * fixed-point optimization with various independence-minded contrast
 * functions. For sample usage, see accompanying file fastica_main.cc
 *
 * @see fastica_main.cc
 *
 * @author Nishant Mehta
 */

#ifndef FASTICA_H
#define FASTICA_H

#include "fastlib/fastlib.h"
#include "lin_alg.h"

#define LOGCOSH 0
#define GAUSS 10
#define KURTOSIS 20
#define SKEW 30

#define SYMMETRIC 0
#define DEFLATION 1

const fx_entry_doc fastica_entries[] = {
  {"seed", FX_PARAM, FX_INT, NULL,
   "Seed for the random number generator.\n"},
  {"approach", FX_PARAM, FX_STR, NULL,
   "Independent component recovery approach: 'deflation' or 'symmetric'.\n"},
  {"nonlinearity", FX_PARAM, FX_STR, NULL,
   "Nonlinear function to use: 'logcosh', 'gauss', 'kurtosis', or 'skew'.\n"},
  {"num_of_IC", FX_PARAM, FX_INT, NULL,
   "  Number of independent components to find: integer between 1 and dimensionality of data.\n"},
  {"fine_tune", FX_PARAM, FX_BOOL, NULL,
   "Enable fine tuning.\n"},
  {"a1", FX_PARAM, FX_DOUBLE, NULL,
   "Numeric constant for logcosh nonlinearity.\n"},
  {"a2", FX_PARAM, FX_DOUBLE, NULL,
   "Numeric constant for gauss nonlinearity.\n"},
  {"mu", FX_PARAM, FX_DOUBLE, NULL,
   "Numeric constant for fine-tuning Newton-Raphson method.\n"},
  {"stabilization", FX_PARAM, FX_BOOL, NULL,
   "Use stabilization.\n"},
  {"epsilon", FX_PARAM, FX_DOUBLE, NULL,
   "Threshold for convergence.\n"},
  {"max_num_iterations", FX_PARAM, FX_INT, NULL,
   "Maximum number of iterations of fixed-point iterations.\n"},
  {"max_fine_tune", FX_PARAM, FX_INT, NULL,
   "Maximum number of fine-tuning iterations.\n"},
  {"percent_cut", FX_PARAM, FX_DOUBLE, NULL,
   "Number in range [0,1] indicating percent data to use in stabilization updates.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc fastica_doc = {
  fastica_entries, NULL,
  "Tuning parameters for FastICA.\n"
};

using namespace linalg__private;


/**
 * Class for running FastICA Algorithm
 *
 * This class takes in a D by N data matrix and runs FastICA, for number
 * of dimensions D and number of samples N
 */
class FastICA {
  
 private:

  /** Module used to pass parameters into the FastICA object */
  struct datanode* module_;

  /** data */
  Matrix X_;

  /** Optimization approach to use (deflation vs symmetric) */
  int approach_;

  /** Nonlinearity (contrast function) to use for evaluating independence */
  int nonlinearity_;

  //const index_t first_eig;
  //const index_t last_eig;

  /** number of independent components to find */
  index_t num_of_IC_;

  /** whether to enable fine tuning */
  bool fine_tune_;

  /** constant used for log cosh nonlinearity */
  double a1_;

  /** constant used for Gauss nonlinearity */
  double a2_;

  /** constant used for fine tuning */
  double mu_;

  /** whether to enable stabilization */
  bool stabilization_;

  /** threshold for convergence */
  double epsilon_;

  /** maximum number of iterations beore giving up */
  index_t max_num_iterations_;

  /** maximum number of times to fine tune */
  index_t max_fine_tune_;

  /** for stabilization, percent of data to include in random draw */
  double percent_cut_;



  /**
   * Symmetric Newton-Raphson using log cosh contrast function
   */
  void SymmetricLogCoshUpdate_(index_t n, Matrix X, Matrix* B) {
    Matrix hyp_tan, col_vector, sum, temp1, temp2;
    
    MapOverwrite(&TanhArg,
		 a1(),
		 MulTransAInit(&X, B, &hyp_tan));
    
    
    ColVector(d, a1(), &col_vector);
    
    Scale(1 / (double) n,
	  AddTo(MulInit(&X, &hyp_tan, &temp1),
		DotMultiplyOverwrite(MulInit(&col_vector,
					     MapOverwrite(&MinusArg,
							  n,
							  MatrixMapSum(&Square, 0, &hyp_tan, &sum)),
					     &temp2),
				     B)));
  }


  /**
   * Fine-tuned Symmetric Newton-Raphson using log cosh contrast
   * function
   */
  void SymmetricLogCoshFineTuningUpdate_(index_t n, Matrix X, Matrix* B) {
    Matrix Y, hyp_tan, Beta, Beta_Diag, D, sum, temp1, temp2, temp3;
	
    MulTransAInit(&X, B, &Y);
    MapInit(&TanhArg, a1(), &Y, &hyp_tan);
    DotMultiplySum(&Y, &hyp_tan, &Beta);
    VectorToDiag(MapOverwrite(&Inv,
			      0,
			      AddTo(&Beta,
				    Scale(a1(),
					  MapOverwrite(&MinusArg,
						       n,
						       MatrixMapSum(&Square, 0, &hyp_tan, &sum))))),
		 &D);
    
    AddExpert(mu(), 
	      MulInit(MulInit(B,
			      SubFrom(VectorToDiag(&Beta, &Beta_Diag),
				      MulTransAInit(&Y, &hyp_tan, &temp1)),
			      &temp2),
		      &D,
		      &temp3),
	      B);
  }


  /**
   * Symmetric Newton-Raphson using Gaussian contrast function
   */
  void SymmetricGaussUpdate_(index_t n, Matrix X, Matrix* B) {
    Matrix U, U_squared, ex, col_vector, sum, temp1, temp2;
    
    MulTransAInit(&X, B, &U);
    
    MapInit(&Square, 0, &U, &U_squared);
    MapInit(&ExpArg, -a2() / 2, &U_squared, &ex);
    DotMultiplyOverwrite(&ex, &U);
    //U is gauss
    AddTo(DotMultiplyOverwrite(&ex,
			       Scale(-a2(), &U_squared)),
	  &ex);
    //ex is dGauss
    
    ColVector(d, a2(), &col_vector);
    
    Scale(1 / (double) n,
	  SubOverwrite(MulInit(&X, &U, &temp1),
		       DotMultiplyOverwrite(B,
					    MulInit(&col_vector,
						    Sum(&ex, &sum),
						    &temp2)),
		       B));
  }


  /**
   * Fine-tuned Symmetric Newton-Raphson using Gaussian contrast function
   */
  void SymmetricGaussFineTuningUpdate_(index_t n, Matrix X, Matrix* B) {
    Matrix Y, Y_squared_a2, ex, gauss, D, Beta, temp1, temp2, temp3;
    Vector Beta_vector, sum_vector;
	
    MulTransAInit(&X, B, &Y);
    MapInit(&SquareArg, a2(), &Y, &Y_squared_a2);
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
    AddExpert(mu(),
	      MulInit(MulInit(B,
			      SubFrom(VectorToDiag(&Beta_vector, &Beta),
				      MulTransAInit(&Y, &gauss, &temp1)),
			      &temp2),
		      &D,
		      &temp3),
	      B);
  }


  /**
   * Symmetric Newton-Raphson using kurtosis contrast function
   */
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


  /**
   * Fine-tuned Symmetric Newton-Raphson using kurtosis contrast function
   */
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

    AddExpert(mu(), 
	      MulInit(MulInit(B,
			      SubFrom(VectorToDiag(&Beta, &Beta_Diag),
				      MulTransAInit(&Y, &G_pow_3, &temp1)),
			      &temp2),
		      &D,
		      &temp3),
	      B);
  }


  /**
   * Symmetric Newton-Raphson using skew contrast function
   */
  void SymmetricSkewUpdate_(index_t n, Matrix X, Matrix* B) {
    Matrix temp1;

    Scale(1 / (double) n,
	  MulOverwrite(&X,
		       MapOverwrite(&Square,
				    0,
				    MulTransAInit(&X, B, &temp1)),
		       B));

  }


  /**
   * Fine-tuned Symmetric Newton-Raphson using skew contrast function
   */
  void SymmetricSkewFineTuningUpdate_(index_t n, Matrix X, Matrix* B) {
    Matrix Y, G_skew, Beta, Beta_Diag, D_vector, D, temp1, temp2, temp3;
	
    MulTransAInit(&X, B, &Y);
    MapInit(&Square, 0, &Y, &G_skew);
    DotMultiplySum(&Y, &G_skew, &Beta);
    VectorToDiag(MapInit(&Inv, 0, &Beta, &D_vector),
		 &D);
	
    AddExpert(mu(), 
	      MulInit(MulInit(B,
			      SubFrom(VectorToDiag(&Beta, &Beta_Diag),
				      MulTransAInit(&Y, &G_skew, &temp1)),
			      &temp2),
		      &D,
		      &temp3),
	      B);
  }

  
  /**
   * Deflation Newton-Raphson using log cosh contrast function
   */
  void DeflationLogCoshUpdate_(index_t n, Matrix X, Vector* w) {
    Vector hyp_tan, temp1;
   
//    printf("-------- w --------\n");
//    w->PrintDebug();
//    printf("-------- X --------\n");
//    X.PrintDebug();
//    printf("-------- hyp_tan --------\n");

    MulInit(w, &X, &hyp_tan);
//    hyp_tan.PrintDebug();
    MapOverwrite(&TanhArg,
		 a1(),
		 &hyp_tan);

       
//    printf("-------- hyp_tan now --------\n");
//    hyp_tan.PrintDebug();
 
    Scale(1 / (double) n,
	  AddTo(MulInit(&X, &hyp_tan, &temp1),
		Scale(a1() * (VectorMapSum(&Square, 0, &hyp_tan) - n),
		      w)));

//    printf("-------- w now ----------\n");
//    w->PrintDebug();
  }


  /**
   * Fine-tuned Deflation Newton-Raphson using log cosh contrast function
   */
  void DeflationLogCoshFineTuningUpdate_(index_t n, Matrix X, Vector* w) {
    Vector hyp_tan, X_hyp_tan, Beta_w, temp1;
    
    MapOverwrite(&TanhArg,
		 a1(),
		 MulInit(w, &X, &hyp_tan));
    
    MulInit(&X, &hyp_tan, &X_hyp_tan);
    double Beta = la::Dot(X_hyp_tan, *w);
    printf("beta is %lf\n", Beta);
    
    double scale = 1 / (a1() * (VectorMapSum(&Square, 0, &hyp_tan) - n) + Beta);
    printf("scale is %lf\n", scale);

    AddExpert(mu(),
	      Scale(scale,
		    SubInit(&X_hyp_tan,
			    ScaleInit(Beta, w, &Beta_w),
			    &temp1)),
	      w);
  }


  /**
   * Deflation Newton-Raphson using Gaussian contrast function
   */
  void DeflationGaussUpdate_(index_t n, Matrix X, Vector* w) {
    Vector u, u_squared, ex, temp1;

    MulInit(w, &X, &u);
    MapInit(&Square, 0, &u, &u_squared);
    MapInit(&ExpArg, -.5 * a2(), &u_squared, &ex);
    DotMultiplyOverwrite(&ex, &u);
    //u is gauss
    AddTo(DotMultiplyOverwrite(&ex,
			       Scale(-a2(), &u_squared)),
	  &ex);
    //ex is dGauss
    Scale(1 / (double) n,
	  AddTo(MulInit(&X, &u, &temp1),
		Scale(-1 * Sum(&ex), w)));
  }
  
  
  /**
   * Fine-tuned Deflation Newton-Raphson using Gaussian contrast function
   */
  void DeflationGaussFineTuningUpdate_(index_t n, Matrix X, Vector* w) {
    Vector u, u_squared, ex, X_gauss, Beta_w, temp1;

    MulInit(w, &X, &u);
    MapInit(&Square, 0, &u, &u_squared);
    MapInit(&ExpArg, -.5 * a2(), &u_squared, &ex);
    DotMultiplyOverwrite(&ex, &u);
    //u is gauss
    AddTo(DotMultiplyOverwrite(&ex,
			       Scale(-a2(), &u_squared)),
	  &ex);
    //ex is dGauss

    MulInit(&X, &u, &X_gauss);
    double Beta = la::Dot(X_gauss, *w);

    AddExpert(mu(),
	      Scale(1 / (Beta - Sum(&ex)),
		    SubInit(&X_gauss,
			    ScaleInit(Beta, w, &Beta_w),
			    &temp1)),
	      w);
  }

  
  /**
   * Deflation Newton-Raphson using kurtosis contrast function
   */
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
  
  
  /**
   * Fine-tuned Deflation Newton-Raphson using kurtosis contrast function
   */
  void DeflationKurtosisFineTuningUpdate_(index_t n, Matrix X, Vector* w) {
    Vector EXG_pow_3, Beta_w, temp1;
	    
    Scale(1 / (double) n,
	  MulInit(&X,
		  MapOverwrite(&pow,
			       3,
			       MulInit(w, &X, &temp1)),
		  &EXG_pow_3));

    double Beta = la::Dot(*w, EXG_pow_3);
	    
    AddExpert(mu() / (Beta - 3),
	      SubFrom(ScaleInit(Beta, w, &Beta_w),
		      &EXG_pow_3),
	      w);
  }

  
  /**
   * Deflation Newton-Raphson using skew contrast function
   */
  void DeflationSkewUpdate_(index_t n, Matrix X, Vector* w) {
    Vector temp1;
	    
    Scale(1 / (double) n,
	  MulInit(&X,
		  MapOverwrite(&Square,
			       0,
			       MulInit(w, &X, &temp1)),
		  w));
  }

  
  /**
   * Fine-tuned Deflation Newton-Raphson using skew contrast function
   */
  void DeflationSkewFineTuningUpdate_(index_t n, Matrix X, Vector* w) {
    Vector EXG_skew, Beta_w, temp1;
	    
    Scale(1 / (double) n,
	  MulInit(&X,
		  MapOverwrite(&Square,
			       0,
			       MulInit(w, &X, &temp1)),
		  &EXG_skew));

    double Beta = la::Dot(*w, EXG_skew);
	    
    AddExpert(mu() / Beta,
	      SubFrom(ScaleInit(Beta, w, &Beta_w),
		      &EXG_skew),
	      w);
  }
  
    
  
 public:

  /** number of dimensions (components) in original data */
  index_t d;
  /** number of samples of original data */
  index_t n;

  int approach() {
    return approach_;
  }

  int nonlinearity() {
    return nonlinearity_;
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

  Matrix X() {
    return X_;
  }


  /** 
   * Default constructor does nothing special
   */
  FastICA() {
  }

  /**
   * Initializes the FastICA object by obtaining everything the algorithm needs
   */
  int Init(Matrix X_in, struct datanode* module_in) {

    module_ = module_in;

    X_.Copy(X_in); // for some reason Alias makes this crash, so copy for now
    d = X_.n_rows();
    n = X_.n_cols();

    long seed = fx_param_int(module_, "seed", clock() + time(0));
    srand48(seed);

    
    const char* string_approach =
      fx_param_str(module_, "approach", "deflation");
    if(strcasecmp(string_approach, "deflation") == 0) {
      VERBOSE_ONLY( printf("using Deflation approach ") );
      approach_ = DEFLATION;
    }
    else if(strcasecmp(string_approach, "symmetric") == 0) {
      VERBOSE_ONLY( printf("using Symmetric approach ") );
      approach_ = SYMMETRIC;
    }
    else {
      printf("ERROR: approach must be 'deflation' or 'symmetric'\n");
      return SUCCESS_FAIL;
    }
    
    const char* string_nonlinearity =
      fx_param_str(module_, "nonlinearity", "logcosh");
    if(strcasecmp(string_nonlinearity, "logcosh") == 0) {
      VERBOSE_ONLY( printf("with log cosh nonlinearity\n") );
      nonlinearity_ = LOGCOSH;
    }
    else if(strcasecmp(string_nonlinearity, "gauss") == 0) {
      VERBOSE_ONLY( printf("with Gaussian nonlinearity\n") );
      nonlinearity_ = GAUSS;
    }
    else if(strcasecmp(string_nonlinearity, "kurtosis") == 0) {
      VERBOSE_ONLY( printf("with kurtosis nonlinearity\n") );
      nonlinearity_ = KURTOSIS;
    }
    else if(strcasecmp(string_nonlinearity, "skew") == 0) {
      VERBOSE_ONLY( printf("with skew nonlinearity\n") );
      nonlinearity_ = SKEW;
    }
    else {
      printf("\nERROR: nonlinearity not in {logcosh, gauss, kurtosis, skew}\n");
      return SUCCESS_FAIL;
    }

    //const index_t first_eig_ = fx_param_int(module_, "first_eig", 1);
    // for now, the last eig must be d, and num_of IC must be d, until I have time to incorporate PCA into this code
    //const index_t last_eig_ = fx_param_int(module_, "last_eig", d);
    num_of_IC_ = fx_param_int(module_, "num_of_IC", d);
    if(num_of_IC_ < 1 || num_of_IC_ > d) {
      printf("ERROR: num_of_IC = %d must be >= 1 and <= dimensionality of data",
	     num_of_IC_);
      return SUCCESS_FAIL;
    }

    fine_tune_ = fx_param_bool(module_, "fine_tune", false);
    a1_ = fx_param_double(module_, "a1", 1);
    a2_ = fx_param_double(module_, "a2", 1);
    mu_ = fx_param_double(module_, "mu", 1);
    stabilization_ = fx_param_bool(module_, "stabilization", false);
    epsilon_ = fx_param_double(module_, "epsilon", 0.0001);
  
    int int_max_num_iterations =
      fx_param_int(module_, "max_num_iterations", 1000);
    if(int_max_num_iterations < 0) {
      printf("ERROR: max_num_iterations = %d must be >= 0\n",
	     int_max_num_iterations);
      return SUCCESS_FAIL;
    }
    max_num_iterations_ = (index_t) int_max_num_iterations;

    int int_max_fine_tune = fx_param_int(module_, "max_fine_tune", 5);
    if(int_max_fine_tune < 0) {
      printf("ERROR: max_fine_tune = %d must be >= 0\n",
	     int_max_fine_tune);
      return SUCCESS_FAIL;
    }
    max_fine_tune_ = (index_t) int_max_fine_tune;

    percent_cut_ = fx_param_double(module_, "percent_cut", 1);
    if((percent_cut() < 0) || (percent_cut() > 1)) {
      printf("ERROR: percent_cut = %f must be an element in [0,1]\n",
	     percent_cut());
      return SUCCESS_FAIL;
    }
    return SUCCESS_PASS;
  }


  // NOTE: these functions currently are public because some of them can
  //       serve as utilities that actually should be moved to lin_alg.h


  /**
   * Select indices < max according to probability equal to parameter
   * percentage, and return indices in a Vector
   * @pre selected_indices is an uninitialized Vector, percentage in [0 1]
   */
  index_t GetSamples(int max, double percentage, Vector* selected_indices) {   
    
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
   * Run FastICA using Symmetric approach
   */
  int SymmetricFixedPointICA(bool stabilization_enabled,
			     bool fine_tuning_enabled,
			     double mu_orig, double mu_k, index_t failure_limit,
			     int used_nonlinearity, int g_fine, double stroke,
			     bool not_fine, bool taking_long,
			     int initial_state_mode,
			     Matrix X, Matrix* B, Matrix* W,
			     Matrix* whitening_matrix) {
    
    if(initial_state_mode == 0) {
      //generate random B
      B -> Init(d, num_of_IC());
      for(index_t i = 0; i < num_of_IC(); i++) {
	Vector b;
	B -> MakeColumnVector(i, &b);
	RandVector(b);
      }
    }
      

    Matrix B_old, B_old2;

    B_old.Init(d, num_of_IC());
    B_old2.Init(d, num_of_IC());

    B_old.SetZero();
    B_old2.SetZero();


    for(index_t round = 1; round <= (max_num_iterations() + 1); round++) {
      if(round == (max_num_iterations() + 1)) {
	printf("No convergence after %d steps\n", max_num_iterations());
	
	
	// orthogonalize B via: newB = B * (B' * B) ^ -.5;
	Matrix temp;
	temp.Copy(*B);
	Orthogonalize(temp, B);

	MulTransAOverwrite(B, whitening_matrix, W);
	return SUCCESS_PASS;
      }
	
      {
	Matrix temp;
	temp.Copy(*B);
	Orthogonalize(temp, B);
      }

      Matrix B_delta_cov;
      MulTransAInit(B, &B_old, &B_delta_cov);
      double min_abs_cos = DBL_MAX;
      for(index_t i = 0; i < d; i++) {
	double current_cos = fabs(B_delta_cov.get(i, i));
	if(current_cos < min_abs_cos) {
	  min_abs_cos = current_cos;
	}
      }
      
      VERBOSE_ONLY( printf("delta = %f\n", 1 - min_abs_cos) );

      if(1 - min_abs_cos < epsilon()) {
	if(fine_tuning_enabled && not_fine) {
	  not_fine = false;
	  used_nonlinearity = g_fine;
	  mu_ = mu_k * mu_orig;
	  B_old.SetZero();
	  B_old2.SetZero();
	}
	else {
	  MulTransAOverwrite(B, whitening_matrix, W);
	  return SUCCESS_PASS;
	}
      }
      else if(stabilization_enabled) {

	Matrix B_delta_cov2;
	MulTransAInit(B, &B_old2, &B_delta_cov2);
	double min_abs_cos2 = DBL_MAX;
	for(index_t i = 0; i < d; i++) {
	  double current_cos2 = fabs(B_delta_cov2.get(i, i));
	  if(current_cos2 < min_abs_cos2) {
	    min_abs_cos2 = current_cos2;
	  }
	}

	VERBOSE_ONLY( printf("stabilization delta = %f\n", 1 - min_abs_cos2) );

	if((stroke == 0) && (1 - min_abs_cos2 < epsilon())) {
	  stroke = mu();
	  mu_ *= .5;
	  if((used_nonlinearity % 2) == 0) {
	    used_nonlinearity += 1;
	  }
	}
	else if(stroke > 0) {
	  mu_ = stroke;
	  stroke = 0;

	  if((mu() == 1) && ((used_nonlinearity % 2) != 0)) {
	    used_nonlinearity -= 1;
	  }
	}
	else if((!taking_long) &&
		(round > ((double) max_num_iterations() / 2))) {
	  taking_long = true;
	  mu_ *= .5;
	  if((used_nonlinearity % 2) == 0) {
	    used_nonlinearity += 1;
	  }
	}
      }

      B_old2.CopyValues(B_old);
      B_old.CopyValues(*B);

      // show progress here, (the lack of code means no progress shown for now)



      // use Newton-Raphson method to update B
      switch(used_nonlinearity) {
	  
      case LOGCOSH: {
	SymmetricLogCoshUpdate_(n, X, B);
	break;
      }
	
      case LOGCOSH + 1: {
	SymmetricLogCoshFineTuningUpdate_(n, X, B);
	break;	
      }
	
      case LOGCOSH + 2: {
	Matrix X_sub;
	index_t num_selected = RandomSubMatrix(n, percent_cut(), X, &X_sub);
	SymmetricLogCoshUpdate_(num_selected, X_sub, B);
	break;
      }
	
      case LOGCOSH + 3: {
	Matrix X_sub;
	index_t num_selected = RandomSubMatrix(n, percent_cut(), X, &X_sub);
	SymmetricLogCoshFineTuningUpdate_(num_selected, X_sub, B);
	break;
      }

      case GAUSS: {
	SymmetricGaussUpdate_(n, X, B);
	break;
      }
	
      case GAUSS + 1: {
	SymmetricGaussFineTuningUpdate_(n, X, B);
	break;
      }

      case GAUSS + 2: {
	Matrix X_sub;
	index_t num_selected = RandomSubMatrix(n, percent_cut(), X, &X_sub);
	SymmetricGaussUpdate_(num_selected, X_sub, B);
	break;
      }

      case GAUSS + 3: {
	Matrix X_sub;
	index_t num_selected = RandomSubMatrix(n, percent_cut(), X, &X_sub);
	SymmetricGaussFineTuningUpdate_(num_selected, X_sub, B);
	break;
      }

      case KURTOSIS: {
	SymmetricKurtosisUpdate_(n, X, B);
	break;
      }
	
      case KURTOSIS + 1: {
	SymmetricKurtosisFineTuningUpdate_(n, X, B);
	break;
      }

      case KURTOSIS + 2: {
	Matrix X_sub;
	index_t num_selected = RandomSubMatrix(n, percent_cut(), X, &X_sub);
	SymmetricKurtosisUpdate_(num_selected, X_sub, B);
	break;
      }

      case KURTOSIS + 3: {
	Matrix X_sub;
	index_t num_selected = RandomSubMatrix(n, percent_cut(), X, &X_sub);
	SymmetricKurtosisFineTuningUpdate_(num_selected, X_sub, B);
	break;
      }

      case SKEW: {
	SymmetricSkewUpdate_(n, X, B);
	break;
      }
	
      case SKEW + 1: {
	SymmetricSkewFineTuningUpdate_(n, X, B);
	break;
      }

      case SKEW + 2: {
	Matrix X_sub;
	index_t num_selected = RandomSubMatrix(n, percent_cut(), X, &X_sub);
	SymmetricSkewUpdate_(num_selected, X_sub, B);
	break;
      }
	  
      case SKEW + 3: {
	Matrix X_sub;
	index_t num_selected = RandomSubMatrix(n, percent_cut(), X, &X_sub);
	SymmetricSkewFineTuningUpdate_(num_selected, X_sub, B);
	break;
      }
	  
      default:
	printf("ERROR: invalid contrast function: used_nonlinearity = %d\n",
	       used_nonlinearity);
	exit(SUCCESS_FAIL);
      }
    }

    // this code should be unreachable
    return SUCCESS_FAIL; 
  }
  
  
  /**
   * Run FastICA using Deflation approach
   */
  int DeflationFixedPointICA(bool stabilization_enabled,
 			     bool fine_tuning_enabled,
			     double mu_orig, double mu_k, index_t failure_limit,
			     int used_nonlinearity, int g_orig, int g_fine,
			     double stroke, bool not_fine, bool taking_long,
			     int initial_state_mode,
			     Matrix X, Matrix* B, Matrix* W,
			     Matrix* whitening_matrix) {

    B -> Init(d, d);
    B -> SetZero();

    index_t round = 0;

    index_t num_failures = 0;

    while(round < num_of_IC()) {
      VERBOSE_ONLY( printf("Estimating IC %d\n", round + 1) );
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
	B -> MakeColumnVector(i, &b_i);
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
      while(i <= max_num_iterations() + gabba) {

	for(index_t j = 0; j < round; j++) {
	  Vector b_j;
	  B -> MakeColumnVector(j, &b_j);
	  la::AddExpert(-la::Dot(b_j, w), b_j, &w);
	}
	la::Scale(1/sqrt(la::Dot(w, w)), &w); // normalize
        printf("--------- beginning of iteration %d, w ------------\n", i);
        w.PrintDebug();

	if(not_fine) {
	  if(i == (max_num_iterations() + 1)) {
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

	double delta1 = la::Dot(w_diff, w_diff);
	double delta2 = DBL_MAX;
	  
	if(delta1 < epsilon()) {
	  converged = true;
	}
	else {
	  la::AddOverwrite(w_old, w, &w_diff);

	  delta2 = la::Dot(w_diff, w_diff);
	    
	  if(delta2 < epsilon()) {
	    converged = true;
	  }
	}

	printf("delta = %f\n", min(delta1, delta2));


	if(converged) {
	  if(fine_tuning_enabled & not_fine) {
	    not_fine = false;
	    gabba = max_fine_tune();
	    w_old.SetZero();
	    w_old2.SetZero();
	    used_nonlinearity = g_fine;
	    mu_ = mu_k * mu_orig;

	    end_fine_tuning = max_fine_tune() + i;
	  }
	  else {
	    num_failures = 0;
	    Vector B_col_round, W_col_round;

	    B -> MakeColumnVector(round, &B_col_round);
	    W -> MakeColumnVector(round, &W_col_round);

	    B_col_round.CopyValues(w);
	    la::MulOverwrite(w, *whitening_matrix, &W_col_round);

	    break; // this line is intended to take us to the next IC
	  }
	}
	else if(stabilization_enabled) {
	  converged = false;
	  la::SubInit(w_old2, w, &w_diff);
	    
	  if(la::Dot(w_diff, w_diff) < epsilon()) {
	    converged = true;
	  }
	  else {
	    la::AddOverwrite(w_old2, w, &w_diff);
	      
	    if(la::Dot(w_diff, w_diff) < epsilon()) {
	      converged = true;
	    }
	  }
	    
	  if((stroke == 0) && converged) {
	    stroke = mu();
	    mu_ *= .5;
	    if((used_nonlinearity % 2) == 0) {
	      used_nonlinearity++;
	    }
	  }
	  else if(stroke != 0) {
	    mu_ = stroke;
	    stroke = 0;
	    if((mu() == 1) && ((used_nonlinearity % 2) != 0)) {
	      used_nonlinearity--;
	    }
	  }
	  else if(not_fine && (!taking_long) &&
		  (i > ((double) max_num_iterations() / 2))) {
	    taking_long = true;
	    mu_ *= .5;
	    if((used_nonlinearity % 2) == 0) {
	      used_nonlinearity++;
	    }
	  }
	}

	w_old2.CopyValues(w_old);
	w_old.CopyValues(w);
	printf("Using nonlinearity %d\n", used_nonlinearity);
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
	  index_t num_selected = RandomSubMatrix(n, percent_cut(), X, &X_sub);
	  DeflationLogCoshUpdate_(num_selected, X_sub, &w);
	  break;
	}

	case LOGCOSH + 3: {
	  Matrix X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut(), X, &X_sub);
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
	  index_t num_selected = RandomSubMatrix(n, percent_cut(), X, &X_sub);
	  DeflationGaussUpdate_(num_selected, X_sub, &w);
	  break;
	}

	case GAUSS + 3: {
	  Matrix X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut(), X, &X_sub);
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
	  index_t num_selected = RandomSubMatrix(n, percent_cut(), X, &X_sub);
	  DeflationKurtosisUpdate_(num_selected, X_sub, &w);
	  break;
	}

	case KURTOSIS + 3: {
	  Matrix X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut(), X, &X_sub);
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
	  index_t num_selected = RandomSubMatrix(n, percent_cut(), X, &X_sub);
	  DeflationSkewUpdate_(num_selected, X_sub, &w);
	  break;
	}

	case SKEW + 3: {
	  Matrix X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut(), X, &X_sub);
	  DeflationSkewFineTuningUpdate_(num_selected, X_sub, &w);
	  break;
	}
	    
	default: 
	  printf("ERROR: invalid contrast function: used_nonlinearity = %d\n",
		 used_nonlinearity);
	  exit(SUCCESS_FAIL);
	}
	
	la::Scale(1/sqrt(la::Dot(w, w)), &w); // normalize
//        printf("-------- scaled w ----------\n");
//        w.PrintDebug();
	i++;
      }
      round++;
    }

    return SUCCESS_PASS; 
  }


  /**
   * Verify the validity of some settings, set some parameters needed by
   * the algorithm, and run the fixed-point FastICA algorithm using either
   * the specified approach
   * @pre{ X is a d by n data matrix, for d dimensions and n samples}
   */
  int FixedPointICA(Matrix X, Matrix whitening_matrix, Matrix* W) {
    // ensure default values are passed into this function if the user doesn't care about certain parameters

    int g = nonlinearity();
    
    if(d < num_of_IC()) {
      printf("ERROR: must have num_of_IC <= Dimension!\n");
      W -> Init(0,0);
      return SUCCESS_FAIL;
    }

    W -> Init(d, num_of_IC());

    if((percent_cut() > 1) || (percent_cut() < 0)) {
      percent_cut_ = 1;
      printf("Setting percent_cut to 1\n");
    }
    else if(percent_cut() < 1) {
      if((percent_cut() * n) < 1000) {
	percent_cut_ = min(1000 / (double) n, (double) 1);
	printf("Warning: Setting percent_cut to %0.3f (%d samples).\n",
	       percent_cut(),
	       (int) floor(percent_cut() * n));
      }
    }
    
    int g_orig = g;

    if(percent_cut() != 1) {
      g_orig += 2;
    }
    
    if(mu() != 1) {
      g_orig += 1;
    }

    bool fine_tuning_enabled = true;
    int g_fine;

    if(fine_tune()) {
      g_fine = g + 1;
    }
    else {
      if(mu() != 1) {
	g_fine = g_orig;
      }
      else {
	g_fine = g_orig + 1;
      }

      fine_tuning_enabled = false;
    }

    bool stabilization_enabled;
    if(stabilization()) {
      stabilization_enabled = true;
    }
    else {
      if(mu() != 1) {
	stabilization_enabled = true;
      }
      else {
	stabilization_enabled = false;
      }
    }

    double mu_orig = mu();
    double mu_k = 0.01;
    index_t failure_limit = 5;
    int used_nonlinearity = g_orig;
    double stroke = 0;
    bool not_fine = true;
    bool taking_long = false;

    // currently we don't allow for guesses for the initial unmixing matrix B
    int initial_state_mode = 0;

    Matrix B;

    int ret_val = SUCCESS_FAIL;
    
    if(approach() == SYMMETRIC) {
      ret_val = 
	SymmetricFixedPointICA(stabilization_enabled, fine_tuning_enabled,
			       mu_orig, mu_k, failure_limit,
			       used_nonlinearity, g_fine, stroke,
			       not_fine, taking_long, initial_state_mode,
			       X, &B, W,
			       &whitening_matrix);
    }
    else if(approach() == DEFLATION) {
      ret_val = 
	DeflationFixedPointICA(stabilization_enabled, fine_tuning_enabled,
			       mu_orig, mu_k, failure_limit,
			       used_nonlinearity, g_orig, g_fine,
			       stroke, not_fine, taking_long, initial_state_mode,
			       X, &B, W,
			       &whitening_matrix);
    }

    return ret_val;
  }


  /**
   * Runs FastICA Algorithm on matrix X and Inits W to unmixing matrix and Y to
   * independent components matrix, such that \f$ X = W * Y \f$
   */
  int DoFastICA(Matrix* W, Matrix* Y) {

    Matrix X_centered, X_whitened, whitening_matrix;

//    puts("---------- X ------------");
//    X_.PrintDebug();

    Center(X(), &X_centered);

//    puts("---------- X_centered ----------");
//    X_centered.PrintDebug();

    WhitenUsingEig(X_centered, &X_whitened, &whitening_matrix);

//    puts("---------- X_whitened ----------");
//    X_whitened.PrintDebug();
//    puts("---------- whitening_matrix ----------");
//    whitening_matrix.PrintDebug();
  
    int ret_val =
      FixedPointICA(X_whitened, whitening_matrix, W);

    if(ret_val == SUCCESS_PASS) {
      la::MulTransAInit(*W, X(), Y);
    }
    else {
      Y -> Init(0,0);
    }

    return ret_val;
  }
}; /* class FastICA */

#endif /* FASTICA_H */
