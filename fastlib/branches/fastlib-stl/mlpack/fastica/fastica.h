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

#include <fastlib/fastlib.h>
#include "lin_alg.h"

#include <armadillo>

#define LOGCOSH 0
#define GAUSS 10
#define KURTOSIS 20
#define SKEW 30

#define SYMMETRIC 0
#define DEFLATION 1

const fx_entry_doc fastica_entries[] = {
  {"seed", FX_PARAM, FX_INT, NULL,
   "  Seed for the random number generator.\n"},
  {"approach", FX_PARAM, FX_STR, NULL,
   "  Independent component recovery approach: 'deflation' or 'symmetric'.\n"},
  {"nonlinearity", FX_PARAM, FX_STR, NULL,
   "  Nonlinear function to use: 'logcosh', 'gauss', 'kurtosis', or 'skew'.\n"},
  {"num_of_IC", FX_PARAM, FX_INT, NULL,
   "  Number of independent components to find: integer between 1 and dimensionality of data.\n"},
  {"fine_tune", FX_PARAM, FX_BOOL, NULL,
   "  Enable fine tuning.\n"},
  {"a1", FX_PARAM, FX_DOUBLE, NULL,
   "  Numeric constant for logcosh nonlinearity.\n"},
  {"a2", FX_PARAM, FX_DOUBLE, NULL,
   "  Numeric constant for gauss nonlinearity.\n"},
  {"mu", FX_PARAM, FX_DOUBLE, NULL,
   "  Numeric constant for fine-tuning Newton-Raphson method.\n"},
  {"stabilization", FX_PARAM, FX_BOOL, NULL,
   "  Use stabilization.\n"},
  {"epsilon", FX_PARAM, FX_DOUBLE, NULL,
   "  Threshold for convergence.\n"},
  {"max_num_iterations", FX_PARAM, FX_INT, NULL,
   "  Maximum number of iterations of fixed-point iterations.\n"},
  {"max_fine_tune", FX_PARAM, FX_INT, NULL,
   "  Maximum number of fine-tuning iterations.\n"},
  {"percent_cut", FX_PARAM, FX_DOUBLE, NULL,
   "  Number in [0,1] indicating percent data to use in stabilization updates.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc fastica_doc = {
  fastica_entries, NULL,
  "Performs fastica.\n"
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
  arma::mat X;

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
  void SymmetricLogCoshUpdate_(const index_t n, const arma::mat& X, arma::mat& B) {
    arma::mat hyp_tan, col_vector, msum;
    
    hyp_tan = trans(X) * B;
    // elementwise tanh()
    for(int i = 0; i < hyp_tan.n_elem; i++)
      hyp_tan[i] = tanh(a1_ * hyp_tan[i]);
   
    col_vector.set_size(d);
    col_vector.fill(a1_); 
   
    // take the squared L2 norm of the hyp_tan matrix rows (sum each squared
    // element) and subtract n
    msum = sum(pow(hyp_tan, 2), 1) - n;
    B %= col_vector * msum; // % is the schur product (elementwise multiply)
    B += (X * hyp_tan);
    B /= (1 / (double) n); // scale
  }


  /**
   * Fine-tuned Symmetric Newton-Raphson using log cosh contrast
   * function
   */
  void SymmetricLogCoshFineTuningUpdate_(index_t n, const arma::mat& X, arma::mat& B) {
    arma::mat Y, hyp_tan, Beta, Beta_Diag, D, msum;
	
    Y = trans(X) * B;
    hyp_tan.set_size(Y.n_rows, Y.n_cols);
    // elementwise tanh()
    for(int i = 0; i < hyp_tan.n_elem; i++)
      hyp_tan[i] = tanh(a1_ * Y[i]);
    Beta = sum(Y % hyp_tan, 1); // sum columns of elementwise multiplication

    // take squared L2 norm of hyp_tan matrix rows and subtract n
    msum = sum(pow(hyp_tan, 2), 1) - n;
    msum *= a1_; // scale
    msum += Beta; // elementwise addition
    
    D.zeros(msum.n_elem, msum.n_elem);
    D.diag() = pow(msum, -1);

    Beta_Diag.zeros(Beta.n_elem, Beta.n_elem);
    Beta_Diag.diag() = Beta;

    B += mu_ * ((B * ((trans(Y) * hyp_tan) - Beta_Diag)) * D);
  }


  /**
   * Symmetric Newton-Raphson using Gaussian contrast function
   */
  void SymmetricGaussUpdate_(index_t n, const arma::mat& X, arma::mat& B) {
    arma::mat U, U_squared, ex, col_vector;
    
    U = trans(X) * B;
    U_squared = -a2_ * pow(U, 2); // scale components
    ex = exp(U_squared / 2);
    U %= ex; // U is gauss
   
    ex += (U_squared % ex); // ex is dGauss

    col_vector.set_size(d);
    col_vector.fill(a2_);
    
    B = (X * U) - (B % col_vector * sum(ex, 1));
    B *= (1 / (double) n);
  }


  /**
   * Fine-tuned Symmetric Newton-Raphson using Gaussian contrast function
   */
  void SymmetricGaussFineTuningUpdate_(index_t n, const arma::mat X, arma::mat& B) {
    arma::mat Y, Y_squared_a2, ex, gauss, D, Beta;
    arma::vec Beta_vector, sum_vector;
    
    Y = trans(X) * B;
    Y_squared_a2 = pow(Y, 2) * a2_;
    ex = exp(Y_squared_a2 / 2);
    gauss = Y % ex;

    Beta_vector.set_size(d);
    Beta_vector = sum(Y % gauss, 1);

    sum_vector.set_size(d);
    sum_vector = sum((Y_squared_a2 - 1) % ex, 1);


    //D = diag(1 ./ (Beta + sum((Y_squared_a2 - 1) .* ex)))
    D.zeros(d, d);
    D.diag() = 1 / (Beta_vector + sum_vector);

    Beta.zeros(d, d);
    Beta.diag() = Beta_vector;

    //B = B + myy * B * (Y' * gauss - diag(Beta)) * D;
    B += mu_ * ((B * (trans(Y) * gauss - Beta)) * D);
  }


  /**
   * Symmetric Newton-Raphson using kurtosis contrast function
   */
  void SymmetricKurtosisUpdate_(index_t n, const arma::mat X, arma::mat& B) {
    B *= -3;    
    B += (X * pow(trans(X) * B, 3)) / (double) n;
  }


  /**
   * Fine-tuned Symmetric Newton-Raphson using kurtosis contrast function
   */
  void SymmetricKurtosisFineTuningUpdate_(index_t n, const arma::mat& X, arma::mat& B) {
    arma::mat Y, G_pow_3, Beta_Diag, D, temp1, temp2, temp3;
    arma::vec Beta, D_vector;

    Y = trans(X) * B;
    G_pow_3 = pow(Y, 3);
    Beta = Y.diag() % G_pow_3.diag();

    D_vector = 1 / (Beta - (3 * n));
    D.zeros(D_vector.n_elem, D_vector.n_elem);
    D.diag() = D_vector;

    Beta_Diag.zeros(Beta.n_elem, Beta.n_elem);
    Beta_Diag.diag() = Beta;

    B += mu_ * ((B * ((trans(Y) * G_pow_3) - Beta_Diag)) * D);
  }


  /**
   * Symmetric Newton-Raphson using skew contrast function
   */
  void SymmetricSkewUpdate_(index_t n, const arma::mat& X, arma::mat& B) {
    B = X * pow((trans(X) * B), 2);
    B /= (double) n;
  }


  /**
   * Fine-tuned Symmetric Newton-Raphson using skew contrast function
   */
  void SymmetricSkewFineTuningUpdate_(index_t n, const arma::mat& X, arma::mat& B) {
    arma::mat Y, G_skew, Beta_Diag, D_vector, D, temp1, temp2, temp3;
    arma::vec Beta;
    
    Y = trans(X) * B;
    G_skew = pow(Y, 2);
    Beta = sum(Y % G_skew, 1);
    D.zeros(Beta.n_elem, Beta.n_elem);
    D.diag() = (1 / Beta);
    Beta_Diag.zeros(Beta.n_elem, Beta.n_elem);
    Beta_Diag.diag() = Beta;

    B = mu_ * ((B * ((trans(Y) * G_skew) - Beta)) * D);
  }

  
  /**
   * Deflation Newton-Raphson using log cosh contrast function
   */
  void DeflationLogCoshUpdate_(index_t n, const arma::mat& X, arma::vec& w) {
    arma::vec hyp_tan, temp1;
    
    hyp_tan = trans(X) * w;
    for(int i = 0; i < hyp_tan.n_elem; i++)
      hyp_tan[i] = tanh(a1_ * hyp_tan[i]);

    w *= a1_ * (accu(pow(hyp_tan, 2)) - n);
    w += (X * hyp_tan);
    w /= (double) n;
  }


  /**
   * Fine-tuned Deflation Newton-Raphson using log cosh contrast function
   */
  void DeflationLogCoshFineTuningUpdate_(index_t n, const arma::mat& X, arma::vec& w) {
    arma::vec hyp_tan, X_hyp_tan;
    
    hyp_tan = trans(X) * w;
    for(int i = 0; i < hyp_tan.n_elem; i++)
      hyp_tan[i] = tanh(a1_ * hyp_tan[i]);
  
    X_hyp_tan = X * hyp_tan;
    double beta = dot(X_hyp_tan, w);  
   
    double scale = (1 / (a1_ * (accu(pow(hyp_tan, 2)) - n) + beta));
    w += mu_ * scale * (X_hyp_tan - (beta * w));
  }


  /**
   * Deflation Newton-Raphson using Gaussian contrast function
   */
  void DeflationGaussUpdate_(index_t n, const arma::mat& X, arma::vec& w) {
    arma::vec u, u_sq, u_sq_a, ex;

    u = trans(X) * w;
    u_sq = pow(u, 2);
    u_sq_a = -a2_ * u_sq;
    // u is gauss
    ex = exp(u_sq_a / 2.0);
    u = (ex % u);
    ex += (ex % u_sq_a);
    // ex is dGauss

    w *= -accu(ex);
    w += (X * u);
    w /= (double) n;
  }
  
  
  /**
   * Fine-tuned Deflation Newton-Raphson using Gaussian contrast function
   */
  void DeflationGaussFineTuningUpdate_(index_t n, const arma::mat& X, arma::vec& w) {
    arma::vec u, u_sq, u_sq_a, ex, x_gauss;

    u = trans(X) * w;
    u_sq = pow(u, 2);
    u_sq_a = -a2_ * u_sq;
    ex = exp(u_sq_a / 2.0);
    u = (ex % u);
    // u is gauss
    ex += (u_sq_a % ex);
    // ex is dGauss

    x_gauss = X * u;

    double beta = dot(x_gauss, w);

    double scale = 1 / (beta - accu(ex));
    w += mu_ * scale * (x_gauss - (beta * w));
  }

  
  /**
   * Deflation Newton-Raphson using kurtosis contrast function
   */
  void DeflationKurtosisUpdate_(index_t n, const arma::mat& X, arma::vec& w) {
    arma::vec temp1, temp2;

    w *= -3;
    w += (X * pow(trans(X) * w, 3)) / (double) n;
  }
  
  
  /**
   * Fine-tuned Deflation Newton-Raphson using kurtosis contrast function
   */
  void DeflationKurtosisFineTuningUpdate_(index_t n, const arma::mat& X, arma::vec& w) {
    arma::vec EXG_pow_3, Beta_w, temp1;
	    
    EXG_pow_3 = (X * pow(trans(X) * w, 3)) / (double) n;
    double beta = dot(w, EXG_pow_3);

    w += (mu_ / (beta - 3)) * ((beta * w) - EXG_pow_3);
  }

  
  /**
   * Deflation Newton-Raphson using skew contrast function
   */
  void DeflationSkewUpdate_(index_t n, const arma::mat& X, arma::vec& w) {
    arma::vec temp1;

    w = X * pow(trans(X) * w, 2);
    w /= (double) n;
  }

  
  /**
   * Fine-tuned Deflation Newton-Raphson using skew contrast function
   */
  void DeflationSkewFineTuningUpdate_(index_t n, const arma::mat& X, arma::vec& w) {
    arma::vec EXG_skew;
    
    EXG_skew = X * pow(trans(X) * w, 2);
    EXG_skew /= (double) n;

    double beta = dot(w, EXG_skew);

    w += (mu_ / beta) * ((beta * w) - EXG_skew);
  }
  
    
  
 public:

  /** number of dimensions (components) in original data */
  index_t d;
  /** number of samples of original data */
  index_t n;

  /** 
   * Default constructor does nothing special
   */
  FastICA() { }

  /**
   * Initializes the FastICA object by obtaining everything the algorithm needs
   */
  int Init(arma::mat& X_in, struct datanode* module_in) {

    module_ = module_in;

    X = X_in; 
    d = X.n_rows;
    n = X.n_cols;

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
    if((percent_cut_ < 0) || (percent_cut_ > 1)) {
      printf("ERROR: percent_cut = %f must be an element in [0,1]\n",
	     percent_cut_);
      return SUCCESS_FAIL;
    }
    return SUCCESS_PASS;
  }


  // NOTE: these functions currently are public because some of them can
  //       serve as utilities that actually should be moved to lin_alg.h


  /**
   * Return Select indices < max according to probability equal to parameter
   * percentage, and return indices in a Vector
   */
  index_t RandomSubMatrix(index_t n, double percent_cut, const arma::mat& X, arma::mat& X_sub) {
    std::vector<int> colnums;
    for(int i = 0; i < X.n_cols; i++) {
      if(drand48() <= percent_cut) // add column
        colnums.push_back(i);
    }

    // now that we have all the column numbers we want, assemble the random
    // submatrix
    X_sub.set_size(X.n_rows, colnums.size());
    for(int i = 0; i < colnums.size(); i++)
      X_sub.col(i) = X.col(colnums[i]);

    return colnums.size();
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
			     const arma::mat& X, arma::mat& B, arma::mat& W,
			     arma::mat& whitening_matrix) {
    
    if(initial_state_mode == 0) {
      //generate random B
      B.set_size(d, num_of_IC_);
      for(index_t i = 0; i < num_of_IC_; i++) {
        arma::vec temp_fail = B.col(i);
	RandVector(temp_fail);
        B.col(i) = temp_fail; // slow slow slow: TODO
      }
    }
      

    arma::mat B_old, B_old2;

    B_old.zeros(d, num_of_IC_);
    B_old2.zeros(d, num_of_IC_);

    // Main algorithm: iterate until convergence (or maximum iterations)
    for(index_t round = 1; round <= max_num_iterations_; round++) {
      Orthogonalize(B);

      arma::mat B_delta_cov;
      B_delta_cov = trans(B) * B_old;
      double min_abs_cos = DBL_MAX;
      for(index_t i = 0; i < d; i++) {
	double current_cos = fabs(B_delta_cov(i, i));
	if(current_cos < min_abs_cos)
	  min_abs_cos = current_cos;
      }
      
      VERBOSE_ONLY( printf("delta = %f\n", 1 - min_abs_cos) );

      if(1 - min_abs_cos < epsilon_) {
	if(fine_tuning_enabled && not_fine) {
	  not_fine = false;
	  used_nonlinearity = g_fine;
	  mu_ = mu_k * mu_orig;
	  B_old.zeros(d, num_of_IC_);
	  B_old2.zeros(d, num_of_IC_);
	}
	else {
          W = trans(B) * whitening_matrix;
	  return SUCCESS_PASS;
	}
      }
      else if(stabilization_enabled) {

	arma::mat B_delta_cov2;
        B_delta_cov2 = trans(B) * B_old2;
	double min_abs_cos2 = DBL_MAX;
	for(index_t i = 0; i < d; i++) {
	  double current_cos2 = fabs(B_delta_cov2(i, i));
	  if(current_cos2 < min_abs_cos2) {
	    min_abs_cos2 = current_cos2;
	  }
	}

	VERBOSE_ONLY( printf("stabilization delta = %f\n", 1 - min_abs_cos2) );

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

      B_old2 = B_old;
      B_old = B;

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
	arma::mat X_sub;
	index_t num_selected = RandomSubMatrix(n, percent_cut_, X, X_sub);
	SymmetricLogCoshUpdate_(num_selected, X_sub, B);
	break;
      }
	
      case LOGCOSH + 3: {
        arma::mat X_sub;
	index_t num_selected = RandomSubMatrix(n, percent_cut_, X, X_sub);
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
	arma::mat X_sub;
	index_t num_selected = RandomSubMatrix(n, percent_cut_, X, X_sub);
	SymmetricGaussUpdate_(num_selected, X_sub, B);
	break;
      }

      case GAUSS + 3: {
	arma::mat X_sub;
	index_t num_selected = RandomSubMatrix(n, percent_cut_, X, X_sub);
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
	arma::mat X_sub;
	index_t num_selected = RandomSubMatrix(n, percent_cut_, X, X_sub);
	SymmetricKurtosisUpdate_(num_selected, X_sub, B);
	break;
      }

      case KURTOSIS + 3: {
	arma::mat X_sub;
	index_t num_selected = RandomSubMatrix(n, percent_cut_, X, X_sub);
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
	arma::mat X_sub;
	index_t num_selected = RandomSubMatrix(n, percent_cut_, X, X_sub);
	SymmetricSkewUpdate_(num_selected, X_sub, B);
	break;
      }
	  
      case SKEW + 3: {
	arma::mat X_sub;
	index_t num_selected = RandomSubMatrix(n, percent_cut_, X, X_sub);
	SymmetricSkewFineTuningUpdate_(num_selected, X_sub, B);
	break;
      }
	  
      default:
	printf("ERROR: invalid contrast function: used_nonlinearity = %d\n",
	       used_nonlinearity);
	exit(SUCCESS_FAIL);
      }
    }

    printf("No convergence after %d steps\n", max_num_iterations_);
	
    // orthogonalize B via: newB = B * (B' * B) ^ -.5;
    Orthogonalize(B);
    W = trans(whitening_matrix) * B;

    return SUCCESS_PASS;
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
			     const arma::mat& X, arma::mat& B, arma::mat& W,
			     arma::mat& whitening_matrix) {

    B.zeros(d, d);

    index_t round = 0;
    index_t num_failures = 0;

    while(round < num_of_IC_) {
      VERBOSE_ONLY( printf("Estimating IC %d\n", round + 1) );
      mu_ = mu_orig;
      used_nonlinearity = g_orig;
      stroke = 0;
      not_fine = true;
      taking_long = false;
      int end_fine_tuning = 0;
	
      arma::vec w(d);
      if(initial_state_mode == 0)
	RandVector(w);

      for(index_t i = 0; i < round; i++)
        w -= dot(B.col(i), w) * B.col(i);
      w /= sqrt(dot(w, w)); // normalize

      arma::vec w_old, w_old2;
      w_old.zeros(d);
      w_old2.zeros(d);

      index_t i = 1;
      index_t gabba = 1;
      while(i <= max_num_iterations_ + gabba) {

	for(index_t j = 0; j < round; j++)
          w -= dot(B.col(j), w) * B.col(j);
	w /= sqrt(dot(w, w)); // normalize

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
	    w_old = w;
	  }
	}

	// check for convergence
	bool converged = false;
	arma::vec w_diff = w_old - w;

	double delta1 = dot(w_diff, w_diff);
	double delta2 = DBL_MAX;
	  
	if(delta1 < epsilon_) {
	  converged = true;
	} else {
          w_diff = w_old + w;

	  delta2 = dot(w_diff, w_diff);
	    
	  if(delta2 < epsilon_)
	    converged = true;
	}

	VERBOSE_ONLY( printf("delta = %f\n", min(delta1, delta2)) );


	if(converged) {
	  if(fine_tuning_enabled & not_fine) {
	    not_fine = false;
	    gabba = max_fine_tune_;
	    w_old.zeros();
	    w_old2.zeros();
	    used_nonlinearity = g_fine;
	    mu_ = mu_k * mu_orig;

	    end_fine_tuning = max_fine_tune_ + i;
	  }
	  else {
	    num_failures = 0;

	    B.col(round) = w;
            W.col(round) = trans(whitening_matrix) * w;

	    break; // this line is intended to take us to the next IC
	  }
	}
	else if(stabilization_enabled) {
	  converged = false;
          w_diff = w_old2 - w;
	    
	  if(dot(w_diff, w_diff) < epsilon_) {
	    converged = true;
	  } else {
            w_diff = w_old2 + w;
	      
	    if(dot(w_diff, w_diff) < epsilon_) {
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
	  else if(not_fine && (!taking_long) &&
		  (i > ((double) max_num_iterations_ / 2))) {
	    taking_long = true;
	    mu_ *= .5;
	    if((used_nonlinearity % 2) == 0) {
	      used_nonlinearity++;
	    }
	  }
	}

	w_old2 = w_old;
	w_old = w;
	  
	switch(used_nonlinearity) {

	case LOGCOSH: {
	  DeflationLogCoshUpdate_(n, X, w);
	  break;
	}

	case LOGCOSH + 1: {
	  DeflationLogCoshFineTuningUpdate_(n, X, w);
	  break;
	}

	case LOGCOSH + 2: {
	  arma::mat X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut_, X, X_sub);
	  DeflationLogCoshUpdate_(num_selected, X_sub, w);
	  break;
	}

	case LOGCOSH + 3: {
	  arma::mat X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut_, X, X_sub);
	  DeflationLogCoshFineTuningUpdate_(num_selected, X_sub, w);
	  break;
	}

	case GAUSS: {
	  DeflationGaussUpdate_(n, X, w);
	  break;
	}

	case GAUSS + 1: {
	  DeflationGaussFineTuningUpdate_(n, X, w);
	  break;
	}

	case GAUSS + 2: {
	  arma::mat X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut_, X, X_sub);
	  DeflationGaussUpdate_(num_selected, X_sub, w);
	  break;
	}

	case GAUSS + 3: {
	  arma::mat X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut_, X, X_sub);
	  DeflationGaussFineTuningUpdate_(num_selected, X_sub, w);
	  break;
	}

	case KURTOSIS: {
	  DeflationKurtosisUpdate_(n, X, w);
	  break;
	}

	case KURTOSIS + 1: {
	  DeflationKurtosisFineTuningUpdate_(n, X, w);
	  break;
	}

	case KURTOSIS + 2: {
	  arma::mat X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut_, X, X_sub);
	  DeflationKurtosisUpdate_(num_selected, X_sub, w);
	  break;
	}

	case KURTOSIS + 3: {
	  arma::mat X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut_, X, X_sub);
	  DeflationKurtosisFineTuningUpdate_(num_selected, X_sub, w);
	  break;
	}

	case SKEW: {
	  DeflationSkewUpdate_(n, X, w);
	  break;
	}

	case SKEW + 1: {
	  DeflationSkewFineTuningUpdate_(n, X, w);
	  break;
	}

	case SKEW + 2: {
	  arma::mat X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut_, X, X_sub);
	  DeflationSkewUpdate_(num_selected, X_sub, w);
	  break;
	}

	case SKEW + 3: {
	  arma::mat X_sub;
	  index_t num_selected = RandomSubMatrix(n, percent_cut_, X, X_sub);
	  DeflationSkewFineTuningUpdate_(num_selected, X_sub, w);
	  break;
	}
	    
	default: 
	  printf("ERROR: invalid contrast function: used_nonlinearity = %d\n",
		 used_nonlinearity);
	  exit(SUCCESS_FAIL);
	}
	
        w /= sqrt(dot(w, w)); // normalize
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
  int FixedPointICA(const arma::mat& X, arma::mat& whitening_matrix, arma::mat& W) {
    // ensure default values are passed into this function if the user doesn't care about certain parameters
    if(d < num_of_IC_) {
      printf("ERROR: must have num_of_IC <= dimension!\n");
      W.set_size(0);
      return SUCCESS_FAIL;
    }

    W.set_size(d, num_of_IC_);

    if((percent_cut_ > 1) || (percent_cut_ < 0)) {
      percent_cut_ = 1;
      printf("Setting percent_cut to 1\n");
    }
    else if(percent_cut_ < 1) {
      if((percent_cut_ * n) < 1000) {
	percent_cut_ = min(1000 / (double) n, (double) 1);
	printf("Warning: Setting percent_cut to %0.3f (%d samples).\n",
	       percent_cut_,
	       (int) floor(percent_cut_ * n));
      }
    }
    
    int g_orig = nonlinearity_;

    if(percent_cut_ != 1) {
      g_orig += 2;
    }
    
    if(mu_ != 1) {
      g_orig += 1;
    }

    bool fine_tuning_enabled = true;
    int g_fine;

    if(fine_tune_) {
      g_fine = nonlinearity_ + 1;
    } else {
      if(mu_ != 1)
	g_fine = g_orig;
      else
	g_fine = g_orig + 1;

      fine_tuning_enabled = false;
    }

    bool stabilization_enabled;
    if(stabilization_) {
      stabilization_enabled = true;
    } else {
      if(mu_ != 1)
	stabilization_enabled = true;
      else
	stabilization_enabled = false;
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

    arma::mat B;

    int ret_val = SUCCESS_FAIL;
    
    if(approach_ == SYMMETRIC) {
      ret_val = 
	SymmetricFixedPointICA(stabilization_enabled, fine_tuning_enabled,
			       mu_orig, mu_k, failure_limit,
			       used_nonlinearity, g_fine, stroke,
			       not_fine, taking_long, initial_state_mode,
			       X, B, W,
			       whitening_matrix);
    }
    else if(approach_ == DEFLATION) {
      ret_val = 
	DeflationFixedPointICA(stabilization_enabled, fine_tuning_enabled,
			       mu_orig, mu_k, failure_limit,
			       used_nonlinearity, g_orig, g_fine,
			       stroke, not_fine, taking_long, initial_state_mode,
			       X, B, W,
			       whitening_matrix);
    }

    return ret_val;
  }


  /**
   * Runs FastICA Algorithm on matrix X and sets W to unmixing matrix and Y to
   * independent components matrix, such that \f$ X = W * Y \f$
   */
  int DoFastICA(arma::mat& W, arma::mat& Y) {
    arma::mat X_centered, X_whitened, whitening_matrix;

    Center(X, X_centered);

    WhitenUsingEig(X_centered, X_whitened, whitening_matrix);

    // X_whitened is equal to the original implementation, but the rows are
    // permuted (apparently somewhat randomly) likely to due the ordering of
    // eigenvalues.  Signs may be different too (whitening_matrix reflects these
    // changes also).

    // by-hand changes to emulate old version's matrix ordering
    // row 4 = -row 1
    // row 5 = -row 2
    // row 3 = row 3
    // row 2 = row 4
    // row 1 = row 5
    /*arma::mat tmp(X_whitened.n_rows, X_whitened.n_cols);
    tmp.row(0) = X_whitened.row(4);
    tmp.row(1) = X_whitened.row(3);
    tmp.row(2) = X_whitened.row(2);
    tmp.row(3) = -X_whitened.row(0);
    tmp.row(4) = -X_whitened.row(1);
    X_whitened = tmp;
    arma::mat tmpw(whitening_matrix.n_rows, whitening_matrix.n_cols);
    tmpw.row(0) = whitening_matrix.row(4);
    tmpw.row(1) = whitening_matrix.row(3);
    tmpw.row(2) = whitening_matrix.row(2);
    tmpw.row(3) = -whitening_matrix.row(0);
    tmpw.row(4) = -whitening_matrix.row(1);
    whitening_matrix = tmpw;*/
  
    int ret_val =
      FixedPointICA(X_whitened, whitening_matrix, W);

    if(ret_val == SUCCESS_PASS) {
      Y = trans(W) * X;
    }
    else {
      Y.set_size(0);
    }

    return ret_val;
  }
}; /* class FastICA */

#endif /* FASTICA_H */
