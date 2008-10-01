/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file mog.h
 *
 * Defines a Gaussian Mixture model and
 * estimates the parameters of the model
 * 
 */

#ifndef MOGL2E_H
#define MOGL2E_H

#include <fastlib/fastlib.h>

const fx_entry_doc mog_l2e_entries[] = {
  {"K", FX_PARAM, FX_INT, NULL,
   " The number of Gaussians in the mixture model."
   " (defaults to 1)\n"},
  {"D", FX_RESERVED, FX_INT, NULL,
   " The number of dimensions of the data on which the"
   " the mixture model is to be fit.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc mog_l2e_doc = {
  mog_l2e_entries, NULL,
  " This program defines a Gaussian mixture model"
  " and calculates the L2 error for the present"
  " parameter setting to be used by an optimizer.\n"
};

/**
 * A Gaussian mixture model class.
 * 
 * This class uses L2 loss function to
 * estimate the parameters of a gaussian mixture 
 * model on a given data.
 * 
 * The parameters are converted for optimization
 * to maintain the following facts:
 * - the weights sum to one
 *  - for this, the weights were parameterized using
 *    the logistic function
 * - the covariance matrix is always positive definite
 *  - for this, the Cholesky decomposition is used
 *
 * Example use:
 *
 * @code
 * MoGL2E mog;
 * ArrayList<double> results;
 * double *params;
 *
 * mog.MakeModel(number_of_gaussians, dimension, params);
 * mog.L2Error(data);
 * mog.OutputResults(&results);
 * @endcode
 */
class MoGL2E {

 private:

  // The parameters of the Mixture model
  ArrayList<Vector> mu_;
  ArrayList<Matrix> sigma_;
  Vector omega_;
  index_t number_of_gaussians_;
  index_t dimension_;
  
  // The differential for the paramterization
  // for optimization
  Matrix d_omega_;
  ArrayList<ArrayList<Matrix> > d_sigma_;

 public:

  MoGL2E() {
    mu_.Init(0);
    sigma_.Init(0);
    d_sigma_.Init(0);
    d_omega_.Init(0, 0);
  }

  ~MoGL2E() {
  }

  void Init(index_t num_gauss, index_t dimension) {

    // Destruct everything to initialize afresh
    mu_.Clear();
    sigma_.Clear();
    d_sigma_.Clear();
    // Initialize the private variables
    number_of_gaussians_ = num_gauss;
    dimension_ = dimension;

    // Resize the ArrayList of Vectors and Matrices
    mu_.Resize(number_of_gaussians_);
    sigma_.Resize(number_of_gaussians_);
  }

  void Resize_d_sigma_() {
    d_sigma_.Resize(number_of_gaussians());
    for(index_t i =0; i < number_of_gaussians(); i++) {
      d_sigma_[i].Init(dimension()*(dimension()+1)/2);
    }
  }

  /**
   *
   * This function uses the parameters used for optimization
   * and converts it into athe parameters of a Gaussian 
   * mixture model. This is to be used when you do not want
   * the gradient values.
   *
   * Example use:
   *
   * @code
   * MoGL2E mog;
   * mog.MakeModel(number_of_gaussians, dimension,
   *               parameters_for_optimization);
   * @endcode
   */

  void MakeModel(index_t num_mods, index_t dimension, double* theta) {
    double *temp_mu; 
    Matrix lower_triangle_matrix, upper_triangle_matrix;
    double sum, s_min = 0.01;

    Init(num_mods, dimension);
    temp_mu = (double*) malloc (dimension * sizeof(double)) ;
    lower_triangle_matrix.Init(dimension, dimension);
    upper_triangle_matrix.Init(dimension, dimension);
			
    // calculating the omega values
    sum = 0;
    double *temp_array;
    temp_array = (double*) malloc (num_mods * sizeof(double));
    for(index_t i = 0; i < num_mods - 1; i++) {
      temp_array[i] = exp(theta[i]) ;
      sum += temp_array[i] ;
    }
    temp_array[num_mods - 1] = 1 ; 
    ++sum ;
    la::Scale(num_mods, (1.0 / sum), temp_array);
    set_omega(dimension, temp_array);

    // calculating the mu values
    for(index_t k = 0; k < num_mods; k++) {
      for(index_t j = 0; j < dimension; j++) {
	temp_mu[j] = theta[num_mods + k*dimension + j - 1];
      }
      set_mu(k, dimension, temp_mu);
    }
			
    // calculating the sigma values
    // using a lower triangular matrix and its transpose
    // to obtain a positive definite symmetric matrix
    Matrix sigma_temp;
    sigma_temp.Init(dimension, dimension);
    for(index_t k = 0; k < num_mods; k++) {
      lower_triangle_matrix.SetAll(0.0);
      for(index_t j = 0; j < dimension; j++) {
	for(index_t i = 0; i < j; i++) {
	  lower_triangle_matrix.set(j, i, 
				    theta[(num_mods - 1) 
					  + num_mods*dimension 
					  + k*(dimension*(dimension + 1) / 2) 
					  + (j*(j + 1) / 2) + i]);
	}
	lower_triangle_matrix.set(j, j, 
				  theta[(num_mods - 1) 
					+ num_mods*dimension 
					+ k*(dimension*(dimension + 1) / 2) 
					+ (j*(j + 1) / 2) + j] + s_min);
					
      }
      la::TransposeOverwrite(lower_triangle_matrix, &upper_triangle_matrix);
      la::MulOverwrite(lower_triangle_matrix, upper_triangle_matrix, &sigma_temp);
      set_sigma(k, sigma_temp);
    }
  }			

  void MakeModel(datanode *mog_l2e_module, double* theta) {
    index_t num_gauss = fx_param_int_req(mog_l2e_module, "K");
    index_t dimension = fx_param_int_req(mog_l2e_module, "D");
    MakeModel(num_gauss, dimension, theta);
  }

 /**
   *
   * This function uses the parameters used for optimization
   * and converts it into athe parameters of a Gaussian 
   * mixture model. This is to be used when you want
   * the gradient values.
   *
   * Example use:
   *
   * @code
   * MoGL2E mog;
   * mog.MakeModelWithGradients(number_of_gaussians, dimension,
   *                            parameters_for_optimization);
   * @endcode
   */

  void MakeModelWithGradients(index_t num_mods, index_t dimension, double* theta) {
    double *temp_mu;
    Matrix lower_triangle_matrix, upper_triangle_matrix;
    double sum, s_min = 0.01;

    Init(num_mods, dimension);
    temp_mu = (double*) malloc (dimension * sizeof(double));
    lower_triangle_matrix.Init(dimension, dimension);
    upper_triangle_matrix.Init(dimension, dimension);
    
    // calculating the omega values
    sum = 0;
    double *temp_array;
    temp_array = (double*) malloc (num_mods * sizeof(double));
    for(index_t i = 0; i < num_mods - 1; i++) {
      temp_array[i] = exp(theta[i]) ;
      sum += temp_array[i] ;
    }
    temp_array[num_mods - 1] = 1 ;  
    ++sum ;
    la::Scale(num_mods, (1.0 / sum), temp_array);
    set_omega(num_mods, temp_array);

    // calculating the d_omega values
    Matrix d_omega_temp;
    d_omega_temp.Init(num_mods - 1, num_mods);
    d_omega_temp.SetAll(0.0);
    for(index_t i = 0; i < num_mods - 1; i++) {
      for(index_t j = 0; j < i; j++) {
	d_omega_temp.set(i,j,-(omega(i)*omega(j)));
	d_omega_temp.set(j,i,-(omega(i)*omega(j)));
      }
      d_omega_temp.set(i,i,omega(i)*(1-omega(i)));
    }
    for(index_t i = 0; i < num_mods - 1; i++) {
      d_omega_temp.set(i, num_mods - 1, -(omega(i)*omega(num_mods - 1)));
    }
    set_d_omega(d_omega_temp);
    			
    // calculating the mu values
    for(index_t k = 0; k < num_mods; k++) {
      for(index_t j = 0; j < dimension; j++) {
	temp_mu[j] = theta[num_mods + k*dimension + j - 1];
      }
      set_mu(k, dimension, temp_mu);
    }
    // d_mu is not computed because it is implicitly known
    // since no parameterization is applied on them
			
    // using a lower triangular matrix and its transpose 
    // to obtain a positive definite symmetric matrix
    
    // initializing the d_sigma values

    Matrix d_sigma_temp;
    d_sigma_temp.Init(dimension, dimension);
    Resize_d_sigma_();

    // calculating the sigma values
    Matrix sigma_temp;
    sigma_temp.Init(dimension, dimension);
    for(index_t k = 0; k < num_mods; k++) {
      lower_triangle_matrix.SetAll(0.0);
      for(index_t j = 0; j < dimension; j++) {
	for(index_t i = 0; i < j; i++) {
	  lower_triangle_matrix.set( j, i, 
				     theta[(num_mods - 1) 
					   + num_mods*dimension 
					   + k*(dimension*(dimension + 1) / 2) 
					   + (j*(j + 1) / 2) + i]) ;
	}
	lower_triangle_matrix.set(j, j, 
				  theta[(num_mods - 1) 
					+ num_mods*dimension 
					+ k*(dimension*(dimension + 1) / 2) 
					+ (j*(j + 1) / 2) + j] + s_min);
      }
      la::TransposeOverwrite(lower_triangle_matrix, &upper_triangle_matrix);
      la::MulOverwrite(lower_triangle_matrix, upper_triangle_matrix, &sigma_temp);
      set_sigma(k, sigma_temp);
				
      // calculating the d_sigma values
      for(index_t i = 0; i < dimension; i++){
	for(index_t in = 0; in < i+1; in++){
	  Matrix d_sigma_d_r,d_sigma_d_r_t,temp_matrix_1,temp_matrix_2;
	  d_sigma_d_r.Init(dimension, dimension);
	  d_sigma_d_r_t.Init(dimension, dimension);
	  d_sigma_d_r.SetAll(0.0);
	  d_sigma_d_r_t.SetAll(0.0);
	  d_sigma_d_r.set(i,in,1.0);
	  d_sigma_d_r_t.set(in,i,1.0);
						
	  la::MulInit(d_sigma_d_r,upper_triangle_matrix,&temp_matrix_1);
	  la::MulInit(lower_triangle_matrix,d_sigma_d_r_t,&temp_matrix_2);
	  la::AddOverwrite(temp_matrix_1,temp_matrix_2,&d_sigma_temp);
	  set_d_sigma(k, (i*(i+1)/2)+in, d_sigma_temp);
	}
      }
    }
  }			
  
  ////// THE GET FUNCTIONS //////
  ArrayList<Vector>& mu() {
    return mu_;
  }				
		
  ArrayList<Matrix>& sigma() {
    return sigma_;
  }
		
  Vector& omega() {
    return omega_;
  }
		
  index_t number_of_gaussians() {
    return number_of_gaussians_;
  }			
		
  index_t dimension() {
    return dimension_;
  }
		
  Vector& mu(index_t i) {
    return mu_[i] ;
  }
		
  Matrix& sigma(index_t i) {
    return sigma_[i];
  }
		
  double omega(index_t i) {
    return omega_.get(i);
  }
  		
  Matrix& d_omega(){
    return d_omega_;
  }
		
  ArrayList<ArrayList<Matrix> >& d_sigma(){
    return d_sigma_;
  }
		
  ArrayList<Matrix>& d_sigma(index_t i){
    return d_sigma_[i];
  }

  ////// THE SET FUNCTIONS //////

  void set_mu(index_t i, Vector& mu) {
    DEBUG_ASSERT(i < number_of_gaussians());
    DEBUG_ASSERT(mu.length() == dimension()); 
    mu_[i].Copy(mu);
    return;
  }

  void set_mu(index_t i, index_t length, const double *mu) {
    DEBUG_ASSERT(i < number_of_gaussians());
    DEBUG_ASSERT(length == dimension());
    mu_[i].Copy(mu, length);
    return;
  }

  void set_sigma(index_t i, Matrix& sigma) {
    DEBUG_ASSERT(i < number_of_gaussians());
    DEBUG_ASSERT(sigma.n_rows() == dimension());
    DEBUG_ASSERT(sigma.n_cols() == dimension());
    sigma_[i].Copy(sigma);
    return;
  }

  void set_omega(Vector& omega) {
    DEBUG_ASSERT(omega.length() == number_of_gaussians());
    omega_.Copy(omega);
    return;
  }

  void set_omega(index_t length, const double *omega) {
    DEBUG_ASSERT(length == number_of_gaussians());
    omega_.Copy(omega, length);
    return;
  }

  void set_d_omega(Matrix& d_omega) {
    d_omega_.Destruct();
    d_omega_.Copy(d_omega);
    return;
  }

  void set_d_sigma(index_t i, index_t j, Matrix& d_sigma_i_j) {
    d_sigma_[i][j].Copy(d_sigma_i_j);
    return;
  }

  /**
   * This function outputs the parameters of the model
   * to an arraylist of doubles 
   *
   * @code
   * ArrayList<double> results;
   * mog.OutputResults(&results);
   * @endcode
   */
  void OutputResults(ArrayList<double> *results) {

    // Initialize the size of the output array
    (*results).Init(number_of_gaussians_ * (1 + dimension_*(1 + dimension_)));

    // Copy values to the array from the private variables of the class
    for (index_t i = 0; i < number_of_gaussians_; i++) {
      (*results)[i] = omega(i);
      for (index_t j = 0; j < dimension_; j++) {
	(*results)[number_of_gaussians_ + i*dimension_ + j] = mu(i).get(j);
	for (index_t k = 0; k < dimension_; k++) {
	  (*results)[number_of_gaussians_*(1 + dimension_) 
		   + i*dimension_*dimension_ + j*dimension_ 
		   + k] = sigma(i).get(j, k);
	}
      }
    }
  }

  /**
   * This function prints the parameters of the model
   *
   * @code
   * mog.Display();
   * @endcode
   */
  void Display(){

    // Output the model parameters as the omega, mu and sigma			
    printf(" Omega : [ ");
    for (index_t i = 0; i < number_of_gaussians_; i++) {
      printf("%lf ", omega(i));
    }
    printf("]\n");
    printf(" Mu : \n[");
    for (index_t i = 0; i < number_of_gaussians_; i++) {
      for (index_t j = 0; j < dimension_ ; j++) {
	printf("%lf ", mu(i).get(j));
      }
      printf(";");
      if (i == (number_of_gaussians_ - 1)) {
	printf("\b]\n");
      }
    }
    printf("Sigma : ");
    for (index_t i = 0; i < number_of_gaussians_; i++) {
      printf("\n[");
      for (index_t j = 0; j < dimension_ ; j++) {
	for(index_t k = 0; k < dimension_ ; k++) {
	  printf("%lf ",sigma(i).get(j, k));
	}
	printf(";");
      }
      printf("\b]");
    }
    printf("\n");
  }

  
  /** 
   * This function calculates the L2 error and 
   * the gradient of the error with respect to the 
   * parameters given the data and the parameterized
   * mixture
   * 
   * Example use:
   * 
   * @code
   * const Matrix data;
   * MoGL2E mog;
   * index_t num_gauss, dimension;
   * double *params; // get the parameters
   * 
   * mog.MakeModel(num_gauss, dimension, params);
   * mog.L2Error(data);
   * @endcode
   */
  long double L2Error(const Matrix&, Vector* = NULL);
  
  /**
   * Calculates the regularization value for a 
   * Gaussian mixture and its gradient with 
   * respect to the parameters
   * 
   * Used by the 'L2Error' function to calculate
   * the regularization part of the error
   */ 
  long double RegularizationTerm_(Vector* = NULL);
   
  /**
   * Calculates the goodness-of-fit value for a 
   * Gaussian mixture and its gradient with 
   * respect to the parameters
   * 
   * Used by the 'L2Error' function to calculate
   * the goodness-of-fit part of the error
   */ 
  long double GoodnessOfFitTerm_(const Matrix&, Vector* = NULL);

  /**
   * This function computes multiple number of starting points
   * required for the Nelder Mead method
   * 
   * Example use:
   * @code
   * double **p;
   * index_t n, num_gauss;
   * const Matrix data;
   * 
   * MoGL2E::MultiplePointsGeneratot(p, n, data, num_gauss);
   * @endcode
   */
  static void MultiplePointsGenerator(double**, index_t, 
				      const Matrix&, index_t);

  /** 
   * This function parameterizes the starting point obtained
   * from the 'k_means" for optimization purposes using the 
   * Quasi Newton method
   * 
   * Example use:
   * @code
   * double *p;
   * index_t num_gauss;
   * const Matrix data;
   * 
   * MoGL2E::InitialPointGeneratot(p, data, num_gauss);
   * @endcode
   */
  static void InitialPointGenerator(double*, const Matrix&, index_t);

  /**
   * This function computes the k-means of the data and stores
   * the calculated means and covariances in the ArrayList
   * of Vectors and Matrices passed to it. It sets the weights 
   * uniformly. 
   * 
   * This function is used to obtain a starting point for 
   * the optimization
   * 
   * Example use:
   * 
   * @code
   * const Matrix data;
   * ArrayList<Vector> *means;
   * ArrayList<Matrix> *covars;
   * Vector *weights;
   * index_t num_gauss;
   * 
   * ...
   * MoGL2E::KMeans(data, means, covars, weights, num_gauss);
   *@endcode 
   */
  static void KMeans_(const Matrix&, ArrayList<Vector>*,
		      ArrayList<Matrix>*, Vector*, index_t);

  /**
   * This function returns the indices of the minimum
   * element in each row of a matrix
   */
  static void min_element(Matrix&, index_t*);

  /**
   * This is the function which would be used for 
   * optimization. It creates its own object of 
   * class MoGL2E and returns the L2 error
   * and the gradient which are computed by
   * the functions of the class
   *
   */
  static long double L2ErrorForOpt(Vector& params,
				   const Matrix& data,
				   Vector *gradient) {

    MoGL2E model;
    index_t dimension = data.n_rows();
    index_t num_gauss;

    num_gauss = (params.length() + 1)*2 / ((dimension+1)*(dimension+2));
    // This check added here to see if 
    // the gradient is actually demanded here
    if (gradient != NULL) {
      model.MakeModelWithGradients(num_gauss, dimension, params.ptr());
      return model.L2Error(data, gradient);
    }
    else {
      model.MakeModel(num_gauss, dimension, params.ptr());
      return model.L2Error(data);
    }
  }

  /**
   * This is the function which should be used for 
   * optimization when there is no need to compute 
   * any gradients
   *
   */
  static long double L2ErrorForOpt(Vector& params, const Matrix& data) {
    return L2ErrorForOpt(params, data, NULL);
  }

};

#endif 
