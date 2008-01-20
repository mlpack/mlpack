/**
 * @file mog.h
 *
 * Defines a Gaussian Mixture model and
 * estimates the parameters of the model
 * 
 */

#ifndef MOG_H
#define MOG_H

#include <fastlib/fastlib.h>

/**
 * A Gaussian mixture model class.
 * 
 * This class uses different loss functions to
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
 * MoG mog;
 * ArrayList<double> results;
 *
 * mog.Init(number_of_gaussians, dimension);
 * mog.L2Estimation(data, &results, optim_flag);
 * @endcode
 */
class MoG {

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

  MoG() {
    mu_.Init(0);
    sigma_.Init(0);
    d_sigma_.Init(0);
    d_omega_.Init(0, 0);
  }

  ~MoG() {
  }

  void Init(index_t num_gauss, index_t dimension) {

    // Initialize the private variables
    number_of_gaussians_ = num_gauss;
    dimension_ = dimension;

    // Resize the ArrayList of Vectors and Matrices
    mu_.Resize(number_of_gaussians_);
    sigma_.Resize(number_of_gaussians_);

    // Initialize sizes of the Vectors and Matrices
    omega_.Init(number_of_gaussians_);
    for (index_t i = 0; i < number_of_gaussians_; i++) {
      mu_[i].Init(dimension_);
      sigma_[i].Init(dimension_, dimension_);
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
   * MoG mog;
   * mog.MakeModel(number_of_gaussians, dimension,
   *               parameters_for_optimization);
   * @endcode
   */

  void MakeModel(index_t num_mods, index_t dimension, double* theta) {
    double *temp_mu; 
    Matrix lower_triangle_matrix, upper_triangle_matrix;
    double sum, s_min = 0.1;

    mu_.Clear();
    sigma_.Clear();
    d_sigma_.Clear();
    Init(num_mods, dimension);
    temp_mu = (double*) malloc (dimension_ * sizeof(double)) ;
    lower_triangle_matrix.Init(dimension_, dimension_);
    upper_triangle_matrix.Init(dimension_, dimension_);
			
    // calculating the omega values
    sum = 0;
    double *temp_array;
    temp_array = (double*) malloc (number_of_gaussians_ * sizeof(double));
    for(index_t i = 0; i < number_of_gaussians_ - 1; i++) {
      temp_array[i] = exp(theta[i]) ;
      sum += temp_array[i] ;
    }
    temp_array[number_of_gaussians_ - 1] = 1 ; 
    ++sum ;
    omega_.CopyValues(temp_array);
    la::Scale((1.0 / sum), &omega_);

    // calculating the mu values
    for(index_t k = 0; k < number_of_gaussians_; k++) {
      for(index_t j = 0; j < dimension_; j++) {
	temp_mu[j] = theta[number_of_gaussians_ + k*dimension_ + j - 1];
      }
      mu_[k].CopyValues(temp_mu);
    }
			
    // calculating the sigma values
    // using a lower triangular matrix and its transpose
    // to obtain a positive definite symmetric matrix
    for(index_t k = 0; k < number_of_gaussians_; k++) {
      lower_triangle_matrix.SetAll(0.0);
      for(index_t j = 0; j < dimension_; j++) {
	for(index_t i = 0; i < j; i++) {
	  lower_triangle_matrix.set(j, i, 
				    theta[(number_of_gaussians_ - 1) 
					  + number_of_gaussians_*dimension_ 
					  + k*(dimension_*(dimension_ + 1) / 2) 
					  + (j*(j + 1) / 2) + i]);
	}
	lower_triangle_matrix.set(j, j, 
				  theta[(number_of_gaussians_ - 1) 
					+ number_of_gaussians_*dimension_ 
					+ k*(dimension_*(dimension_ + 1) / 2) 
					+ (j*(j + 1) / 2) + j] + s_min);
					
      }
      la::TransposeOverwrite(lower_triangle_matrix, &upper_triangle_matrix);
      la::MulOverwrite(lower_triangle_matrix, upper_triangle_matrix, &sigma_[k]);
    }
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
   * MoG mog;
   * mog.MakeModelWithGradients(number_of_gaussians, dimension,
   *               parameters_for_optimization);
   * @endcode
   */

  void MakeModelWithGradients(index_t num_mods, index_t dimension, double* theta) {
    double *temp_mu;
    Matrix lower_triangle_matrix, upper_triangle_matrix;
    double sum, s_min = 0.1;

    Init(num_mods, dimension);
    temp_mu = (double*) malloc (dimension_ * sizeof(double));
    lower_triangle_matrix.Init(dimension_, dimension_);
    upper_triangle_matrix.Init(dimension_, dimension_);
    
    // calculating the omega values
    sum = 0;
    double *temp_array;
    temp_array = (double*) malloc (number_of_gaussians_ * sizeof(double));
    for(index_t i = 0; i < number_of_gaussians_ - 1; i++) {
      temp_array[i] = exp(theta[i]) ;
      sum += temp_array[i] ;
    }
    temp_array[number_of_gaussians_ - 1] = 1 ;  
    ++sum ;
    omega_.CopyValues(temp_array);
    la::Scale((1.0 / sum), &omega_);

    // calculating the d_omega values
    d_omega_.Destruct();
    d_omega_.Init(number_of_gaussians_ - 1, number_of_gaussians_);
    d_omega_.SetAll(0.0);
    for(index_t i = 0; i < number_of_gaussians_ - 1; i++) {
      for(index_t j = 0; j < i; j++) {
	d_omega_.set(i,j,-(omega(i)*omega(j)));
	d_omega_.set(j,i,-(omega(i)*omega(j)));
      }
      d_omega_.set(i,i,omega(i)*(1-omega(i)));
    }
    			
    // calculating the mu values
    for(index_t k = 0; k < number_of_gaussians_; k++) {
      for(index_t j = 0; j < dimension_; j++) {
	temp_mu[j] = theta[number_of_gaussians_ + k*dimension_ + j - 1];
      }
      mu_[k].CopyValues(temp_mu);
    }
    // d_mu is not computed because it is implicitly known
    // since no parameterization is applied on them
			
    // using a lower triangular matrix and its transpose 
    // to obtain a positive definite symmetric matrix
    
    // initializing the d_sigma values
    d_sigma_.Resize(number_of_gaussians_);
    for(index_t i = 0; i < number_of_gaussians_; i++)
      d_sigma_[i].Init(dimension_*(dimension_ + 1) / 2);

    // calculating the sigma values
    for(index_t k = 0; k < number_of_gaussians_; k++) {
      lower_triangle_matrix.SetAll(0.0);
      for(index_t j = 0; j < dimension_; j++) {
	for(index_t i = 0; i < j; i++) {
	  lower_triangle_matrix.set( j, i, 
				     theta[(number_of_gaussians_ - 1) 
					   + number_of_gaussians_*dimension_ 
					   + k*(dimension_*(dimension_ + 1) / 2) 
					   + (j*(j + 1) / 2) + i]) ;
	}
	lower_triangle_matrix.set(j, j, 
				  theta[(number_of_gaussians_ - 1) 
					+ number_of_gaussians_*dimension_ 
					+ k*(dimension_*(dimension_ + 1) / 2) 
					+ (j*(j + 1) / 2) + j] + s_min);
      }
      la::TransposeOverwrite(lower_triangle_matrix, &upper_triangle_matrix);
      la::MulOverwrite(lower_triangle_matrix, upper_triangle_matrix, &sigma_[k]);
				
      // calculating the d_sigma values
      for(index_t i = 0; i < dimension_; i++){
	for(index_t in = 0; in < i+1; in++){
	  Matrix d_sigma_d_r,d_sigma_d_r_t,temp_matrix_1,temp_matrix_2;
	  d_sigma_d_r.Init(dimension_, dimension_);
	  d_sigma_d_r_t.Init(dimension_, dimension_);
	  d_sigma_d_r.SetAll(0.0);
	  d_sigma_d_r_t.SetAll(0.0);
	  d_sigma_d_r.set(i,in,1.0);
	  d_sigma_d_r_t.set(in,i,1.0);
						
	  la::MulInit(d_sigma_d_r,upper_triangle_matrix,&temp_matrix_1);
	  la::MulInit(lower_triangle_matrix,d_sigma_d_r_t,&temp_matrix_2);
	  la::AddInit(temp_matrix_1,temp_matrix_2,&d_sigma_[k][(i*(i+1)/2)+in]);
	}
      }
      
    }
  }			
  
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
		
  ArrayList<Matrix>& d_sigma( index_t i){
    return d_sigma_[i];
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
   *
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
   * This function calculates the parameters of the model
   * using the Maximum Likelihood function via the 
   * Expectation Maximization (EM) Algorithm.
   *
   * @code
   * MoG mog;
   * Matrix data = "the data on which you want to fit the model";
   * ArrayList<double> results;
   * mog.ExpectationMaximization(data, &results);
   * @endcode
   */
  void ExpectationMaximization(Matrix& data_points, ArrayList<double> *results);

  /**
   * This function computes the loglikelihood of model.
   * This function is used by the 'ExpectationMaximization'
   * function.
   * 
   */
  long double Loglikelihood(Matrix& data_points, ArrayList<Vector>& means,
			    ArrayList<Matrix>& covars, Vector& weights);

  /**
   * This function computes the k-means of the data and stores
   * the calculated means and covariances in the ArrayList
   * of Vectors and Matrices passed to it. It sets the weights 
   * uniformly. 
   * 
   * This function is used to obtain a starting point for 
   * the optimization
   */
  void KMeans(Matrix& data, ArrayList<Vector> *means,
	      ArrayList<Matrix> *covars, Vector *weights, index_t value_of_k);

  /**
   * This function calculates the parameters of the model
   * using the L2 loss function and optimizes
   * using the polytope method or the quasi newton 
   * method as per the user's choice.
   *  -'1' is for the polytope method
   *  -anything else leads to quasi newton
   *
   * @code
   * MoG mog;
   * Matrix data = "the data on which you want to fit the model";
   * ArrayList<double> results;
   * mog.L2Estimation(data, &results, choice_of_optimizer);
   * @endcode
   */
  void L2Estimation(Matrix& data, ArrayList<double> *results, int optim_flag);

  /**
   * This function computes multiple number of starting points
   * required for the polytope method
   */
  void points_generator(Matrix& d, double **points, index_t number_of_points,
			index_t number_of_components);

  /** 
   * This function parameterizes the starting point obtained
   * from the 'KMeans" for optimization purposes
   *
   */
  void initial_point_generator (double *theta, Matrix& data,
				index_t number_of_components);
};

#endif
