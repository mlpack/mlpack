/**
 * @author pram
 * @file mog.h
 *
 * Defines a Gaussian Mixture model and
 * estimates the parameters of the model
 */

#ifndef MOGEM_H
#define MOGEM_H

#include <fastlib/fastlib.h>

/**
 * A Gaussian mixture model class.
 * 
 * This class uses maximum likelihood loss functions to
 * estimate the parameters of a gaussian mixture 
 * model on a given data via the EM algorithm.
 * 
 *
 * Example use:
 *
 * @code
 * MoGEM mog;
 * ArrayList<double> results;
 *
 * mog.Init(number_of_gaussians, dimension);
 * mog.ExpectationMaximization(data, &results, optim_flag);
 * @endcode
 */
class MoGEM {

 private:

  // The parameters of the mixture model
  ArrayList<Vector> mu_;
  ArrayList<Matrix> sigma_;
  Vector omega_;
  index_t number_of_gaussians_;
  index_t dimension_;

 public:

  MoGEM() {
    mu_.Init(0);
    sigma_.Init(0);
  }

  ~MoGEM() {
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

  void set_mu(index_t i, Vector& mu) {
    mu_[i].CopyValues(mu);
  }

  void set_sigma(index_t i, Matrix& sigma) {
    sigma_[i].CopyValues(sigma);
  }

  void set_omega(Vector& omega) {
    omega_.CopyValues(omega);
  }
		
  Matrix& sigma(index_t i) {
    return sigma_[i];
  }
		
  double omega(index_t i) {
    return omega_.get(i);
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
};

#endif
