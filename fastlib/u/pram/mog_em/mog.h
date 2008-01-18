/**
 * @file mog.h
 *
 * Defines a Gaussian Mixture model to find the optimal parameters of the model
 */

#ifndef MOG_H
#define MOG_H

#include <fastlib/fastlib.h>

class MoG {

 private:

  ArrayList<Vector> mu_;
  ArrayList<Matrix> sigma_;
  Vector omega_;
  index_t number_of_gaussians_;
  index_t dimension_;

 public:

  MoG() {
    mu_.Init(0);
    sigma_.Init(0);
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

  void ExpectationMaximization(Matrix& data_points, ArrayList<double> *results);

  long double Loglikelihood(Matrix& data_points, ArrayList<Vector>& means,
			    ArrayList<Matrix>& covars, Vector& weights);

  void KMeans(Matrix& data, ArrayList<Vector> *means,
	      ArrayList<Matrix> *covars, Vector *weights, index_t value_of_k);
};

#endif
