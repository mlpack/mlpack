#include<iostream>
#include "huber_regression.hpp"
int main(){
      // Create some sample data
  arma::mat X = arma::randu<arma::mat>(100, 5);
  arma::vec y = 2 * X.col(0) + 0.5 * X.col(1) + 1.5 * X.col(2) +
  arma::randn<arma::vec>(100);

  // Instantiate the HuberRegressor
  mlpack::HuberRegressor hr(X,y,1.35, 100, 1e-5);



  // Print the coefficients
  std::cout << "Coefficients: " << std::endl << hr.getCoef() << std::endl;

  // Create some test data
  arma::mat X_test = arma::randu<arma::mat>(3, 5);

  // Predict the target values
  arma::vec y_pred = hr.Predict(X_test);

  // Print the predicted values
  std::cout << "Predicted values: " << std::endl << y_pred << std::endl;

  return 0;

}