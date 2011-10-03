#include <mlpack/core.h>
#include "linear_regression.h"

int main(int argc, char* argv[]) {

  arma::vec B;
  arma::colvec responses;
  arma::mat predictors, file, points;

  // The input predictors NxD
  data::Load("input.csv", file);
  predictors = file.submat(0,0, file.n_rows-2, file.n_cols-1);
  // The initial predictors for y, Nx1
  responses = arma::trans(file.row(file.n_rows-1));
  
  size_t n_cols = predictors.n_cols,
	 n_rows = predictors.n_rows;

  predictors.insert_rows(0, arma::ones<arma::rowvec>(n_cols));
  ++n_rows;

  arma::rowvec predictions;

  mlpack::LinearRegression lr(predictors, responses);
  lr.predict(predictions, points);

  //data.row(n_rows) = predictions;
  //data::Save("out.csv", data);
  std::cout << "predictions: " << arma::trans(predictions) << '\n';

  return 0;
}
