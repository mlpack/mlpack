/**
 * @file elm_main.cpp
 * @author Siddharth Agrawal
 *
 * Main function for elm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 */

#include <mlpack/core.hpp>
#include "elm.hpp"

using namespace mlpack;
using namespace mlpack::elm;
using namespace arma;
using namespace std;

int main()
{
	ELM elm;

	mat x;
	mat y;

	/*Preparing Data */

	/*Load Train Data*/
	x.load("x_train.csv",csv_ascii);
	y.load("y_train.csv",csv_ascii);

	uint16_t Nh  = 2000; 	 //Number of Hidden Neurons
	uint16_t D   = x.n_cols; //Dimension of each X vector
	uint16_t N   = x.n_rows; //Number of data points
	uint16_t act = 0; 		 //Activation Type


	/*Configure the ELM Parameters*/
	elm.Config_ELM(Nh,N,D);

	/*Train the Data*/
	bool r = elm.Train_ELM(x,y, act); 

	/*Load the Test Data*/
        x.load("x_test.csv",csv_ascii);
	y.load("y_test.csv",csv_ascii);

	/*Test the Data*/
	r = elm.Test_ELM(x,y);

	return 0;
}
