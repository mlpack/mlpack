/**
 * @file elm.hpp
 * @author Siddharth Agrawal
 *
 * Basic Extreme Learning Machine
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ELM_HPP
#define MLPACK_METHODS_ELM_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace elm {

class ELM
{
private:
	uint16_t N_prime; //Number of Hidden Neurons
	uint16_t Dim; //Data Dimension
	uint16_t NI; //Number of data points
	mat Weight;
	vec bias;
	mat beta;
	uint16_t Activation;

	double Train_time;
	double Test_time;

	double Train_Accuracy;
	double Test_Accuracy;

public: 
	void Set_Dim(uint16_t Nh, uint16_t D, uint16_t N);
	void Init_Weight_bias(); //Initialise Weights and Biases
	void Config_ELM(uint16_t Nh, uint16_t N, uint16_t D);
	bool Train_ELM(mat &x_train, mat &y_train, uint16_t Act);
	bool Test_ELM(mat &x_test, mat &y_test);

	void Save_Model();
};

}
}

#endif
