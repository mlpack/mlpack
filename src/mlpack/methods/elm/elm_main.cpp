/**
 * @file    elm.hpp
 * @author  Siddharth Agrawal
 * @mail    siddharthcore@gmail.com
 *
 * Main function of Basic Extreme Learning Machine
 * Extreme Learning Machine(ELM) is a single-hidden layer feedforward neural  networks(SLFNs) which randomly chooses hidden nodes and  
 * analytically determines the output weights of SLFNs. 
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 */

#include <mlpack/core.hpp>
#include "elm.hpp"


PROGRAM_INFO("Basic Extreme Learning Machine(ELM)",
    "This program trains the ELM Algorithm on the given labeled training set. "
    "\n\n"

    "ELM is a three step model in which 1st the weights and bias are randomly"
    "chosen and in the 2nd step the hidden layer output matrix is calculated."
    "In the final step the output weight beta is caluclated."
    "\n\n"
    "The learning speed of ELM is extremely fast.the hidden node parameters"
    "are not only independent of the training data but also of each other. "
    "Although hidden nodes are important and critical,they need not be tuned."
    "\n\n"
    "Unlike conventional learning methods which MUST see the training data "
    "before generating the hidden node parameters, ELM could generate the hidden"
    "node parameters before seeing the training data."
);

// Training parameters.
PARAM_MATRIX_IN("training", "A matrix containing the input training set.", "trainingData_x");
PARAM_MATRIX_IN("training", "A matrix containing the target training set.", "trainingData_y");

// Testing parameters.
PARAM_MATRIX_IN("test", "A matrix containing the input test set.", "testingData_x");
PARAM_MATRIX_IN("test", "A matrix containing the target test set.", "testingData_y");

using namespace mlpack;
using namespace mlpack::elm;
using namespace arma;
using namespace std;

int main(int argc, char* argv[])
{
        // Handle parameters.
  	CLI::ParseCommandLine(argc, argv);
	
	ELM elm;

        elm.Lambda() = 5;
        elm.Alpha() = 0.2;


        /*Load the Train Data*/

	mat trainingData_x = data::Load("training_x.csv",training_x);
        mat trainingData_y = data::Load("training_y.csv",training_y);
    
	/*Load the Test Data*/

	mat testingData_x = data::Load("testing_x.csv",testing_x);
        mat testingData_y = data::Load("testing_y.csv",testing_y);
	
	Log::Info << "Choose the number of hidden neurons" << std::endl;
	const uint16_t Nh = CLI::GetParam<uint16_t>("Number_of_hidden_neurons");
        const uint16_t D = trainingData_x.n_cols;
	const uint16_t N = trainingData_x.n_rows;

	Log::Info << "Choose an activation function" << std::endl;
 	Log::Info << "0 - Sigmoid Function	1 - Sine Function	2 - Hardlim Function	3 - Triangular Bias Function	4 - Radial Basis Function" << std::endl;
		
	const uint16_t act = CLI::GetParam<uint16_t>("Activation_type");
        elm.Nh = Nh;
	elm.D = D;
	elm.N = N;
	elm.act = act;

         /*Train the Data*/
	elm.Train(trainingData_x,trainingData_y,act);

	/*Test the Data*/
	elm.Test(testingData_x,testingData_y);

	return 0;
}
