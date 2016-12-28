/**
 * @file    elm.cpp
 * @author  Siddharth Agrawal
 * @mail    siddharthcore@gmail.com
 *
 * Implementation of Basic Extreme Learning Machine
 * Extreme Learning Machine(ELM) is a single-hidden layer feedforward neural networks(SLFNs) which randomly chooses hidden nodes and  
 * analytically determines the output weights of SLFNs. 
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "elm.hpp"
using namespace mlpack;
using namespace mlpack::elm;

ELM::ELM (const arma::mat& x_train,
           const arma::mat& y_train,
           const uint16_t act,
	   const uint16_t Nh,	 //Number of Hidden Neurons
           const uint16_t N,    //Number of data points
           const uint16_t D,   //Data Dimension
	   const double lambda = 0,
	   const double alpha = 0):
           act(act),
           Nh(Nh),
           N(N),
           D(D),
           lambda(lambda),
           alpha(alpha)

{
  arma::mat Weight = arma::randu<arma::mat>(Nh);
  arma::mat bias = arma::randu<arma::mat>(Nh);
  arma::mat beta = arma::randu<arma::mat>(Nh);
  //Train(x_train,y_train,act);
  arma_rng::set_seed_random();   
  Init_Weight_bias();
}

void ELM::Init_Weight_bias()
{
	bias.randu(Nh,1);
	std::mt19937 engine(time(0));  // Mersenne twister random number engine
	std::uniform_real_distribution<double> distr(1.0, 2.0); 
	Weight.set_size(Nh, D); 
	Weight.imbue( [&]() { return distr(engine); } );
}  

/*
 Train ELM
 Training Data set x_train and y_train; 
 Activation function
		0 - Sigmoid Function
		1 - Sine Function
		2 - Hardlim Function
		3 - Triangular Bias Function
		4 - Radial Basis Function
*/
void ELM::Train(const arma::mat& x_train,
            const arma::mat& y_train,
            const uint16_t act)
{
		mat param = x_train*Weight.t(); 
		mat H = zeros(N,Nh);
		
		switch(act)
		{
			case 0 : //Sigmoid Activation Function
					for(int i=0;i<H.n_rows;++i)
					{
						for(int j=0; j<H.n_cols;++j)
						{
							H(i,j) = 1.0 / (1.0 + exp(- (param(i,j)+bias(j))));
						}
					}
					break;
			case 1 :  //Sine Activation Function
					for(int i=0;i<H.n_rows;++i)
					{
						for(int j=0; j<H.n_cols;++j)
						{
							H(i,j) = sin(param(i,j)+bias(j));
						}
					}
					break;
			case 2 :  //Hardlim Activation Function
					for(int i=0;i<H.n_rows;++i)
					{
						for(int j=0; j<H.n_cols;++j)
						{
							H(i,j) = ((param(i,j)+bias(j)) > 0)? 1 : 0;
						}
					}
					break;
			case 3 :  //Traingular Bias Activation Function
					for(int i=0;i<H.n_rows;++i)
					{
						for(int j=0; j<H.n_cols;++j)
						{
							H(i,j) = ((param(i,j)+bias(j)) <= 1)&&((param(i,j)+bias(j)) >= -1) ? 
									  (1-abs(param(i,j)+bias(j))) : 0.0;
						}
					}
					break;
			case 4 ://Radial Basis	Activation Function
					
					for(int i=0;i<H.n_rows;++i)
					{
						for(int j=0; j<H.n_cols;++j)
						{
							H(i,j) = exp(- ((param(i,j)+bias(j))*(param(i,j)+bias(j))));
						}
					}
					break;
			default :   Log::Fatal << "Please select a suitable activation function to proceed:" << std::endl;

		}

		mat H_inv = pinv(H); // Moore-Penrose pseudo-inverse of matrix H

		beta = H_inv * y_train; //Calculate output weights 

		mat y_out = H * beta;  // Calculate training accuracy
	
		vec temp = y_train - y_out;

		double error = stddev(temp); //calculate training error
	        Log::Info << "Train RMSE :" << error <<std::endl;

}

/*
 Test ELM
 Testing Data set x_test and y_test; 
 Activation function
		0 - Sigmoid Function
		1 - Sine Function
		2 - Hardlim Function
		3 - Triangular Bias Function
		4 - Radial Basis Function
*/
void ELM::Test(const arma::mat& x_test,
            const arma::mat& y_test)
{
	
		mat param = x_test*Weight.t(); 
		mat H = zeros(N,Nh);
	 	
		switch(act)
		{
			case 0 : //Sigmoid Activation Function
					for(int i=0;i<H.n_rows;++i)
					{
						for(int j=0; j<H.n_cols;++j)
						{
							H(i,j) = 1.0 / (1.0 + exp(- (param(i,j)+bias(j))));
						}
					}
					break;
			case 1 :  //Sine Activation Function
					for(int i=0;i<H.n_rows;++i)
					{
						for(int j=0; j<H.n_cols;++j)
						{
							H(i,j) = sin(param(i,j)+bias(j));
						}
					}
					break;
			case 2 :  //Hardlim Activation Function
					for(int i=0;i<H.n_rows;++i)
					{
						for(int j=0; j<H.n_cols;++j)
						{
							H(i,j) = ((param(i,j)+bias(j)) > 0)? 1 : 0;
						}
					}
					break;
			case 3 :  //Traingular Bias Activation Function
					for(int i=0;i<H.n_rows;++i)
					{
						for(int j=0; j<H.n_cols;++j)
						{
							H(i,j) = ((param(i,j)+bias(j)) <= 1)&&((param(i,j)+bias(j)) >= -1) ? 
									  (1-abs(param(i,j)+bias(j))) : 0.0;
						}
					}
					break;
			case 4 ://Radial Basis	Activation Function
					
					for(int i=0;i<H.n_rows;++i)
					{
						for(int j=0; j<H.n_cols;++j)
						{
							H(i,j) = exp(- ((param(i,j)+bias(j) - alpha)*(param(i,j)+bias(j) - alpha))/lambda);
						}
					}
					break;
			default : Log::Fatal << "Please select a suitable activation function to proceed:" << std::endl;

		}

		mat y_out = H * beta; // calculate testing accuracy
		
		vec temp = y_test - y_out;
		
		double error = stddev(temp); //calculate testing error
		Log::Info << "Test RMSE :" << error <<std::endl;
        	
		arma::mat Elm_output = arma::randu<arma::mat>(N);
        	data::Save("Elm_output.csv", Elm_output);
		
}
