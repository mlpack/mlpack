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

ELM::ELM(const arma::mat& predictors,
         const arma::mat& responses,
         const size_t act,
	 const size_t Nh,	    //Number of Hidden Neurons
         const size_t N,           //Number of data points
         const size_t D,          //Data Dimension
	 const double lambda = 0,
	 const double alpha = 0):
    act(act),
    Nh(Nh),
    N(N),
    D(D),
    lambda(lambda),
    alpha(alpha)

{
  arma::mat weight = arma::randu<arma::mat>(Nh);
  arma::mat bias = arma::randu<arma::mat>(Nh);
  arma::mat beta = arma::randu<arma::mat>(Nh);
  //Train(predictors,responses,act);
  arma_rng::set_seed_random();   
  Initweightbias();
}

void ELM::Initweightbias()
{
  bias.randu(Nh,1);
  std::mt19937 engine(time(0));  // Mersenne twister random number engine
  std::uniform_real_distribution<double> distr(1.0, 2.0); 
  weight.set_size(Nh, D); 
  weight.imbue( [&]() { return distr(engine); } );
}  

/*
 Train ELM
 Training Data set predictors and responses; 
 Activation function
		0 - Sigmoid Function
		1 - Sine Function
		2 - Hardlim Function
		3 - Triangular Bias Function
		4 - Radial Basis Function
*/

void ELM::Train(const arma::mat& predictors,
                const arma::mat& responses,
                const size_t act)
{
  mat param = predictors*weight.t(); 
  mat H = zeros(N,Nh);
		
  switch(act)
  {
    case 0 :for(size_t i=0; i<H.n_rows; ++i)
	    {
	     for(size_t j=0; j<H.n_cols; ++j)
	     {
	      H(i,j) = 1.0 / (1.0 + exp(- (param(i,j)+bias(j))));
	     }
	    }
	    break;

    case 1 :for(size_t i=0; i<H.n_rows; ++i)
	    {
	     for(size_t j=0; j<H.n_cols; ++j)
	     {
	      H(i,j) = sin(param(i,j)+bias(j));
	     }
	    }
	    break;

    case 2 :for(size_t i=0; i<H.n_rows; ++i)
	    {
	     for(size_t j=0; j<H.n_cols; ++j)
	     {
	      H(i,j) = ((param(i,j)+bias(j)) > 0)? 1 : 0;
	     }
	    }
	    break;

    case 3 :for(size_t i=0;i<H.n_rows;++i)
	    {
	     for(size_t j=0; j<H.n_cols;++j)
	     {
	      H(i,j) = ((param(i,j)+bias(j)) <= 1)&&((param(i,j)+bias(j)) >= -1) ? (1-abs(param(i,j)+bias(j))) : 0.0;
	     }
	    }
	    break;

    case 4 :for(size_t i=0; i<H.n_rows; ++i)
	    {
	     for(size_t j=0; j<H.n_cols; ++j)
	     {
	      H(i,j) = exp(- ((param(i,j)+bias(j))*(param(i,j)+bias(j))));
             }
	    }
	    break;

    default : Log::Fatal << "Please select a suitable activation function to proceed:" << std::endl;
  }

  mat H_inv = pinv(H); // Moore-Penrose pseudo-inverse of matrix H
  beta = H_inv * responses; //Calculate output weights 
  mat trainingOutput = H * beta;  // Calculate training accuracy
  vec temp = responses - trainingOutput;
  double trainError = stddev(temp); //calculate training error
  Log::Info << "Train RMSE :" << trainError <<std::endl;
}

/*
 Predict ELM
 Data set points and predictions; 
 Activation function
		0 - Sigmoid Function
		1 - Sine Function
		2 - Hardlim Function
		3 - Triangular Bias Function
		4 - Radial Basis Function
*/

void ELM::Predict(const arma::mat& points,
                  const arma::mat& predictions)
{
  mat param = points*weight.t(); 
  mat H = zeros(N,Nh);
	 	
  switch(act)
  {
    case 0 :for(size_t i=0; i<H.n_rows; ++i)
	    {
	     for(size_t j=0; j<H.n_cols; ++j)
             {
	      H(i,j) = 1.0 / (1.0 + exp(- (param(i,j)+bias(j))));
	     }
	    }
	    break;

    case 1 :for(size_t i=0; i<H.n_rows; ++i)
	    {
	     for(size_t j=0; j<H.n_cols; ++j)
	     {
	      H(i,j) = sin(param(i,j)+bias(j));
	     }
	    }
	    break;

    case 2 :for(size_t i=0; i<H.n_rows; ++i)
	    {
	     for(size_t j=0; j<H.n_cols; ++j)
	     {
	      H(i,j) = ((param(i,j)+bias(j)) > 0)? 1 : 0;
	     }
	    }
	    break;

    case 3 :for(size_t i=0; i<H.n_rows; ++i)
	    {
	     for(size_t j=0; j<H.n_cols; ++j)
	     {
	      H(i,j) = ((param(i,j)+bias(j)) <= 1)&&((param(i,j)+bias(j)) >= -1) ? (1-abs(param(i,j)+bias(j))) : 0.0;
	     }
	    }
	    break;

    case 4 :for(size_t i=0; i<H.n_rows; ++i)
	    {
	     for(size_t j=0; j<H.n_cols; ++j)
  	     {
	      H(i,j) = exp(- ((param(i,j)+bias(j) - alpha)*(param(i,j)+bias(j) - alpha))/lambda);
	     }
	    }
	    break;

    default : Log::Fatal << "Please select a suitable activation function to proceed:" << std::endl;
  }

  mat predictions = H * beta; // predict the output of ELM
  arma::mat Elm_output = arma::randu<arma::mat>(N);
  Elm_output = predictions;
  data::Save("Elm_output.csv", predictions);	
}
