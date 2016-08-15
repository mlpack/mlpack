#include "test_function.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;


 
   
   
arma::mat CGTestFunction1::GetInitialPoint() const
{
return arma::mat(" 5;6");
}

   
double CGTestFunction1::Evaluate(const arma::mat& iterate) const
{
    

  return(arma::det(0.5*iterate.t()*A*iterate+B.t()*iterate)+c);

}

void CGTestFunction1::gradient(arma::mat& iterate,arma::mat& con_gradient)
{
  con_gradient=A*iterate+B;

}
double CGTestFunction1::ComputeStepSize(arma::mat& iterate,arma::mat& con_gradient)
{
  return(-arma::det(con_gradient.t()*(A*iterate+B))/arma::det(con_gradient.t()*A*con_gradient));
}
  


arma::mat CGTestFunction2::GetInitialPoint() const
{
  return arma::mat("15;20");
}


   
double CGTestFunction2::Evaluate(const arma::mat& iterate) const
{

      return(arma::det(0.5*(iterate.t()*A*iterate)+B.t()*iterate)+c);

}

void CGTestFunction2::gradient(arma::mat& iterate,arma::mat& con_gradient)
{
      con_gradient=A*iterate+B;

}



//Here we can use differect linesearch method. basically here we have to minimize g(alpha) wrt alpha. where g(alpha)=f(iterate+alpha*con_direction);
double CGTestFunction2::ComputeStepSize(arma::mat& iterate,arma::mat& con_gradient)
{ 

      return(-arma::det(con_gradient.t()*(A*iterate+B))/arma::det(con_gradient.t()*A*con_gradient));

}


