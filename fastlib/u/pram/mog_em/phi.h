#include "fastlib/fastlib.h"


/* Calculates the multivariate Gaussian probability density function */

long double phi( Vector& x , Vector& mean , Matrix& cov) {
	
  long double det, f;
  double exponent;
  index_t dim;
  Matrix inv;
  Vector diff, tmp;
	
  dim = x.length();
  inv.Init( dim, dim );
  la::InverseOverwrite(cov, &inv);
  det = la::Determinant(cov);
  if( det < 0){
    det = -det;
  }
  la::SubInit(mean,x,&diff);
  la::MulInit(inv, diff, &tmp);
  exponent = la::Dot(diff, tmp);
  long double tmp1, tmp2, tmp3;
  tmp1 = 1;
  tmp2 = dim;
  tmp2 = tmp2/2;
  tmp2 = pow((2*(math::PI)),tmp2);
  tmp1 = tmp1/tmp2;
  tmp3 = 1;
  tmp2 = sqrt(det);
  tmp3 = tmp3/tmp2;
  tmp2 = -exponent;
  tmp2 = tmp2 / 2;
  f = (tmp1*tmp3*exp(tmp2));
  //printf("f --> %Lf\n",f);
  return f;
}

