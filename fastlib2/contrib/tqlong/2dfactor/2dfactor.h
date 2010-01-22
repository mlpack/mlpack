
#ifndef __2D_FACTOR_H
#define __2D_FACTOR_H
#include <fastlib/fastlib.h>

namespace la {

void row2dPCA(ArrayList<Matrix>& imageList, 
			  Vector& eigenValues, Matrix& rowBasis_out, Matrix& mean_out);

void col2dPCA(ArrayList<Matrix>& imageList, 
			  Vector& eigenValues, Matrix& colBasis_out, Matrix& mean_out);
			  
}; // namespace

#endif
