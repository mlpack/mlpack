/*
 * =====================================================================================
 *
 *       Filename:  conjugate_gradient.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  06/21/2007 01:13:30 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */


namespace sparse {
// solves the problem A*x=b
// where A is sparse semipositive definite 
template<typename T> 
void ConjugateGradient(Matrix<T> &A, T *b, T* x, T tolerance) {
	index_t dimension=A.get_dimension();
  T* r =  NewVector(dimension);
	T* p =  NewVector(dimension);	
  A.Multiply(x,r); // r=A*x
	VectorMinus(b, r) // r=b-r;
	memcpy(p, r, dimension*sizeof(T));
	T *temp1=NewVector(dimension);
  while() {
		A.Multiply(p, temp1); // temp1=A*p;
		T rr=VectorDotProduct(r, r, dimension);
	  T alpha=rr/VectorDotProduct(temp1, p, dimension); // a=(r,r)/(A*p,p)
    VectorPlusTimes(x, alpha, p, dimension, x);    // x=x+a*p
    T error=alpha*VectorDotProduct(p, p)/VectorDotProduct(x, x);
		if (error<tolerance) {
		  break;
		}
		VectorMinusTimes(r, alpha, temp1, r);          // r=r-a*A*p;
	  // beta=(r_j+1, r_j+1) / (r_j, r_j)
		T beta=VectorDotProduct(r, r, dimension)/rr;  
		// p = r + b*p
		VectorPlusTimes(r, beta, p, dimension, p);	
	} 
	DeleteVector(r, dimension);
	DeleteVector(p, dimension);
	DeleteVector(temp1, dimension);
}
};
