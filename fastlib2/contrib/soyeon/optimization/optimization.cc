#include "optimization.h"
#include <iostream>

int Optimization::CheckPositiveDefiniteness(Matrix &hessian) {

}



void Optimization::ComputeDoglegDirection(int num_of_par,
																					double radius, 
																					double g_norm,
																					Matrix &hessian,
																					double *delta_m) {
	//check positive definiteness of the hessian
	if( CheckPositiveDefiniteness(hessian)==1) {

    //p_b=-hessian^-1*g
		Vector p_b;
		p_b.Init(num_of_par);

	}		//if
	else {	//if hessian is indefinite->use cauchy point

	}		//else



 

	
}







