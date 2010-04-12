#include "obj_grad_hess_class.h"

void ObjectGradientHessian::Init(int &num_points_t){
	num_points_t_.Copy(num_points_t);
}



void ObjectGradientHessian::ApproxBetafnNominator(int &num_points_t, double *beta_nominator){
	random_t.init(num_points_t);
	int i;
	double w2=1/num_points_t;	//equal weights

	for (i=0; i<num_points_t; i++){
		random_t[i]=(i+1)/m;
		printf("%g ", random_t[t]); //cout<<random_t[i]
	}
	*beta_nominator = 0;
}




