#include "test_obj.h"
#include "iostream.h"

void RosenbrockFunction::Init(fx_module *module) {
  module_=module;
}

void RosenbrockFunction::ComputeObjective(Vector &x, double *objective) {
  
	double f1;
	double f2;
	f1=10*(x[1]-x[0]*x[0]);
	f2=1-x[0];

	*objective=0.5*(f1*f1+f2*f2);
 }

void RosenbrockFunction::ComputeGradient(Vector &x, Vector *gradient){
  
  //Matrix sum_quad;
	
  Vector gradient_temp;
	gradient_temp.Init(x.length());
	gradient_temp[0]=-1*200*x[0]*(x[1]-x[0]*x[0])-(1-x[0]);
	gradient_temp[1]=100*(x[1]-x[0]*x[0]);
	gradient->Copy(gradient_temp);
}

void RosenbrockFunction::ComputeHessian(Vector &x, Matrix *hessian){
	//Be careful!!
	//hessian = &sum_quad_; //hessian: address, *hessian: value at address
	//*hessian = sum_quad_;
	Matrix hessian_temp;
	hessian_temp.Init(x.length(), x.length());
	hessian_temp.set(0,0, -1*200*(x[1]-x[0]*x[0])+400*x[0]*x[0]+1);
	hessian_temp.set(1,1,100);
	hessian_temp.set(0,1,-1*200*x[0]);
	hessian_temp.set(1,0,-1*200*x[0]);
	
	/*
	//Check positive definiteness
	Vector eigen_hessian;
	la::EigenvaluesInit (hessian_temp, &eigen_hessian);
	double min_eigen=eigen_hessian[0];
	//cout<<"eigen_value:"<<endl;
	for(index_t i=0; i<eigen_hessian.length(); i++){
		//cout<<eigen_hessian[i]<<" ";
		if(eigen_hessian[i]<min_eigen){
			min_eigen=eigen_hessian[i];
		}

	}
	//cout<<endl;
	//cout<<"max_eigen="<<max_eigen<<endl;
	for(index_t i=0; i<eigen_hessian.length(); i++){
		hessian_temp.set(i,i,(hessian_temp.get(i,i)+min_eigen*(1.01)));
	}

	*/



	hessian->Copy(hessian_temp);

	
}





