#include "optimization.h"
#include "math.h"
#include <iostream>
#include <algorithm>	//max
using namespace std;	//max
 



void Optimization::ComputeDoglegDirection(int num_of_par,
																					double radius, 
																					Vector &gradient,
																					Matrix &hessian,
																					Vector &p,
																					double *delta_m) {
	//check positive definiteness of the hessian
	Matrix inverse_hessian;
	if( !PASSED(la::InverseInit(hessian, &inverse_hessian)) ) {

		//if hessian is indefinite->use cauchy point
		double gHg;

		Matrix transpose_hessian;
		la::TransposeInit(hessian, &transpose_hessian);

		Matrix temp1; //H*g
		la::MulInit(hessian, gradient, &temp1);
		gHg=la::Dot(temp1, gradient);

		double gradient_norm=sqrt(la::Dot(gradient, gradient));

		if(gHg<=0){
			//double gradient_norm=sqrt(la::Dot(gradient, gradient));
			//p=-(radius/gradient_norm)*gradient;
			la::ScaleInit(-1*radius/gradient_norm, gradient, p);
		}
		else{
			double zeta=0;
			zeta=min(pow(gradient_norm, 3)/(radius*gHg), 1);
			la::ScaleInit(-1*zeta*radius/gradient_norm, gradient, p);

		}

	}	//if
	else {	//if hessian matrix is positive definite
		la::InverseInit(hessian, &inverse_hessian);
		Vector p_b;
		//p_b= - (hessian)^-1 * g
		la::MulInit(inverse_hessian, gradient, &p_b);
		la::Scale(-1, p_b);

		double p_b_norm;
		//p_b_norm=la::Dot(p_b, p_b);
		p_b_norm=sqrt(la::Dot(p_b, p_b));

		if(radius>=p_b_norm){
			p->Copy(p_b);
		}
		else{
			//g'*H*g = (H*g)'g
			double gHg;

			Matrix transpose_hessian;
			la::TransposeInit(hessian, &transpose_hessian);

			/*
			//check whether hessian is symmetric
			double cnt=0;
			for(index_t i=0; i<hessian.n_rows(); i++) {
				for(index_t j=0; j<hessian.n_cols(); j++) {
					if(hessian.get(i,j) != transpose_hessian.get(i,j)){
						cnt+=1;
					}	//if
				}	//j
			}		//i
			if( cnt !=0 ) {
				NOTIFY("Hessian matrix is NOT symmetric.");
			}
			*/
			
			Matrix temp1; //H*g
			la::MulInit(hessian, gradient, &temp1);
			gHg=la::Dot(temp1, gradient);

			Vector p_u;
			//p_u= -(g'g/g'Hg)*g
			double p_u_norm;
			la::ScaleInit(-1*la::Dot(gradient, gradient)/gHg, gradient, &p_u);
			p_u_norm=math::Sqr(la::Dot(p_u, p_u));

			if( p_u_norm>=radius ) {	//p=radius/p_u_norm * p_u)
				la::ScaleInit(radius/p_u_norm, p_u, p);
			}	//if
			else{	//combination of p_u and p_b
				//solve the quadratic equation ||p_u-zeta(p_b-p_u)||^2=radius^2
				Vector diff; //p_b-p_u
				la::SubInit(p_u, p_b, &diff);
				double a=la::Dot(diff, diff);
				Vector diff2; //2p_u-p_b
				la::ScaleInit(2, p_u, &diff2);
				la::SubFrom(p_b, &diff2);
				double b=la::Dot(diff, diff2);
				double c=la::Dot(diff2, diff2)-math::Sqr(radius);

				double discriminant=sqrt(b*b-4*a*c);
				double zeta1=(-b+discriminant)/(2*a);
				double zeta2=(-b-discriminant)/(2*a);
				double zeta=-1;

				if( (zeta1<2)&&(zeta1>0)&&(zeta2<2)&&(zeta2>0)){
					zeta=max(zeta1, zeta2);
				}
				else if( (zeta1<2)&&(zeta1>0) ){
					zeta=zeta1;
				}
				else if((zeta2<2)&&(zeta2>0)){
					zeta=zeta2;
				}
				else{
					DEBUG_ASSERT_MSG((zeta>0), "Fail to get zeta");
				}

				if(zeta<=1){
					la::ScaleInit(zeta, p_u, p);
				}
				else{
					Vector temp; //(zeta-1)*(p_b-p_u)
					la::ScaleInit((zeta-1), diff, &temp);
					la::AddInit(p_u, temp, p);
				}		//else
			
			}	//else
		}
	}
}


	







