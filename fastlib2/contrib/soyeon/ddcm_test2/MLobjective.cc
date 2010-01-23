#include "MLobjective.h"
#include <cmath>
#include <iostream>
#include <iostream>
#include <algorithm> //sqrt

using namespace std;
  


void MLObjective::Init2(int count_init2) {

	if(count_init2 ==0) {
	first_stage_x_.Init();
	first_stage_y_.Init();
}


	exp_betas_times_x1_.Init();
  ///exp_betas_times_x2_.Init();
	///postponed_probability_.Init();
	//for(index_t i=0; i<postponed_probability_.size(); i++) {
  //  postponed_probability_[i]=0;
  //}

	///denumerator_beta_function_=0;
	///num_of_t_beta_fn_=100;
	///t_weight_=1;
	///num_of_alphas_=100;
	///alpha_weight_=1;  

	//from here for the gradient
	first_stage_dot_logit_.Init();
	first_stage_ddot_logit_.Init();

	//for(index_t i=0; i<first_stage_dot_logit_.size(); i++) {
    //exp_betas_times_x1_[i]=0;
		//first_stage_dot_logit_[i].Init(first_stage_x_[i].n_cols());
		//first_stage_dot_logit_[i].SetZero();
		//first_stage_ddot_logit_[i].Init(first_stage_x_[i].n_cols(),first_stage_x_[i].n_cols());
		//first_stage_ddot_logit_[i].SetZero();

  //}
	

	///second_stage_dot_logit_.Init();
	///second_stage_ddot_logit_.Init();

	//for(index_t i=0; i<second_stage_dot_logit_.size(); i++) {
    //exp_betas_times_x2_[i]=0;
		//second_stage_dot_logit_[i].Init(first_stage_x_[i].n_cols());
		//second_stage_dot_logit_[i].SetZero();
		//second_stage_ddot_logit_[i].Init(first_stage_x_[i].n_cols(),first_stage_x_[i].n_cols());
		//second_stage_ddot_logit_[i].SetZero();

  //}

	

	


	//for(index_t i=0; i<first_stage_x_.size(); i++) {
		//sum_first_derivative_p_beta_fn_[i]=0;
		//sum_second_derivative_p_beta_fn_[i]=0;
		//sum_first_derivative_q_beta_fn_[i]=0;
		//sum_second_derivative_q_beta_fn_[i]=0;
		//sum_second_derivative_p_q_beta_fn_[i]=0;
		//sum_second_derivative_conditionl_postponed_p_[i].Init(num_of_betas_);
		//sum_second_derivative_conditionl_postponed_p_[i].SetZero();
		//sum_second_derivative_conditionl_postponed_q_[i].Init(num_of_betas_);
		//sum_second_derivative_conditionl_postponed_q_[i].SetZero();

	//}
	

	count_init2+=1;


}





void MLObjective::Init3(int sample_size,
						ArrayList<Matrix> &added_first_stage_x,
						ArrayList<index_t> &added_first_stage_y) {

	
	num_of_betas_=added_first_stage_x[0].n_rows();
	int num_selected_people=added_first_stage_x.size();

	for(index_t i=sample_size; i<num_selected_people; i++){
		first_stage_x_.PushBackCopy(added_first_stage_x[i]);
		first_stage_y_.PushBackCopy(added_first_stage_y[i]);

	}
	exp_betas_times_x1_.Destruct();
	exp_betas_times_x1_.Init(num_selected_people);

	
	//from here for the gradient
	first_stage_dot_logit_.Destruct();
	first_stage_dot_logit_.Init(num_selected_people);
	first_stage_ddot_logit_.Destruct();
	first_stage_ddot_logit_.Init(num_selected_people);

	for(index_t i=0; i<first_stage_dot_logit_.size(); i++) {
    exp_betas_times_x1_[i]=0;
		first_stage_dot_logit_[i].Init(first_stage_x_[i].n_cols());
		first_stage_dot_logit_[i].SetZero();
		first_stage_ddot_logit_[i].Init(first_stage_x_[i].n_cols(),first_stage_x_[i].n_cols());
		first_stage_ddot_logit_[i].SetZero();

  }
	
	

	




	

	
	//first_stage_x_.Init(num_select
}


//void Objective::ComputeObjective(Matrix &x, double *objective) {
void MLObjective::ComputeObjective(double current_sample,
																 Vector &current_parameter, 
																 double *objective) { 
	
	Vector betas;
  //betas.Alias(x.ptr(), x.n_rows());
	
	

	betas.Alias(current_parameter.ptr(), num_of_betas_);
  
	
	ComputeExpBetasTimesX1_(betas);

	
  ///ComputeDeumeratorBetaFunction_(p, q);
	
  ///ComputePostponedProbability_(betas, 
  ///                             p, 
  ///                             q);

//cout<<"term1="<<ComputeTerm1_(betas)<<endl;
//cout<<"term2="<<ComputeTerm2_()<<endl;
//cout<<"term3="<<ComputeTerm3_()<<endl;
  /**objective = ComputeTerm1_(betas) 
               + ComputeTerm2_()
               + ComputeTerm3_();
*/

	
	*objective = (-1/current_sample)*(ComputeTerm1_(betas));

//	cout<<"The objective="<<*objective<<endl;
	


	//*objective=2;

	
	
}

////////////////////////////////////////////////
////Calculate gradient
////////////////////////////////////////////////

void MLObjective::ComputeGradient(double current_sample,
																Vector &current_parameter, 
																Vector *gradient) { 
	
	Vector betas;
  //betas.Alias(x.ptr(), x.n_rows());
	betas.Alias(current_parameter.ptr(), num_of_betas_);
  
	
	ComputeExpBetasTimesX1_(betas);
	
  	
	/*
  *objective = ComputeTerm1_(betas) 
               + ComputeTerm2_() 
               + ComputeTerm3_();

	*/
	
	ComputeDotLogit_(betas);
	ComputeDDotLogit_();
	//cout<<"ddot done"<<endl;
	
	Vector dummy_beta_term1;
		

	ComputeDerivativeBetaTerm1_(&dummy_beta_term1);
	
	//cout<<"DerivativeBetaTerm3 done"<<endl;
	
	/*
	cout<<"dummy_beta_term1="<<endl;
	for(index_t i=0; i<dummy_beta_term1.length(); i++){
		cout<<dummy_beta_term1[i]<<" "<<endl;
	}
	*/

	/*
	cout<<"dummy_beta_term2="<<endl;
	for(index_t i=0; i<dummy_beta_term2.length(); i++){
		cout<<dummy_beta_term2[i]<<" "<<endl;
	}
	*/


	/*
	cout<<"dummy_beta_term3="<<endl;
	for(index_t i=0; i<dummy_beta_term3.length(); i++){
		cout<<dummy_beta_term3[i]<<" "<<endl;
	}
	*/

	
  Vector dummy_gradient;
	dummy_gradient.Init(num_of_betas_);
	dummy_gradient.SetZero();

	for(index_t i=0; i<num_of_betas_; i++){
		dummy_gradient[i]=dummy_beta_term1[i];
	}
	

	la::Scale(-1.0/current_sample, &dummy_gradient);
	//la::Scale(+1.0/current_sample, &dummy_gradient);
	gradient->Copy(dummy_gradient);
												
}



////////////////////////////////////////////////
////Calculate hessian
////////////////////////////////////////////////

void MLObjective::ComputeHessian(double current_sample,
															 Vector &current_parameter, 
															 Matrix *hessian) { 
	
	Vector betas;
  //betas.Alias(x.ptr(), x.n_rows());
	betas.Alias(current_parameter.ptr(), num_of_betas_);
  
	ComputeExpBetasTimesX1_(betas);
	
  
	
	ComputeDotLogit_(betas);
	ComputeDDotLogit_();
	////cout<<"ddot done"<<endl;
  
	
	
	Matrix dummy_second_beta_term1;
	

	ComputeSecondDerivativeBetaTerm1_(&dummy_second_beta_term1);
		

	
	/*
	cout<<"p_beta_term1"<<endl;
	for(index_t i=0; i<dummy_p_beta_term1.length(); i++){
		cout<<dummy_p_beta_term1[i]<<" ";
	}
	cout<<endl;

	
	cout<<"p_beta_term2"<<endl;
	for(index_t i=0; i<dummy_p_beta_term1.length(); i++){
		cout<<dummy_p_beta_term2[i]<<" ";
	}
	cout<<endl;
  
  cout<<"p_beta_term3"<<endl;
	for(index_t i=0; i<dummy_p_beta_term3.length(); i++){
		cout<<dummy_p_beta_term3[i]<<" ";
	}
	cout<<endl;

	//index_t i=
	cout<<"p_beta_term: "<<endl;
	for(index_t i=0; i<dummy_p_beta_term1.length(); i++){
		cout<<( (dummy_p_beta_term1[i])+(dummy_p_beta_term2[i])+(dummy_p_beta_term2[i]) )<<" ";
	}
	cout<<endl;
	*/

	
  
	Matrix dummy_hessian;
  dummy_hessian.Init(num_of_betas_,num_of_betas_);
  dummy_hessian.SetZero();
	//la::AddOverwrite(dummy_second_beta_term1, dummy_second_beta_term2, &dummy_hessian_beta);
	//la::AddTo(dummy_second_beta_term3, &dummy_hessian_beta);
	 
	/*
  cout<<"Hessian matrix beta"<<endl;	
	for (index_t j=0; j<dummy_second_beta_term3.n_rows(); j++){
		for (index_t k=0; k<dummy_second_beta_term3.n_cols(); k++){
				cout<<dummy_hessian_beta.get(j,k) <<"  ";
		}
		cout<<endl;
	}
	*/


	for(index_t i=0; i<num_of_betas_; i++){
		for(index_t j=0; j<num_of_betas_; j++){

			dummy_hessian.set(i,j, dummy_second_beta_term1.get(i,j));
				
		 
		}	//j
	}		//i

	/*
	cout<<"Hessian matrix beta1"<<endl;	
	for (index_t j=0; j<dummy_second_beta_term1.n_rows(); j++){
		for (index_t k=0; k<dummy_second_beta_term1.n_cols(); k++){
				cout<<dummy_second_beta_term1.get(j,k) <<"  ";
		}
		cout<<endl;
	}

	cout<<"Hessian matrix beta2"<<endl;	
	for (index_t j=0; j<dummy_second_beta_term1.n_rows(); j++){
		for (index_t k=0; k<dummy_second_beta_term1.n_cols(); k++){
				cout<<dummy_second_beta_term2.get(j,k) <<"  ";
		}
		cout<<endl;
	}

	cout<<"Hessian matrix beta3"<<endl;	
	for (index_t j=0; j<dummy_second_beta_term3.n_rows(); j++){
		for (index_t k=0; k<dummy_second_beta_term3.n_cols(); k++){
				cout<<dummy_hessian.get(j,k) <<"  ";
		}
		cout<<endl;
	}
	*/

	/*
	cout<<"Hessian matrix part1"<<endl;	
	for (index_t j=0; j<dummy_second_beta_term3.n_rows(); j++){
		for (index_t k=0; k<dummy_second_beta_term3.n_cols(); k++){
				cout<<dummy_hessian.get(j,k) <<"  ";
		}
		cout<<endl;
	}
	*/


		
	//la::Scale(-1.0/current_sample, &dummy_hessian);
	cout<<"Hessian matrix before correction"<<endl;	
	for (index_t j=0; j<dummy_hessian.n_rows(); j++){
		for (index_t k=0; k<dummy_hessian.n_cols(); k++){
				cout<<dummy_hessian.get(j,k) <<"  ";
		}
		cout<<endl;
	}
	

	

	
	la::Scale(-1.0, &dummy_hessian);	
  
	//Check positive definiteness
	Vector eigen_hessian;
	la::EigenvaluesInit (dummy_hessian, &eigen_hessian);

	cout<<"eigen values"<<endl;

	for(index_t i=0; i<eigen_hessian.length(); i++){
		cout<<eigen_hessian[i]<<" ";
	}
	cout<<endl;

	double max_eigen=0;
	//cout<<"eigen_value:"<<endl;
	for(index_t i=0; i<eigen_hessian.length(); i++){
		//cout<<eigen_hessian[i]<<" ";
		if(eigen_hessian[i]>max_eigen){
			max_eigen=eigen_hessian[i];
		}

	}
	//cout<<endl;
	cout<<"max_eigen="<<(max_eigen/current_sample)<<endl;

	if(max_eigen>0){
		NOTIFY("Hessian is not Negative definite..Modify...");
		for(index_t i=0; i<eigen_hessian.length(); i++){
			dummy_hessian.set(i,i,(dummy_hessian.get(i,i)-max_eigen*(1.01)));
		}
	}


	
	


	//handle minimization
	//la::Scale(-1.0, &dummy_hessian);
	//la::Scale(-1.0/current_sample, &dummy_hessian);
	la::Scale(+1.0/current_sample, &dummy_hessian);

	cout<<"Modified Hessian matrix"<<endl;	
	for (index_t j=0; j<dummy_hessian.n_rows(); j++){
		for (index_t k=0; k<dummy_hessian.n_cols(); k++){
				cout<<dummy_hessian.get(j,k) <<"  ";
		}
		cout<<endl;
	}



	hessian->Copy(dummy_hessian);
	//cout<<"hessian done"<<endl;
												
}

void MLObjective::ComputeChoiceProbability(Vector &current_parameter, 
																				 Vector *choice_probability) {

	Vector betas;
  //betas.Alias(x.ptr(), x.n_rows());
	
	

	betas.Alias(current_parameter.ptr(), num_of_betas_);
  
	choice_probability->Init(first_stage_x_.size());

	
	ComputeExpBetasTimesX1_(betas);

	
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) { 
			//(*choice_probability)[n]=log(postponed_probability_[n]);
			(*choice_probability)[n]=0;
			
    } else {
		

      Vector temp;
      first_stage_x_[n].MakeColumnVector((first_stage_y_[n]-1), &temp);
			//(*choice_probability)[n]=la::Dot(betas, temp)-log(exp_betas_times_x1_[n])+log((1-postponed_probability_[n]));
			(*choice_probability)[n]=la::Dot(betas, temp)-log(exp_betas_times_x1_[n]);
     }
			
		
  }


}



///////////////////////////////////////////
//////////////////////////////////////////////////////////////////

double MLObjective::ComputeTerm1_(Vector &betas) {
  double term1=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) { 
			//first_stage_y_[n]=-1 if all==zero, j_i is n chose j_i
      continue;
    } else {
      Vector temp;
      first_stage_x_[n].MakeColumnVector((first_stage_y_[n]-1), &temp);
			//cout<<"term1"<<endl;
			//cout<<(la::Dot(betas, temp) - log(exp_betas_times_x1_[n]))<<endl;
			//cout<<(la::Dot(betas, temp))<<endl;
			//cout<<(log(exp_betas_times_x1_[n]))<<endl;
			//cout<<"exp_betas_x1="<<exp_betas_times_x1_[n]<<endl;

			//if(la::Dot(betas, temp) - log(exp_betas_times_x1_[n])<1e-7){
			//	term1+=0;
			//}
			//else{
				term1+=la::Dot(betas, temp) - log(exp_betas_times_x1_[n]);
			//}

			
    }
  }
	//cout<<"term1="<<term1<<endl;
  return term1;
	
}





void MLObjective::ComputeExpBetasTimesX1_(Vector &betas) {
  
  //double sum=0;
	for(index_t n=0; n<first_stage_x_.size(); n++){
		exp_betas_times_x1_[n]=0;
		for(index_t j=0; j<first_stage_x_[n].n_cols(); j++) {
			exp_betas_times_x1_[n]+=exp(la::Dot(betas.length(), 
															betas.ptr(), 
															first_stage_x_[n].GetColumnPtr(j)));
		}
  }
}




//////////////////////////////////////////////////////////
//add new things from here for objective2 (Compute gradient) 
//Compute dot_logit
void MLObjective::ComputeDotLogit_(Vector &betas) {

	/*for(index_t n=0; n<first_stage_dot_logit_.size(); n++) {
		first_stage_dot_logit_[n].Init(first_stage_x_[n].n_cols());
    first_stage_dot_logit_[n].SetZero();
  }
	*/


	//cout<<"test "<<first_stage_dot_logit_[1][1]<<endl;
	for(index_t n=0; n<first_stage_x_.size(); n++){
		first_stage_dot_logit_[n].SetZero();
		for(index_t i=0; i<first_stage_x_[n].n_cols(); i++){
			first_stage_dot_logit_[n][i]=(exp(la::Dot( betas.length(), betas.ptr(),
																				 first_stage_x_[n].GetColumnPtr(i) )))/
																				 exp_betas_times_x1_[n];
			//cout<<"test "<<first_stage_dot_logit_[n][i]<<endl;
		}	//i
	}	//n
	

	/*cout<<"first_stage_dot_logit_[1]="<<endl;
	for(index_t i=0; i<first_stage_x_[1].n_cols(); i++){
		cout<<first_stage_dot_logit_[1][i]<<" ";
		}	//i
	cout<<endl;
*/

	


}


void MLObjective::ComputeDDotLogit_() {
	/*for(index_t n=0; n<first_stage_ddot_logit_.size(); n++) {
		first_stage_ddot_logit_[n].Init(first_stage_x_[n].n_cols(), first_stage_x_[n].n_cols());
		first_stage_ddot_logit_[n].SetZero();
	}
	*/


	for(index_t n=0; n<first_stage_x_.size(); n++){
		for(index_t i=0; i<first_stage_x_[n].n_cols(); i++){
			first_stage_ddot_logit_[n].set(i, i, first_stage_dot_logit_[n][i]);
		}	//i
		//for(index_t j=0; i<second_stage_x_.n_cols(); j++){
		//	second_stage_ddot_logit_[n].set(j, j, second_stage_dot_logit_[n].get(j,1));
		//}	//j
	}	//n

	/*
	for(index_t i=0; i<first_stage_x_[1].n_cols(); i++){
			cout<<first_stage_ddot_logit_[1].get(i, i)<<" ";
	}	//i
	*/

	/*
	cout<<"first_stage_ddot_logit[0]"<<endl;
	for(index_t i=0; i<first_stage_ddot_logit_[0].n_rows(); i++){
		for(index_t j=0; j<first_stage_ddot_logit_[0].n_cols(); j++){
			cout<<first_stage_ddot_logit_[0].get(i,j)<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
	*/


}


void MLObjective::ComputeDerivativeBetaTerm1_(Vector *beta_term1) {
	//Vector derivative_beta_term1;
	//derivative_beta_term1.Init(betas.length());
	//derivative_beta_term1.SetZero();

	Vector temp;
	temp.Init(num_of_betas_);

	Vector temp2;
	temp2.Init(num_of_betas_);

	Vector temp3;
	temp3.Init(num_of_betas_);
	temp3.SetZero();

  for(index_t n=0; n<first_stage_x_.size(); n++) {
		
    if (first_stage_y_[n]<0) { 
			//first_stage_y_[n]=-1 if all==zero, j_i is n chose j_i
      continue;
    } else {
  
			la::MulOverwrite(first_stage_x_[n], first_stage_dot_logit_[n], &temp);
			//check2

			la::SubOverwrite(num_of_betas_, temp.ptr(), first_stage_x_[n].GetColumnPtr(first_stage_y_[n]-1), temp2.ptr());
			
																									
		}	//else
		la::AddTo(temp2, &temp3);
		
  }	//n
	//beta_term1=&temp3;
	beta_term1->Copy(temp3);
  //return derivative_beta_term1;
	
}





void MLObjective::ComputeSecondDerivativeBetaTerm1_(Matrix *second_beta_term1) {
	//check
	Matrix second_derivative_beta_term1;
	second_derivative_beta_term1.Init(num_of_betas_, num_of_betas_);
	second_derivative_beta_term1.SetZero();

	Vector temp1;
	temp1.Init(num_of_betas_);

	//Matrix matrix_temp1;
	//matrix_temp1.Init(num_of_betas_, 1);

	//Matrix tmatrix_temp1;
	//tmatrix_temp1.Init(num_of_betas_, 1);


	Matrix temp2;
	temp2.Init(num_of_betas_, num_of_betas_);

	//Matrix temp3;
	//temp3.Init(num_of_betas_, first_stage_x_[n].n_cols());

	Matrix temp4;
	temp4.Init(num_of_betas_, num_of_betas_);

  for(index_t n=0; n<first_stage_x_.size(); n++) {

		Matrix temp3;		
		temp3.Init(num_of_betas_, first_stage_x_[n].n_cols());

		


    if (first_stage_y_[n]<0) { 
			//first_stage_y_[n]=-1 if all==zero, j_i is n chose j_i
      continue;
    } else {
			

			//check from here
      //Vector temp1;
			//temp1.Init(betas.length());
			la::MulOverwrite(first_stage_x_[n], first_stage_dot_logit_[n], &temp1);


			Matrix matrix_temp1;
			matrix_temp1.Alias(temp1.ptr(), temp1.length(), 1);
			Matrix tmatrix_temp1;
			tmatrix_temp1.Alias(temp1.ptr(), 1, temp1.length());

			

			//Matrix temp2
			//temp2.Init(betas.length(), betas.length());

			//la::MulTransBOverwrite(temp1, temp1, &temp2);
			la::MulOverwrite(matrix_temp1, tmatrix_temp1, &temp2);

			//Matrix temp3;
			//temp3.Init(betas.length(), first_stage_x_[n].n_cols());
			la::MulOverwrite(first_stage_x_[n], first_stage_ddot_logit_[n], &temp3);

			//Matrix temp4;
			//temp3.Init(betas.length(), betas.length());
			la::MulTransBOverwrite(temp3, first_stage_x_[n], &temp4);
			//check
			la::SubFrom(temp4, &temp2);
			la::AddTo(temp2, &second_derivative_beta_term1);

			//matrix_temp1.Destruct();
			//tmatrix_temp1.Destruct();

			//temp3.Destruct();
																							
		}
  }
  //return second_derivative_beta_term1;
	/*
	cout<<"second_derivative_beta_term1:"<<endl;
	for(index_t i=0; i<second_derivative_beta_term1.n_rows(); i++){
		for(index_t j=0; j<second_derivative_beta_term1.n_cols(); j++){
			cout<<second_derivative_beta_term1.get(i,j)<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
	*/

	second_beta_term1->Copy(second_derivative_beta_term1);


}



//////////////////////////////////////////////////
///Finite differene derivative approximation
//////////////////////////////////////////////////
void MLObjective::CheckGradient(double current_sample,
															Vector &current_parameter, 
									 Vector *approx_gradient) {
										 

	//approx_gradient(i)=(f(x+epsilon*ei)-f(x-epsilon*ei))/(2*epsilon)
	Vector e;
	e.Init(num_of_betas_+2);

	Vector u1; //x+epsillon*ei
	u1.Init(num_of_betas_+2);
	u1.SetZero();

	Vector u2; //x-epsillon*ei
	u2.Init(num_of_betas_+2);
  u2.SetZero();

	double epsilon;
	epsilon=sqrt(1e-8);

	double u1_objective=0;
	double u2_objective=0;

	Vector dummy_approx_gradient;
	dummy_approx_gradient.Init(num_of_betas_+2);
	dummy_approx_gradient.SetZero();

	for(index_t i=0; i<(num_of_betas_+2); i++){
		e.SetZero();
		e[i]=1.0;
		la::Scale(epsilon, &e);
		/*
		cout<<"e="<<endl;
		for(index_t i=0; i<e.length(); i++){
			cout<<e[i]<<" ";
		}
		cout<<endl;
    */

		la::AddOverwrite(current_parameter, e, &u1);
		la::SubOverwrite(e, current_parameter, &u2);
		
		/*
		cout<<"u1="<<endl;
		for(index_t i=0; i<e.length(); i++){
			cout<<u1[i]<<" ";
		}
		cout<<endl;

		cout<<"u2="<<endl;
		for(index_t i=0; i<e.length(); i++){
			cout<<u2[i]<<" ";
		}
		cout<<endl;
    */
		

		ComputeObjective(current_sample, u1, &u1_objective);
    
		
		//cout<<"u1_objective="<<u1_objective<<endl;
		ComputeObjective(current_sample, u2, &u2_objective);

		dummy_approx_gradient[i]=(u1_objective-u2_objective)/(2*epsilon);
		//cout<<"dummy_approx_gradient[i]="<<dummy_approx_gradient[i]<<endl;
	  
	}

	approx_gradient->Copy(dummy_approx_gradient);


}



	


void MLObjective::CheckHessian(double current_sample,
														 Vector &current_parameter, 
														 Matrix *approx_hessian){


//approx_hessian(i,j)=(gradi(x+epsilon*ej)-gradi(x-epsilon*ej))/(4*epsilon)
//									+(gradj(x+epsilon*ei)-gradj(x-epsilon*ei))/(4*epsilon)
//OR


	Vector ei;
	ei.Init(num_of_betas_);

	Vector ej;
	ej.Init(num_of_betas_);


	Vector u1i; //x+epsillon*ei
	u1i.Init(num_of_betas_);
	u1i.SetZero();

	Vector u2i; //x-epsillon*ei
	u2i.Init(num_of_betas_);
  u2i.SetZero();

	Vector u1j; //x+epsillon*ej
	u1j.Init(num_of_betas_);
	u1j.SetZero();

	Vector u2j; //x-epsillon*ej
	u2j.Init(num_of_betas_);
  u2j.SetZero();

	double epsilon;
	//epsilon=sqrt(1e-8);
	epsilon=1e-4;

	//Vector u1i_gradient;
  //u1i_gradient.Init(num_of_betas_+2);
  //u1i_gradient.SetZero();

	//Vector u1j_gradient;
  //u1j_gradient.Init(num_of_betas_+2);
  //u1j_gradient.SetZero();

	//Vector u2i_gradient;
  //u2i_gradient.Init(num_of_betas_+2);
  //u2i_gradient.SetZero();

	//Vector u2j_gradient;
  //u2j_gradient.Init(num_of_betas_+2);
  //u2j_gradient.SetZero();

	/*
	Vector diff_gradient_i;
  diff_gradient_i.Init(num_of_betas_+2);
  diff_gradient_i.SetZero();

	Vector diff_gradient_j;
  diff_gradient_j.Init(num_of_betas_+2);
  diff_gradient_j.SetZero();
  */
	double diff_gradient_i=0;
	double diff_gradient_j=0;

	

	Matrix dummy_approx_hessian;
	dummy_approx_hessian.Init(num_of_betas_, num_of_betas_);
	dummy_approx_hessian.SetZero();

	for(index_t i=0; i<(num_of_betas_); i++){
		ei.SetZero();
		ei[i]=1.0;
		la::Scale(epsilon, &ei);

		la::AddOverwrite(current_parameter, ei, &u1i);
		la::SubOverwrite(ei, current_parameter, &u2i);

		Vector u1i_gradient;
		Vector u2i_gradient;
		ComputeGradient(current_sample, u1i, &u1i_gradient);
		ComputeGradient(current_sample, u2i, &u2i_gradient);

		//la::Scale(-1.0, &u1i_gradient);
		//la::Scale(-1.0, &u2i_gradient);


		//la::SubOverwrite(u2i_gradient, u1i_gradient, &diff_gradient_i);
		//la::Scale((1.0/(4*epsilon)), &diff_gradient_i);

		//dummy_approx_gradient[i]=(u1_objective-u2_objective)/(2*epsilon);
		

		for(index_t j=0; j<(num_of_betas_); j++){
			ej.SetZero();
			ej[j]=1.0;
			la::Scale(epsilon, &ej);

			la::AddOverwrite(current_parameter, ej, &u1j);
		  la::SubOverwrite(ej, current_parameter, &u2j);


      Vector u1j_gradient;
			Vector u2j_gradient;
			ComputeGradient(current_sample, u1j, &u1j_gradient);		
		  ComputeGradient(current_sample, u2j, &u2j_gradient);
			//la::Scale(-1.0, &u1j_gradient);
		  //la::Scale(-1.0, &u2j_gradient);

			//la::SubOverwrite(u2j_gradient, u1j_gradient, &diff_gradient_j);
		  //la::Scale((1.0/(4*epsilon)), &diff_gradient_j);

			diff_gradient_i=(u1j_gradient[i]-u2j_gradient[i])/(4*epsilon);
			diff_gradient_j=(u1i_gradient[j]-u2i_gradient[j])/(4*epsilon);

			double temp=0;
			temp=-1*(diff_gradient_i+diff_gradient_j);

			dummy_approx_hessian.set(i,j,temp);
		}
	}

	/*
	cout<<"dummy_hessian_nonsymmetric"<<endl;
	for (index_t j=0; j<dummy_approx_hessian.n_rows(); j++){
		for (index_t k=0; k<dummy_approx_hessian.n_cols(); k++){
			cout<<dummy_approx_hessian.get(j,k) <<"  ";
		}
		cout<<endl;
	}
  */


	//Make hessian symmetric
	for(index_t i=0; i<(num_of_betas_); i++){
		for(index_t j=0; j<(num_of_betas_); j++){

			double temp2=0;
			temp2=(dummy_approx_hessian.get(i,j)+dummy_approx_hessian.get(j,i))/(2.0);
			dummy_approx_hessian.set(i,j, temp2);
			dummy_approx_hessian.set(j,i, temp2);
		}
	}

	//Check positive definiteness
	Vector eigen_hessian;
	la::EigenvaluesInit(dummy_approx_hessian, &eigen_hessian);

	
	cout<<"eigen values"<<endl;

	for(index_t i=0; i<eigen_hessian.length(); i++){
		cout<<eigen_hessian[i]<<" ";
	}
	cout<<endl;

	double max_eigen=0;
	//cout<<"eigen_value:"<<endl;
	for(index_t i=0; i<eigen_hessian.length(); i++){
		//cout<<eigen_hessian[i]<<" ";
		if(eigen_hessian[i]>max_eigen){
			max_eigen=eigen_hessian[i];
		}

	}
	//cout<<endl;
	cout<<"max_eigen="<<(max_eigen)<<endl;

	if(max_eigen>0){
		NOTIFY("Hessian is not Negative definite..Modify...");
		for(index_t i=0; i<eigen_hessian.length(); i++){
			//dummy_approx_hessian.set(i,i,(dummy_approx_hessian.get(i,i)-(max_eigen+0.5)));
			dummy_approx_hessian.set(i,i,(dummy_approx_hessian.get(i,i)-(max_eigen*1.1)));
		}
	}

	Vector eigen_hessian2;
	Matrix eigenvec_hessian;

    
	la::EigenvectorsInit(dummy_approx_hessian, &eigen_hessian2, 
																				&eigenvec_hessian);

	cout<<"eigen values of updated hessian"<<endl;

	for(index_t i=0; i<eigen_hessian2.length(); i++){
		cout<<eigen_hessian2[i]<<" ";
	}
	cout<<endl;
	cout<<endl;

	cout<<"eigen vectors of updated hessian"<<endl;
  for (index_t j=0; j<dummy_approx_hessian.n_rows(); j++){
		for (index_t k=0; k<dummy_approx_hessian.n_cols(); k++){
			cout<<eigenvec_hessian.get(j,k) <<"  ";
		}
		cout<<endl;
	}
	cout<<endl;


	
	approx_hessian->Copy(dummy_approx_hessian);

}

void MLObjective::CheckHessian3(double current_sample,
														 Vector &current_parameter, 
														 Matrix *approx_hessian){


	

	Matrix dummy_approx_hessian;
	dummy_approx_hessian.Init(num_of_betas_, num_of_betas_);
	dummy_approx_hessian.SetZero();

	for(index_t i=0; i<(num_of_betas_); i++){
		dummy_approx_hessian.set(i,i,-1);
	}

	cout<<"hessian0(1,1)="<<dummy_approx_hessian.get(1,1)<<endl;

/*
	cout<<"hessian0"<<endl;
	for (index_t j=0; j<dummy_approx_hessian.n_rows(); j++){
		for (index_t k=0; k<dummy_approx_hessian.n_cols(); k++){
			cout<<dummy_approx_hessian.get(j,k) <<"  ";
		}
		cout<<endl;
	}
*/
	approx_hessian->Copy(dummy_approx_hessian);

}
 
void MLObjective::CheckHessian2(double current_sample, 
									  Vector &current_parameter, 
										Matrix *approx_hessian) {

//(i,j)=(f(x+epsilon*ei+epsilon*ej)-f(x+epsilon*ei)-f(x+epsilon*ej)+f(x))/(epsilon^2)
//(i,j)=(f(x+epsilon*ei+epsilon*ej)-f(x+epsilon*ei-epsilon*ej)
//															 -f(x-epsilon*ei+epsilon*ej)+f(x-epsilon*ei-epsilon*ej))/(4*epsilon^2)

	Vector ei;
	ei.Init(num_of_betas_);

	Vector ej;
	ej.Init(num_of_betas_);


	Vector u1i; //x+epsillon*ei
	u1i.Init(num_of_betas_);
	u1i.SetZero();

	Vector u1j; //x+epsillon*ej
	u1j.Init(num_of_betas_);
	u1j.SetZero();

	Vector uij; //x+epsillon*ei+epsilon*ej
	uij.Init(num_of_betas_);
  uij.SetZero();

	double epsilon;
	//epsilon=sqrt(1e-8);
	epsilon=1e-4;

		
	Matrix dummy_approx_hessian;
	dummy_approx_hessian.Init(num_of_betas_, num_of_betas_);
	dummy_approx_hessian.SetZero();

	for(index_t i=0; i<(num_of_betas_); i++){
		ei.SetZero();
		ei[i]=1.0;
		la::Scale(epsilon, &ei);

		la::AddOverwrite(current_parameter, ei, &u1i);
				
		double first_term; //f(x+epsilon*ei+epsilon*ej)
		double second_term; //-f(x+epsilon*ei)
		double third_term; //-f(x+epsilon*ej)
		double fourth_term; //+f(x))

		ComputeObjective(current_sample, u1i, &second_term);
		//second_term=-1*second_term;
		ComputeObjective(current_sample, current_parameter, &fourth_term);
		fourth_term=-1*fourth_term;
		
		for(index_t j=0; j<(num_of_betas_); j++){
			ej.SetZero();
			ej[i]=1.0;
			la::Scale(epsilon, &ej);

			la::AddOverwrite(current_parameter, ej, &u1j);
		  la::AddOverwrite(u1i, ej, &uij);

   		ComputeObjective(current_sample, u1j, &third_term);
			//third_term=-1.0*third_term;
			ComputeObjective(current_sample, uij, &first_term);
			first_term=-1.0*first_term;

			double temp=0;
			temp=(first_term+second_term+third_term+fourth_term)/(epsilon*epsilon);

			dummy_approx_hessian.set(i,j,temp);
		}
	}

	//Make hessian symmetric
	for(index_t i=0; i<(num_of_betas_); i++){
		for(index_t j=0; j<(num_of_betas_); j++){

			double temp2=0;
			temp2=(dummy_approx_hessian.get(i,j)+dummy_approx_hessian.get(j,i))/(2.0);
			dummy_approx_hessian.set(i,j, temp2);
			dummy_approx_hessian.set(j,i, temp2);
		}
	}

	cout<<"dummy_hessian2"<<endl;
	for (index_t j=0; j<dummy_approx_hessian.n_rows(); j++){
		for (index_t k=0; k<dummy_approx_hessian.n_cols(); k++){
			cout<<dummy_approx_hessian.get(j,k) <<"  ";
		}
		cout<<endl;
	}

	//Check positive definiteness
	Vector eigen_hessian;
	la::EigenvaluesInit (dummy_approx_hessian, &eigen_hessian);

	cout<<"eigen values"<<endl;

	for(index_t i=0; i<eigen_hessian.length(); i++){
		cout<<eigen_hessian[i]<<" ";
	}
	cout<<endl;

	double max_eigen=0;
	//cout<<"eigen_value:"<<endl;
	for(index_t i=0; i<eigen_hessian.length(); i++){
		//cout<<eigen_hessian[i]<<" ";
		if(eigen_hessian[i]>max_eigen){
			max_eigen=eigen_hessian[i];
		}

	}
	//cout<<endl;
	cout<<"max_eigen="<<(max_eigen)<<endl;

	if(max_eigen>0){
		NOTIFY("Hessian is not Negative definite..Modify...");
		for(index_t i=0; i<eigen_hessian.length(); i++){
			dummy_approx_hessian.set(i,i,(dummy_approx_hessian.get(i,i)-max_eigen*(1.01)));
		}
	}

	
	approx_hessian->Copy(dummy_approx_hessian);




}






void MLObjective::ComputePredictionError(double current_sample, 
									  Vector &current_parameter,
										ArrayList<index_t> &true_decision,
										double *postponed_prediction_error,
										double *choice_prediction_error){



	Vector betas;
  betas.Alias(current_parameter.ptr(), num_of_betas_);
  
	ComputeExpBetasTimesX1_(betas);

  
	//Vector predicted_postponed_probability;
  //predicted_postponed_probability.Init(true_decision.length());
	//predicted_postponed_probability=postponed_probability_[n]

	index_t number_of_test=true_decision.size();

	ArrayList<Vector> predicted_choice_probability_all;
  predicted_choice_probability_all.Init(number_of_test);
	for(index_t n=0; n<predicted_choice_probability_all.size(); n++){
		predicted_choice_probability_all[n].Init(first_stage_x_[n].n_cols());
    predicted_choice_probability_all[n].SetZero();
	}

	Vector predicted_choice_probability;
  predicted_choice_probability.Init(number_of_test);
	predicted_choice_probability.SetZero();


  //predicted_choice_probability.SetZero();
	//predicted_choice_probability.Init(first_stage_x_.size());

	Vector predicted_decision;
  predicted_decision.Init(number_of_test);
  predicted_decision.SetAll(-1.0);

	for(index_t n=0; n<number_of_test; n++){
		double max_choice_probability=0;
		index_t index_chosen_alternative=-100;
		for(index_t i=0; i<first_stage_x_[n].n_cols(); i++){
				Vector temp;
				first_stage_x_[n].MakeColumnVector(i, &temp);
				predicted_choice_probability_all[n][i]=
					exp(la::Dot(betas, temp))/exp_betas_times_x1_[n];
			
			//find the alternative with maximum choice_prob.
			for(index_t i=0; i<first_stage_x_[n].n_cols(); i++){
				if(max_choice_probability<predicted_choice_probability_all[n][i]){
						index_chosen_alternative=i;
						//cout<<"n="<<n<<"index_chosen_alternative"<<index_chosen_alternative<<endl;
						max_choice_probability=predicted_choice_probability_all[n][i];
				}	//if
			}	//for


		}
		predicted_decision[n]=index_chosen_alternative+1;
		predicted_choice_probability[n]=max_choice_probability;
		
	}	//for n

	Matrix mtx_predicted_decision;
  mtx_predicted_decision.Init(number_of_test,1);

	for(index_t n=0; n<number_of_test; n++){
		mtx_predicted_decision.set(n,0, predicted_decision[n]);
	}

  Matrix mtx_predicted_choice_probability;
  mtx_predicted_choice_probability.Init(number_of_test,1);

	for(index_t n=0; n<number_of_test; n++){
		mtx_predicted_choice_probability.set(n,0, predicted_choice_probability[n]);
	}


	cout<<"predicted probability..."<<endl;
	data::Save("ML_predicted_choice_probability.csv",  mtx_predicted_choice_probability);

	cout<<"Save predicted decision..."<<endl;
	data::Save("ML_predicted_decision.csv", mtx_predicted_decision);



	//error calculation
	double temp_postponed_prediction_error=0;
	double temp_choice_prediction_error=0;

	
	double count_correct_postponed=0;
	double count_correct_choice=0;

	for(index_t n=0; n<number_of_test; n++){
		//if(true_decision[n]==-1 &&predicted_decision[n]==-1){
	  
		//if(true_decision[n]*predicted_decision[n]>0){
		//	count_correct_postponed=count_correct_postponed+1;
		//	//temp_postponed_prediction_error_prob;

		//}
		if((true_decision[n]==predicted_decision[n])){
			count_correct_choice=count_correct_choice+1;
		}
	}

	temp_postponed_prediction_error=
			(number_of_test-count_correct_postponed)/number_of_test*100;
	temp_choice_prediction_error=
		  (number_of_test-count_correct_choice)/number_of_test*100;
			//(count_correct_choice)/number_of_test*100;

	cout<<"postponed_prediction_error="<<temp_postponed_prediction_error<<endl;
	cout<<"choice_prediction_error="<<temp_choice_prediction_error<<endl;
  cout<<"num_correct_choice_prediction="<<count_correct_choice<<endl;
  
	//error probability calculation
	Vector predicted_prob_of_n_choose_i;
	predicted_prob_of_n_choose_i.Init(number_of_test);
	predicted_prob_of_n_choose_i.SetZero();
  
	double count_true_postponed=0;
	double count_predicted_postponed=0;

	for(index_t n=0; n<number_of_test; n++){
		if(true_decision[n]<0) {
			count_true_postponed+=1;
			predicted_prob_of_n_choose_i[n]=0;
			if(predicted_decision[n]<0){
				count_predicted_postponed+=1;
			}

		}	//if
		else {
			predicted_prob_of_n_choose_i[n]=predicted_choice_probability_all[n][true_decision[n]-1];
		}
	}
	//cout<<"Test2"<<endl;
	cout<<"count_true_postponed="<<count_true_postponed<<endl;
	cout<<"Percent_true_postponed="<<count_true_postponed/number_of_test*100<<endl;
  cout<<"count_predicted_postponed="<<count_predicted_postponed<<endl;
  cout<<"(all)Percent_count_predicted_postponed="<<count_predicted_postponed/number_of_test*100<<endl;
  cout<<"(postponed)Percent_count_predicted_postponed="<<count_predicted_postponed/count_true_postponed*100<<endl;

	cout<<endl;

	//double temp_postponed_prediction_error_prob=0;
	double temp_choice_prediction_error_prob=0;
	double temp_postpone_prediction_error_prob=0;
	for(index_t n=0; n<number_of_test; n++){
		//temp_choice_prediction_error_prob+=(1-predicted_prob_of_n_choose_i[n]);
		temp_choice_prediction_error_prob+=(1-0);
		//temp_postpone_prediction_error_prob+=(1-postponed_probability_[n]);
	}
	temp_choice_prediction_error_prob*=2;
	temp_postpone_prediction_error_prob*=2;

	cout<<"choice_prediction_error_prob_rate="<<temp_choice_prediction_error_prob/number_of_test<<endl;
	cout<<"postpone_prediction_error_prob_rate="<<temp_postpone_prediction_error_prob/number_of_test<<endl;
	(*postponed_prediction_error)=temp_postponed_prediction_error;
	(*choice_prediction_error)=temp_choice_prediction_error;

	
	//cout<<"predicted_decision"<<endl;
	//for(index_t n=0; n<number_of_test; n++){
	//	cout<<predicted_decision[n]<<" ";
	//}
	//cout<<endl;
	






}













