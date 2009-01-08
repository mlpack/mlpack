#include "objective2.h"
#include <cmath>
#include <iostream>
    
using namespace std;

void Objective::Init(fx_module *module) {
  module_=module;
  const char *data_file1=fx_param_str_req(module_, "data1");
  const char *info_file1=fx_param_str_req(module_, "info1");
  Matrix x;
  data::Load(data_file1, &x);
  num_of_betas_=x.n_rows();
  Matrix info1;
  data::Load(info_file1, &info1);
  first_stage_x_.Init(info1.n_cols());
  index_t start_col=0;
  for(index_t i=0; i<info1.n_cols(); i++) {
    first_stage_x_[i].Init(x.n_rows(), (index_t)info1.get(0, i));
    first_stage_x_[i].CopyColumnFromMat(0, start_col, 
        (index_t)info1.get(0,i), x);
    start_col+=(index_t)info1.get(0, i);
  }
	
  const char *data_file2=fx_param_str_req(module_, "data2");
  //const char *info_file2=fx_param_str_req(module_, "info2");
	//info1==info2
  x.Destruct();
	//x.get(1,1);
  data::Load(data_file2, &x);
  //Matrix info2;
  //data::Load(info_file2, &info2);
  second_stage_x_.Init(info1.n_cols());
  start_col=0;
  for(index_t i=0; i<info1.n_cols(); i++) {
    second_stage_x_[i].Init(x.n_rows(), (index_t)info1.get(0,i));
	  //second_stage_x_[i].Init(first_stage_x_[0].n_rows(), (index_t)info2.get(0,i));
	
		second_stage_x_[i].CopyColumnFromMat(0, 
        start_col, 
        (index_t)info1.get(0,i),
        x);
    start_col+=(index_t)info1.get(0,i);
  }

	const char *data_file3=fx_param_str_req(module_, "data3");
  //const char *info_file3=fx_param_str_req(module_, "info3");
	//info1==info2==info3
  x.Destruct();
  data::Load(data_file3, &x);
  //Matrix info3;
  //data::Load(info_file3, &info3);
  unknown_x_past_.Init(info1.n_cols());
  start_col=0;
  for(index_t i=0; i<info1.n_cols(); i++) {
    unknown_x_past_[i].Init(x.n_rows(), (index_t)info1.get(0,i));
    unknown_x_past_[i].CopyColumnFromMat(0, start_col, 
        (index_t)info1.get(0,i), x);
    start_col+=(index_t)info1.get(0, i);
  }
	

	index_t num_selected_people=first_stage_x_.size();
	

	
	//Initilize memeber variables
	first_stage_y_.Init(num_selected_people);
	first_stage_y_[0]=-1;
	first_stage_y_[1]=2;

	second_stage_y_.Init(num_selected_people);
	second_stage_y_[0]=1;
	second_stage_y_[1]=-1;

	ind_unknown_x_.Init(1);
	ind_unknown_x_[0]=3;

	exp_betas_times_x1_.Init(num_selected_people);
  exp_betas_times_x2_.Init(num_selected_people);
	postponed_probability_.Init(num_selected_people);
	for(index_t i=0; i<postponed_probability_.size(); i++) {
    postponed_probability_[i]=0;
  }

	denumerator_beta_function_=0;
	num_of_t_beta_fn_=100;
	t_weight_=0;
	num_of_alphas_=100;
	alpha_weight_=0;  

	//from here for the gradient
	first_stage_dot_logit_.Init(num_selected_people);
	first_stage_ddot_logit_.Init(num_selected_people, num_selected_people);

	for(index_t i=0; i<first_stage_dot_logit_.size(); i++) {
    exp_betas_times_x1_[i]=0;
		first_stage_dot_logit_[i].Init(first_stage_x_[i].n_cols());
		first_stage_dot_logit_[i].SetZero();
		first_stage_ddot_logit_[i].Init(first_stage_x_[i].n_cols(),first_stage_x_[i].n_cols());
		first_stage_ddot_logit_[i].SetZero();

  }

	second_stage_dot_logit_.Init(num_selected_people);
	second_stage_ddot_logit_.Init(num_selected_people, num_selected_people);

	for(index_t i=0; i<second_stage_dot_logit_.size(); i++) {
    exp_betas_times_x2_[i]=0;
		second_stage_dot_logit_[i].Init(first_stage_x_[i].n_cols());
		second_stage_dot_logit_[i].SetZero();
		second_stage_ddot_logit_[i].Init(first_stage_x_[i].n_cols(),first_stage_x_[i].n_cols());
		second_stage_ddot_logit_[i].SetZero();

  }

	sum_first_derivative_conditional_postpond_prob_.Init(num_selected_people);
	sum_second_derivative_conditional_postpond_prob_.Init(num_selected_people);

	for(index_t n=0; n<first_stage_x_.size(); n++) {
		sum_first_derivative_conditional_postpond_prob_[n].Init(num_of_betas_);
		sum_first_derivative_conditional_postpond_prob_[n].SetZero();
		sum_second_derivative_conditional_postpond_prob_[n].Init(num_of_betas_, num_of_betas_);
		sum_second_derivative_conditional_postpond_prob_[n].SetZero();

	}

	sum_first_derivative_p_beta_fn_.Init(num_selected_people);
	sum_second_derivative_p_beta_fn_.Init(num_selected_people);
	sum_first_derivative_q_beta_fn_.Init(num_selected_people);
	sum_second_derivative_q_beta_fn_.Init(num_selected_people);
	sum_second_derivative_p_q_beta_fn_.Init(num_selected_people);
	sum_second_derivative_conditionl_postponed_p_.Init(num_selected_people);
	sum_second_derivative_conditionl_postponed_q_.Init(num_selected_people);


	for(index_t i=0; i<first_stage_x_.size(); i++) {
		sum_first_derivative_p_beta_fn_[i]=0;
		sum_second_derivative_p_beta_fn_[i]=0;
		sum_first_derivative_q_beta_fn_[i]=0;
		sum_second_derivative_q_beta_fn_[i]=0;
		sum_second_derivative_p_q_beta_fn_[i]=0;
		sum_second_derivative_conditionl_postponed_p_[i].Init(num_of_betas_);
		sum_second_derivative_conditionl_postponed_p_[i].SetZero();
		sum_second_derivative_conditionl_postponed_q_[i].Init(num_of_betas_);
		sum_second_derivative_conditionl_postponed_q_[i].SetZero();

	}

	
	





	
}

//void Objective::ComputeObjective(Matrix &x, double *objective) {
void Objective::ComputeObjective(Vector &current_parameter, double *objective) { 
	
	Vector betas;
  //betas.Alias(x.ptr(), x.n_rows());
	betas.Alias(current_parameter.ptr(), num_of_betas_);
  //double p=first_stage_x_[1].get(0, 0);
  //double q=first_stage_x_[1].get(0, 1);
  double p=current_parameter[num_of_betas_];
	double q=current_parameter[num_of_betas_+1];
	
	ComputeExpBetasTimesX1_(betas);
	
  ComputeDeumeratorBetaFunction_(p, q);
	
  ComputePostponedProbability_(betas, 
                               p, 
                               q);

	

  *objective = ComputeTerm1_(betas) 
               + ComputeTerm2_()
               + ComputeTerm3_();


	//*objective=2;

	
	
}

////////////////////////////////////////////////
////Calculate gradient
////////////////////////////////////////////////

void Objective::ComputeGradient(Vector &current_parameter, Vector *gradient) { 
	
	Vector betas;
  //betas.Alias(x.ptr(), x.n_rows());
	betas.Alias(current_parameter.ptr(), num_of_betas_);
  //double p=first_stage_x_[1].get(0, 0);
  //double q=first_stage_x_[1].get(0, 1);
  double p=current_parameter[num_of_betas_];
	double q=current_parameter[num_of_betas_+1];
	
	ComputeExpBetasTimesX1_(betas);
	
  ComputeDeumeratorBetaFunction_(p, q);
	
  ComputePostponedProbability_(betas, 
                               p, 
                               q);

	
	/*
  *objective = ComputeTerm1_(betas) 
               + ComputeTerm2_() 
               + ComputeTerm3_();

	*/
	
	ComputeDotLogit_(betas);
	ComputeDDotLogit_();
	//cout<<"ddot done"<<endl;
	ComputeSumDerivativeConditionalPostpondProb_(betas, p, q);
	//cout<<"sumDerivativeCondPostpondprob done"<<endl;
	

	Vector dummy_beta_term1;
	Vector dummy_beta_term2;
	Vector dummy_beta_term3;
	

	ComputeDerivativeBetaTerm1_(&dummy_beta_term1);
	ComputeDerivativeBetaTerm2_(&dummy_beta_term2);
	ComputeDerivativeBetaTerm3_(&dummy_beta_term3);
	//cout<<"DerivativeBetaTerm3 done"<<endl;
	

	
	ComputeSumDerivativeBetaFunction_(betas, p, q);
	//cout<<"SumDerivativeBetaFunction done"<<endl;

  Vector dummy_gradient;
	dummy_gradient.Init(num_of_betas_+2);
	dummy_gradient.SetZero();

	for(index_t i=0; i<num_of_betas_; i++){
		dummy_gradient[i]=dummy_beta_term1[i]+dummy_beta_term2[i]+dummy_beta_term3[i];
	}

	dummy_gradient[num_of_betas_]=ComputeDerivativePTerm1_()
																+ComputeDerivativePTerm2_()
																+ComputeDerivativePTerm3_();

	dummy_gradient[num_of_betas_+1]=ComputeDerivativeQTerm1_()
																+ComputeDerivativeQTerm2_()
																+ComputeDerivativeQTerm3_();


	gradient->Copy(dummy_gradient);
	
												
}



////////////////////////////////////////////////
////Calculate hessian
////////////////////////////////////////////////

void Objective::ComputeHessian(Vector &current_parameter, Matrix *hessian) { 
	
	Vector betas;
  //betas.Alias(x.ptr(), x.n_rows());
	betas.Alias(current_parameter.ptr(), num_of_betas_);
  //double p=first_stage_x_[1].get(0, 0);
  //double q=first_stage_x_[1].get(0, 1);
  double p=current_parameter[num_of_betas_];
	double q=current_parameter[num_of_betas_+1];
	
	
	ComputeExpBetasTimesX1_(betas);
	
  ComputeDeumeratorBetaFunction_(p, q);
	
  ComputePostponedProbability_(betas, 
                               p, 
                               q);

	
	/*
  *objective = ComputeTerm1_(betas) 
               + ComputeTerm2_() 
               + ComputeTerm3_();

	*/
	
	ComputeDotLogit_(betas);
	ComputeDDotLogit_();
	//cout<<"ddot done"<<endl;
	ComputeSumDerivativeConditionalPostpondProb_(betas, p, q);
	//cout<<"sumDerivativeCondPostpondprob done"<<endl;
	ComputeSumDerivativeBetaFunction_(betas, p, q);
	//cout<<"SumDerivativeBetaFunction done"<<endl;
	

	Matrix dummy_second_beta_term1;
	Matrix dummy_second_beta_term2;
	Matrix dummy_second_beta_term3;
	

	ComputeSecondDerivativeBetaTerm1_(&dummy_second_beta_term1);
	ComputeSecondDerivativeBetaTerm2_(&dummy_second_beta_term2);
	ComputeSecondDerivativeBetaTerm3_(&dummy_second_beta_term3);
	

	Vector dummy_p_beta_term1;
	Vector dummy_p_beta_term2;
	Vector dummy_p_beta_term3;
	Vector dummy_q_beta_term1;
	Vector dummy_q_beta_term2;
	Vector dummy_q_beta_term3;


	
	ComputeSecondDerivativePBetaTerm1_(&dummy_p_beta_term1);
	ComputeSecondDerivativePBetaTerm2_(&dummy_p_beta_term2);
	ComputeSecondDerivativePBetaTerm3_(&dummy_p_beta_term3);

	ComputeSecondDerivativeQBetaTerm1_(&dummy_q_beta_term1);
	ComputeSecondDerivativeQBetaTerm2_(&dummy_q_beta_term2);
	ComputeSecondDerivativeQBetaTerm3_(&dummy_q_beta_term3);
	//cout<<"SecondDerivativeQBetaTerm3_ done"<<endl;

  Matrix dummy_hessian;
	dummy_hessian.Init(num_of_betas_+2, num_of_betas_+2);
	dummy_hessian.SetZero();

	for(index_t i=0; i<num_of_betas_+2; i++){
		for(index_t j=0; j<num_of_betas_+2; j++){

			if(i<num_of_betas_ && j<num_of_betas_){
				dummy_hessian.set(i,j, dummy_second_beta_term1.get(i,j)
															+dummy_second_beta_term2.get(i,j)
															+dummy_second_beta_term2.get(i,j));
				
			} else if(i<num_of_betas_ &&j>=num_of_betas_){
				
				dummy_hessian.set(i,num_of_betas_, dummy_p_beta_term1[i]
															+dummy_p_beta_term2[i]
															+dummy_p_beta_term2[i]);
				dummy_hessian.set(i,num_of_betas_+1, dummy_q_beta_term1[i]
															+dummy_q_beta_term2[i]
															+dummy_q_beta_term2[i]);
															
			}	else if(j<num_of_betas_ &&i>=num_of_betas_){
				dummy_hessian.set(num_of_betas_, j, dummy_p_beta_term1[j]
															+dummy_p_beta_term2[j]
															+dummy_p_beta_term2[j]);
				dummy_hessian.set(num_of_betas_+1, j, dummy_q_beta_term1[j]
															+dummy_q_beta_term2[j]
															+dummy_q_beta_term2[j]);
			} 

		}	//j
	}		//i

	

	dummy_hessian.set(num_of_betas_, num_of_betas_, 
													ComputeSecondDerivativePTerm1_()
													+ComputeSecondDerivativePTerm2_()
													+ComputeSecondDerivativePTerm3_());

	dummy_hessian.set(num_of_betas_+1, num_of_betas_+1, 
													ComputeSecondDerivativeQTerm1_()
													+ComputeSecondDerivativeQTerm2_()
													+ComputeSecondDerivativeQTerm3_());

	dummy_hessian.set(num_of_betas_+1, num_of_betas_, 
													ComputeSecondDerivativePQTerm1_()
													+ComputeSecondDerivativePQTerm2_()
													+ComputeSecondDerivativePQTerm3_());

	dummy_hessian.set(num_of_betas_, num_of_betas_+1, 
													ComputeSecondDerivativePQTerm1_()
													+ComputeSecondDerivativePQTerm2_()
													+ComputeSecondDerivativePQTerm3_());

	
	
	
	/*for (index_t j=0; j<dummy_hessian.n_rows(); j++){
		for (index_t k=0; k<dummy_hessian.n_cols(); k++){
				cout<<dummy_hessian.get(j,k) <<"  ";
		}
		cout<<endl;
	}
	*/
	hessian->Copy(dummy_hessian);
	//cout<<"hessian done"<<endl;
												
}

///////////////////////////////////////////
//////////////////////////////////////////////////////////////////

double Objective::ComputeTerm1_(Vector &betas) {
  double term1=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) { 
			//first_stage_y_[n]=-1 if all==zero, j_i is n chose j_i
      continue;
    } else {
      Vector temp;
      first_stage_x_[n].MakeColumnVector(first_stage_y_[n], &temp);
			term1+=la::Dot(betas, temp) - log(exp_betas_times_x1_[n]);
    }
  }
  return term1;
}

double Objective::ComputeTerm2_() {
  double term2=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) {
      continue;
    } else {
      DEBUG_ASSERT(1-postponed_probability_[n]);
      term2+=log(1-postponed_probability_[n]);
    }
  }
  return term2;
}

double Objective::ComputeTerm3_() {
  double term3=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (second_stage_y_[n]<0) {
      continue;
    } else {
      DEBUG_ASSERT(postponed_probability_[n]>0);
      term3+=log(postponed_probability_[n]);
    }
  }
  return term3;
}

//Compute x^2_{ni}(alpha), beta'x^2_{ni}(alpha), and postponedprob.
void Objective::ComputePostponedProbability_(Vector &betas, 
                                             double p, 
                                             double q) {
	//double numerator=0;
	//need to specify
	//num_of_alphas_=10;
	alpha_weight_=(double)1/num_of_alphas_;
	double exponential_temp=0;
	//double exp_betas_times_x2=0;
  /*for(index_t i=0; i<postponed_probability_.size(); i++) {
    postponed_probability_[i]=0;
  }
	*/
  
	for(index_t n=0; n<first_stage_x_.size(); n++){
		for(index_t l=0; l<num_of_alphas_-1; l++){
			double alpha_temp;
	    double beta_function_temp;
      alpha_temp=(l+1)*(alpha_weight_);
			beta_function_temp=pow(alpha_temp, p-1)*
          pow((1-alpha_temp), q-1)/denumerator_beta_function_;
		
			
			//Calculate x^2_{ni}(alpha_l)
			for(index_t i=0; i<first_stage_x_[n].n_cols(); i++){
				int count=0;
				for(index_t j=ind_unknown_x_[0]; j<=ind_unknown_x_[ind_unknown_x_.size()-1]; j++){
					exponential_temp=alpha_temp*first_stage_x_[n].get(j-1, i)
													+(alpha_temp)*(1-alpha_temp)*unknown_x_past_[n].get(count,0)
													+(alpha_temp)*pow((1-alpha_temp),2)*unknown_x_past_[n].get(count,1);
					second_stage_x_[n].set(j-1, i, exponential_temp);
					count+=first_stage_x_[n].n_cols();
				}	//j
			}	//i

		  for(index_t i=0; i<exp_betas_times_x2_.size(); i++) {
			  exp_betas_times_x2_[i]=0;
		  }

			for(index_t i=0; i<second_stage_x_[n].n_cols(); i++) {
				exp_betas_times_x2_[n]=0;
				exp_betas_times_x2_[n]+=exp(la::Dot(betas.length(), 
																betas.ptr(),
																second_stage_x_[n].GetColumnPtr(i)));			}
			//cout<<"exp_betas_times_x2_"<<exp_betas_times_x2_[n]<<endl;
			//conditional_postponed_probability_[n]
			postponed_probability_[n]=0;
			postponed_probability_[n]+=( (exp_betas_times_x2_[n]/(exp_betas_times_x1_[n]
																  + exp_betas_times_x2_[n]) )
																		*beta_function_temp );	
			//cout<<"beta_fn_temp "<<beta_function_temp<<endl;
			//cout<<"postpond_prob "<<postponed_probability_[n]<<endl;
			//cout<<"denumerator_beta_function_ "<<denumerator_beta_function_<<endl;
   
    }	//alpha


		postponed_probability_[n]*=alpha_weight_;	
	}	//n

}

void Objective::ComputeExpBetasTimesX1_(Vector &betas) {
  
  //double sum=0;
	for(index_t n=0; n<first_stage_x_.size(); n++){
		for(index_t j=0; j<first_stage_x_[n].n_cols(); j++) {
			exp_betas_times_x1_[n]=0;
			exp_betas_times_x1_[n]+=exp(la::Dot(betas.length(), 
															betas.ptr(), 
															first_stage_x_[n].GetColumnPtr(j)));
		}

		/*cout<<"first_stage_x="<<endl;
		for(index_t i=0; i<first_stage_x_.size(); i++){
			cout<<first_stage_x_[i].get(0,0)<<" ";
		}
		cout<<endl;
		*/

		/*cout<<"exp_betas_times_x1:"<<endl;
		for(index_t i=0; i<exp_betas_times_x1_.size(); i++){
			cout<<exp_betas_times_x1_[i]<<" ";
		}
		cout<<endl;
*/

  }
}


void Objective::ComputeDeumeratorBetaFunction_(double p, double q) {
	denumerator_beta_function_=0;
	//Need to choose number of t points to approximate integral
	//num_of_t_beta_fn_=10;
	t_weight_=(double)1/(num_of_t_beta_fn_);
	double t_temp;
	for(index_t tnum=0; tnum<num_of_t_beta_fn_-1; tnum++){
		t_temp=(tnum+1)*(t_weight_);
		
		//double pow( double base, double exp );
		denumerator_beta_function_+=pow(t_temp, p-1)*pow((1-t_temp), q-1);
	
	}
	denumerator_beta_function_*=(t_weight_);
	
}





//////////////////////////////////////////////////////////
//add new things from here for objective2 (Compute gradient) 
//Compute dot_logit
void Objective::ComputeDotLogit_(Vector &betas) {

	/*for(index_t n=0; n<first_stage_dot_logit_.size(); n++) {
		first_stage_dot_logit_[n].Init(first_stage_x_[n].n_cols());
    first_stage_dot_logit_[n].SetZero();
  }
	*/


	//cout<<"test "<<first_stage_dot_logit_[1][1]<<endl;
	for(index_t n=0; n<first_stage_x_.size(); n++){
		for(index_t i=0; i<first_stage_x_[n].n_cols(); i++){
			first_stage_dot_logit_[n][i]=(exp(la::Dot( betas.length(), betas.ptr(),
																				 first_stage_x_[n].GetColumnPtr(i) )))/
																				 exp_betas_times_x1_[n];
			//cout<<"test "<<first_stage_dot_logit_[n][i]<<endl;
		}	//i
	}	//n
	
}


void Objective::ComputeDDotLogit_() {
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

}


void Objective::ComputeDerivativeBetaTerm1_(Vector *beta_term1) {
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

			la::SubOverwrite(num_of_betas_, temp.ptr(), first_stage_x_[n].GetColumnPtr(first_stage_y_[n]), temp2.ptr());
			
																									
		}	//else
		la::AddTo(temp2, &temp3);
		
  }	//n
	//beta_term1=&temp3;
	beta_term1->Copy(temp3);
  //return derivative_beta_term1;
	

}



void Objective::ComputeSumDerivativeConditionalPostpondProb_(Vector &betas, double p, double q){
	
	Vector temp1;
	temp1.Init(betas.length());	//dotX1*dotLogit1
	//SumSecondDerivativeConditionalPostpondProb_.SetZero();

	double alpha_temp=0;
	double beta_function_temp=0;
	//double numerator=0;
	double exponential_temp=0;
	//need to specify
	//num_of_alphas_=10;
	alpha_weight_=(double)1/num_of_alphas_;


	double conditional_postponed_prob=0;
	Vector first_derivative_conditional_postpond_prob;
	first_derivative_conditional_postpond_prob.Init(betas.length());

	Matrix matrix_first_derivative_conditional_postpond_prob;
	//tmatrix_first_derivative_conditional_postpond_prob.Init(betas.length(),1);

	Matrix tmatrix_first_derivative_conditional_postpond_prob;
	//tmatrix_first_derivative_conditional_postpond_prob.Init(betas.length(),1);

	Matrix second_derivative_conditional_postpond_prob;
	second_derivative_conditional_postpond_prob.Init(betas.length(),betas.length() );



	Vector temp2;	//dotX2*dotLogit2
	temp2.Init(betas.length());

	Matrix first_term_temp;
	first_term_temp.Init(betas.length(), betas.length());

	

	Matrix temp10; //temp9*dotX1'
	temp10.Init(betas.length(), betas.length());

	Matrix second_term_temp;
	second_term_temp.Init(betas.length(), betas.length());

	Matrix matrix_first_stage_dot_logit;
	//matrix_first_stage_dot_logit.Init(first_stage_x_[n].n_cols(),1);

	Matrix tmatrix_first_stage_dot_logit;
	//tmatrix_first_stage_dot_logit.Init(first_stage_x_[n].n_cols(),1);
	Matrix matrix_second_stage_dot_logit;
	//matrix_second_stage_dot_logit.Init(second_stage_x_[n].n_cols(), 1);

	Matrix tmatrix_second_stage_dot_logit;
	//tmatrix_second_stage_dot_logit.Init(second_stage_x_[n].n_cols(), 1);



	for(index_t n=0; n<first_stage_x_.size(); n++){

		

		Matrix temp3; //dotLogit2*dotLogit2'
		temp3.Init(second_stage_x_[n].n_cols(), second_stage_x_[n].n_cols());
	
		Matrix temp4; //ddotLogit2-dotLogit2*dotLogit2'
		temp4.Init(second_stage_x_[n].n_cols(), second_stage_x_[n].n_cols());

		Matrix temp5;	//dotX2(temp4)
		temp5.Init(betas.length(), second_stage_x_[n].n_cols());

		Matrix temp6; //temp5*dotX2'
		temp6.Init(betas.length(), betas.length());

		//terms for the first stage
		Matrix temp7; //dotLogit1*dotLogit1
		temp7.Init(first_stage_x_[n].n_cols(), first_stage_x_[n].n_cols());

		Matrix temp8; //ddotLogit1-dotLogit1*dotLogit1'
		temp8.Init(first_stage_x_[n].n_cols(), first_stage_x_[n].n_cols());

		Matrix temp9;	//dotX1(temp8)
		temp9.Init(betas.length(), second_stage_x_[n].n_cols());

		

		la::MulOverwrite(first_stage_x_[n], first_stage_dot_logit_[n], &temp1);

		
		
		
		//matrix_first_stage_dot_logit.Init(first_stage_x_[n].n_cols(), 1);
		
		matrix_first_stage_dot_logit.Alias(first_stage_dot_logit_[n].ptr(), first_stage_dot_logit_[n].length(), 1);
		tmatrix_first_stage_dot_logit.Alias(first_stage_dot_logit_[n].ptr(), 1, first_stage_dot_logit_[n].length());
		
		//matrix_first_stage_dot_logit.CopyVectorToColumn(1, first_stage_dot_logit_[n]);
		//la::MulTransBOverwrite(first_stage_dot_logit_[n], first_stage_dot_logit_[n], &temp7);
		
		
		la::MulOverwrite(matrix_first_stage_dot_logit, 
														tmatrix_first_stage_dot_logit, &temp7);
		
		la::SubOverwrite(temp7, first_stage_ddot_logit_[n], &temp8);

		//la::MulOverwrite(first_stage_dot_logit_[n], temp8, &temp9);
		la::MulOverwrite(first_stage_x_[n], temp8, &temp9);
		
		//la::MulTransBOverwrite(temp9, first_stage_dot_logit_[n], &temp10);
		la::MulTransBOverwrite(temp9, first_stage_x_[n], &temp10);
		
		

		

		for(index_t l=0; l<num_of_alphas_-1; l++){
			alpha_temp=(l+1)*(alpha_weight_);
		
			beta_function_temp=pow(alpha_temp, p-1)*pow((1-alpha_temp), q-1)/denumerator_beta_function_;

			//Calculate x^2_{ni}(alpha_l)
			for(index_t i=0; i<first_stage_x_[n].n_cols(); i++){
				//cout<<"i="<<i<<endl;
				int count=0;
				for(index_t j=ind_unknown_x_[0]; j<=ind_unknown_x_[ind_unknown_x_.size()-1]; j++){
					//cout<<"j="<<j<<endl;
					
					exponential_temp=alpha_temp*first_stage_x_[n].get(j-1, i)
													+(alpha_temp)*(1-alpha_temp)*unknown_x_past_[n].get(count,0)
													+(alpha_temp)*pow((1-alpha_temp),2)*unknown_x_past_[n].get(count,1);
					second_stage_x_[n].set(j-1, i, exponential_temp);
					count+=first_stage_x_[n].n_cols();
				}	//j
			}	//i
			

			for(index_t i=0; i<second_stage_x_[n].n_cols(); i++) {
				exp_betas_times_x2_[n]+=exp(la::Dot(betas.length(), betas.ptr(),
											 second_stage_x_[n].GetColumnPtr(i) ));
			}	//i




			for(index_t i=0; i<second_stage_x_[n].n_cols(); i++) {
				//Calculate second_stage_dot_logit_
				second_stage_dot_logit_[n][i]=((exp(la::Dot(betas.length(), betas.ptr(),
																				 second_stage_x_[n].GetColumnPtr(i))))/
																				 exp_betas_times_x2_[n]);
																				 
				second_stage_ddot_logit_[n].set(i, i, first_stage_dot_logit_[n][i]);
			}	//i

		
			conditional_postponed_prob=exp_betas_times_x2_[n]/(exp_betas_times_x1_[n]+exp_betas_times_x2_[n]);
			la::MulOverwrite(second_stage_x_[n], second_stage_dot_logit_[n], &temp2);
			la::SubOverwrite(temp2, temp1, &first_derivative_conditional_postpond_prob);

			//Calculate SecondDerivativePostponedProb.
			//Matrix first_term_temp;
			//first_term_temp.Init(betas.length(), betas.length());

			//handle vector transpose				
			matrix_first_derivative_conditional_postpond_prob.Alias(first_derivative_conditional_postpond_prob.ptr(), 
																										first_derivative_conditional_postpond_prob.length(),
																										1);
			tmatrix_first_derivative_conditional_postpond_prob.Alias(first_derivative_conditional_postpond_prob.ptr(), 
																										1,
																										first_derivative_conditional_postpond_prob.length());



			la::MulOverwrite(matrix_first_derivative_conditional_postpond_prob, 
														 tmatrix_first_derivative_conditional_postpond_prob, 
														 &first_term_temp);
			la::Scale( (1-2*conditional_postponed_prob)*(conditional_postponed_prob)*(1-conditional_postponed_prob),
								&first_term_temp);

			

			//check
			//Matrix temp3; //dotLogit*dotLogit'
			//temp3.Init(second_stage_x_[n].n_cols(), second_stage_x_[n].n_cols());

			//Handle vector transpose
			//Matrix matrix_second_stage_dot_logit_;



			matrix_second_stage_dot_logit.Alias(second_stage_dot_logit_[n].ptr(), 
																					second_stage_dot_logit_[n].length(),
																					1);
			tmatrix_second_stage_dot_logit.Alias(second_stage_dot_logit_[n].ptr(), 
																					1,
																					second_stage_dot_logit_[n].length());

			//la::MulTransBOverwrite(second_stage_dot_logit_[n], second_stage_dot_logit_[n], &temp3);
			la::MulOverwrite(matrix_second_stage_dot_logit, 
														 tmatrix_second_stage_dot_logit, &temp3);
			
			//Matrix temp4; //ddotLogit-dotLogit*dotLogit'
			//temp4.Init(second_stage_x_[n].n_cols(), second_stage_x_[n].n_cols());
			la::SubOverwrite(temp3, second_stage_ddot_logit_[n], &temp4);

			//Matrix temp5;	//dotX2(temp4)
			//temp5.Init(betas.length(), second_stage_x_[n].n_cols());

			//la::MulOverwrite(second_stage_dot_logit_[n], temp4, &temp5);
			la::MulOverwrite(second_stage_x_[n], temp4, &temp5);

			//Matrix temp6; //temp5*dotX2'
			//temp6.Init(betas.length(), betas.length());
			//la::MulTransBOverwrite(temp5, second_stage_dot_logit_[n], &temp6);
			la::MulTransBOverwrite(temp5, second_stage_x_[n], &temp6);

			la::SubOverwrite(temp10, temp6, &second_term_temp);
			la::Scale( (conditional_postponed_prob)*(1-conditional_postponed_prob), &second_term_temp);

			la::AddOverwrite(second_term_temp, first_term_temp, &second_derivative_conditional_postpond_prob);
			//end of calculation of second_derivative_conditional_postpond_prob
			
			//Scale with beta_function
			la::Scale( beta_function_temp, &second_derivative_conditional_postpond_prob );
			//check
			

			la::AddTo(second_derivative_conditional_postpond_prob, 
								&sum_second_derivative_conditional_postpond_prob_[n]);


			la::Scale( (conditional_postponed_prob)*(1-conditional_postponed_prob), &first_derivative_conditional_postpond_prob);

			//Scale with beta_function
			la::Scale( beta_function_temp, &first_derivative_conditional_postpond_prob );

			//Check
			la::AddTo(first_derivative_conditional_postpond_prob, &sum_first_derivative_conditional_postpond_prob_[n]);
			
			if(l>=num_of_alphas_-2 && n>=first_stage_x_.size()-1) {
				continue;
			} else {
				matrix_first_derivative_conditional_postpond_prob.Destruct();
				tmatrix_first_derivative_conditional_postpond_prob.Destruct();
				//matrix_first_stage_dot_logit.Destruct();
				//tmatrix_first_stage_dot_logit.Destruct();
				matrix_second_stage_dot_logit.Destruct();
				tmatrix_second_stage_dot_logit.Destruct();
			}



		}	//alpha


		la::Scale(alpha_weight_, &sum_first_derivative_conditional_postpond_prob_[n]);
		la::Scale(alpha_weight_, &sum_second_derivative_conditional_postpond_prob_[n]);
		

		if(n<first_stage_x_.size()-1) {
			matrix_first_stage_dot_logit.Destruct();
			tmatrix_first_stage_dot_logit.Destruct();
			
		}
	}	//n
		//cout<<"test n end"<<endl;


}



void Objective::ComputeDerivativeBetaTerm2_(Vector *beta_term2) {

	//derivative_beta_term2.Init(betas.length());
	//derivative_beta_term2.SetZero();
	Vector temp;
	temp.Init(num_of_betas_);

	Vector temp2;
	temp2.Init(num_of_betas_);
	temp2.SetZero();

	for(index_t n=0; n<first_stage_x_.size(); n++){
		if (first_stage_y_[n]<0) {
      continue;
    } else {
			//check
			la::ScaleOverwrite((1/(1-postponed_probability_[n])), sum_first_derivative_conditional_postpond_prob_[n], &temp);
			//temp=SumFirstDerivativeConditionalPostpondProb_[n]/(1-postponed_probability_[n]);
			//check
			

		}	//if-else
		la::AddTo(temp, &temp2);

	}	//n
	//return derivative_beta_term2;
	beta_term2->Copy(temp2);

}

void Objective::ComputeSecondDerivativeBetaTerm1_(Matrix *second_beta_term1) {
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
	second_beta_term1->Copy(second_derivative_beta_term1);


}


void Objective::ComputeSecondDerivativeBetaTerm2_(Matrix *second_beta_term2) { 

	Matrix second_derivative_beta_term2;
	second_derivative_beta_term2.Init(num_of_betas_, num_of_betas_);
	second_derivative_beta_term2.SetZero();

	Matrix first_temp;
	first_temp.Init(num_of_betas_, num_of_betas_);

	Matrix second_temp;
	second_temp.Init(num_of_betas_, num_of_betas_);

	Matrix second_derivative_beta_temp;
	second_derivative_beta_temp.Init(num_of_betas_, num_of_betas_);

	Matrix matrix_sum_first_derivative_conditional_postpond_prob;
	//matrix_sum_first_derivative_conditional_postpond_prob.Init(num_of_betas_, 1);

	Matrix tmatrix_sum_first_derivative_conditional_postpond_prob;
	//tmatrix_sum_first_derivative_conditional_postpond_prob.Init(num_of_betas_, 1);


	for(index_t n=0; n<first_stage_x_.size(); n++){
		  if (first_stage_y_[n]<0) { 
				continue;
			} else {
				la::ScaleOverwrite( (1/(1-postponed_probability_[n])), sum_second_derivative_conditional_postpond_prob_[n], &first_temp);

				//handle vector transpose
				
				matrix_sum_first_derivative_conditional_postpond_prob.Alias(sum_first_derivative_conditional_postpond_prob_[n].ptr(),
																																		sum_first_derivative_conditional_postpond_prob_[n].length(),
																																		1);
				tmatrix_sum_first_derivative_conditional_postpond_prob.Alias(sum_first_derivative_conditional_postpond_prob_[n].ptr(),
																																		1,
																																		sum_first_derivative_conditional_postpond_prob_[n].length());																	



				//la::MulTransBOverwrite(sum_first_derivative_conditional_postpond_prob_[n], 
				//											 sum_first_derivative_conditional_postpond_prob_[n], &second_temp);
				la::MulOverwrite(matrix_sum_first_derivative_conditional_postpond_prob, 
															 tmatrix_sum_first_derivative_conditional_postpond_prob, &second_temp);

				la::Scale( 1/pow((1-postponed_probability_[n]), 2), &second_temp);

				la::AddOverwrite(second_temp, first_temp, &second_derivative_beta_temp);

				//check
				la::AddTo(second_derivative_beta_temp, &second_derivative_beta_term2);

							
			}	//else

	}	//n
	la::Scale(-1, &second_derivative_beta_term2);
	second_beta_term2->Copy(second_derivative_beta_term2);
	//return second_derivative_beta_term2;
}



void Objective::ComputeDerivativeBetaTerm3_(Vector *beta_term3) {
	//derivative_beta_term3.Init(betas.length());
	//derivative_beta_term3.SetZero();
	Vector temp;
	temp.Init(num_of_betas_);

	Vector temp2;
	temp2.Init(num_of_betas_);
	temp2.SetZero();

	for(index_t n=0; n<first_stage_x_.size(); n++){
		if (second_stage_y_[n]<0) {
      continue;
    } else {
			//check
			la::ScaleOverwrite( (1/postponed_probability_[n]), sum_first_derivative_conditional_postpond_prob_[n], &temp);
			//temp=SumFirstDerivativeConditionalPostpondProb_[n]/(postponed_probability_[n]);
			//check
			

		}	//if-else
		la::AddTo(temp, &temp2);

	}	//n
	//return derivative_beta_term3;
	beta_term3->Copy(temp2);

}


void Objective::ComputeSecondDerivativeBetaTerm3_(Matrix *second_beta_term3) {
	Matrix second_derivative_beta_term3;
	second_derivative_beta_term3.Init(num_of_betas_, num_of_betas_);
	second_derivative_beta_term3.SetZero();

	Matrix first_temp;
	first_temp.Init(num_of_betas_, num_of_betas_);

	Matrix second_temp;
	second_temp.Init(num_of_betas_, num_of_betas_);

	Matrix second_derivative_beta_temp;
	second_derivative_beta_temp.Init(num_of_betas_, num_of_betas_);

	Matrix matrix_sum_first_derivative_conditional_postpond_prob;
	//matrix_sum_first_derivative_conditional_postpond_prob.Init(num_of_betas_,1);

	Matrix tmatrix_sum_first_derivative_conditional_postpond_prob;
	//tmatrix_sum_first_derivative_conditional_postpond_prob.Init(num_of_betas_,1);



	for(index_t n=0; n<first_stage_x_.size(); n++){
		  if (second_stage_y_[n]<0) { 
				continue;
			} else {
				la::Scale( (1/(postponed_probability_[n])), &first_temp);
	
				//handle vector transpose

				matrix_sum_first_derivative_conditional_postpond_prob.Alias(sum_first_derivative_conditional_postpond_prob_[n].ptr(),
																									sum_first_derivative_conditional_postpond_prob_[n].length(),
																									1);
				tmatrix_sum_first_derivative_conditional_postpond_prob.Alias(sum_first_derivative_conditional_postpond_prob_[n].ptr(),
																									1,
																									sum_first_derivative_conditional_postpond_prob_[n].length());		


				//la::MulTransBOverwrite(sum_first_derivative_conditional_postpond_prob_[n], 
				//											 sum_first_derivative_conditional_postpond_prob_[n], &second_temp);

				la::MulOverwrite(matrix_sum_first_derivative_conditional_postpond_prob, 
															 tmatrix_sum_first_derivative_conditional_postpond_prob, &second_temp);


				la::Scale( 1/pow((1-postponed_probability_[n]), 2), &second_temp);

				la::SubOverwrite(second_temp, first_temp, &second_derivative_beta_temp);

				//check
				la::AddTo(second_derivative_beta_temp, &second_derivative_beta_term3);

							
			}	//else

	}	//n
	second_beta_term3->Copy(second_derivative_beta_term3);
	
	//return second_derivative_beta_term3;
}



double Objective::ComputeDerivativePTerm1_() {
	double derivative_p_term1=0;

	return derivative_p_term1;
}



double Objective::ComputeSecondDerivativePTerm1_() {
	double second_derivative_p_term1=0;

	return second_derivative_p_term1;
}



double Objective::ComputeDerivativeQTerm1_() {
	double derivative_q_term1=0;

	return derivative_q_term1;
}



double Objective::ComputeSecondDerivativeQTerm1_() {
	double second_derivative_q_term1=0;

	return second_derivative_q_term1;
}

void Objective::ComputeSumDerivativeBetaFunction_(Vector &betas, double p, double q) {
	double alpha_temp=0;
	double t_temp=0;

	//num_of_alphas_=10;
	//num_of_t_beta_fn_=10;
	alpha_weight_=(double)1/num_of_alphas_;
	t_weight_=(double)1/num_of_t_beta_fn_;
	
	/*for(index_t i=0; i<sum_first_derivative_p_beta_fn_.size(); i++) {
		sum_first_derivative_p_beta_fn_[i]=0;
		sum_second_derivative_p_beta_fn_[i]=0;
		sum_first_derivative_q_beta_fn_[i]=0;
		sum_second_derivative_q_beta_fn_[i]=0;
		sum_second_derivative_p_q_beta_fn_[i]=0;
	}
	*/


	double beta_fn_temp1=0;
	double beta_fn_temp2=0;
	double beta_fn_temp3=0;
	double beta_fn_temp4=0;
	double beta_fn_temp5=0;
	double beta_fn_temp6=0;

	double exponential_temp=0;
	double powtemp=0;
	double conditional_postponed_prob=0;

	Vector first_derivative_conditional_postpond_prob;
	first_derivative_conditional_postpond_prob.Init(betas.length());

	Vector sum_second_derivative_conditionl_postponed_p_temp;
	sum_second_derivative_conditionl_postponed_p_temp.Init(betas.length());
	

	Vector sum_second_derivative_conditionl_postponed_q_temp;
	sum_second_derivative_conditionl_postponed_q_temp.Init(betas.length());




	Vector temp1;	//dotX1*dotLogit1
	temp1.Init(betas.length());

	Vector temp2;	//dotX2*dotLogit2
	temp2.Init(betas.length());



	beta_fn_temp1=(double)1/denumerator_beta_function_;
	
	for(index_t m=0; m<num_of_t_beta_fn_-1; m++){
		t_temp=(m+1)*(t_weight_);
		
		beta_fn_temp2+=(pow(t_temp, p-1)*pow(1-t_temp, q-1)*log(t_temp));
		beta_fn_temp3+=pow(t_temp, p-1)*pow(1-t_temp, q-1)*pow(log(t_temp), 2);
		beta_fn_temp4+=pow(t_temp, p-1)*pow(1-t_temp, q-1)*log(1-t_temp);
		beta_fn_temp5+=pow(t_temp, p-1)*pow(1-t_temp, q-1)*pow(log(1-t_temp), 2);
		beta_fn_temp6+=pow(t_temp, p-1)*pow(1-t_temp, q-1)*log(1-t_temp)*log(t_temp);


	}		//m
	beta_fn_temp2*=(t_weight_/(pow(denumerator_beta_function_, 2)));
	beta_fn_temp3*=(t_weight_/pow(denumerator_beta_function_, 2));
	beta_fn_temp4*=(t_weight_/pow(denumerator_beta_function_, 2));
	beta_fn_temp5*=(t_weight_/pow(denumerator_beta_function_, 2));
	beta_fn_temp6*=(t_weight_/pow(denumerator_beta_function_, 2));

	for(index_t n=0; n<first_stage_x_.size(); n++){

		la::MulOverwrite(first_stage_x_[n], first_stage_dot_logit_[n], &temp1);
		for(index_t l=0; l<num_of_alphas_-1; l++){
			alpha_temp=(l+1)*(alpha_weight_);

			//Calculate x^2_{ni}(alpha_l)
			for(index_t i=0; i<first_stage_x_[n].n_cols(); i++){
				
				int count=0;
				for(index_t j=ind_unknown_x_[0]; j<=ind_unknown_x_[ind_unknown_x_.size()-1]; j++){
					
					
					exponential_temp=alpha_temp*first_stage_x_[n].get(j-1, i)
													+(alpha_temp)*(1-alpha_temp)*unknown_x_past_[n].get(count,0)
													+(alpha_temp)*pow((1-alpha_temp),2)*unknown_x_past_[n].get(count,1);
					second_stage_x_[n].set(j-1, i, exponential_temp);
					count+=first_stage_x_[n].n_cols();
				}	//j
			}	//i
			

			for(index_t i=0; i<second_stage_x_[n].n_cols(); i++) {
				exp_betas_times_x2_[n]+=exp(la::Dot(betas.length(), betas.ptr(),
											 second_stage_x_[n].GetColumnPtr(i) ));
			}	//i




			for(index_t i=0; i<second_stage_x_[n].n_cols(); i++) {
				//Calculate second_stage_dot_logit_
				second_stage_dot_logit_[n][i]=((exp(la::Dot(betas.length(), betas.ptr(),
																				 second_stage_x_[n].GetColumnPtr(i))))/
																				 exp_betas_times_x2_[n]);
																				 
				second_stage_ddot_logit_[n].set(i, i, first_stage_dot_logit_[n][i]);
			}	//i

			
			conditional_postponed_prob=exp_betas_times_x2_[n]/(exp_betas_times_x1_[n]+exp_betas_times_x2_[n]);
			la::MulOverwrite(second_stage_x_[n], second_stage_dot_logit_[n], &temp2);
			la::SubOverwrite(temp2, temp1, &first_derivative_conditional_postpond_prob);
			la::Scale( (conditional_postponed_prob*(1-conditional_postponed_prob)), &first_derivative_conditional_postpond_prob);

			
			powtemp=pow(alpha_temp, p-1)*pow(1-alpha_temp, q-1);

			sum_first_derivative_p_beta_fn_[n]+=conditional_postponed_prob
																					*( powtemp
																					*(log(alpha_temp)*beta_fn_temp1
																					- beta_fn_temp2) );
			sum_second_derivative_p_beta_fn_[n]+=conditional_postponed_prob
																					 *( powtemp
																					 *( pow(log(alpha_temp), 2)*beta_fn_temp1
																					 -2*log(alpha_temp)*beta_fn_temp2
																					 +beta_fn_temp3));
			sum_first_derivative_q_beta_fn_[n]+=conditional_postponed_prob
																					*( powtemp
																					*(log(1-alpha_temp)*beta_fn_temp1
																					- beta_fn_temp4) );
			sum_second_derivative_q_beta_fn_[n]+=conditional_postponed_prob
																					 *( powtemp
																					 *( pow(log(1-alpha_temp), 2)*beta_fn_temp1
																					 -2*log(1-alpha_temp)*beta_fn_temp4
																					 +beta_fn_temp5));
			sum_second_derivative_p_q_beta_fn_[n]+=conditional_postponed_prob
																					 *((powtemp*(log(1-alpha_temp)*log(alpha_temp)*beta_fn_temp1+log(1-alpha_temp)*beta_fn_temp2-log(alpha_temp)*beta_fn_temp4-beta_fn_temp6))
																					 -((powtemp*(log(1-alpha_temp)*beta_fn_temp1- beta_fn_temp4))*(2*beta_fn_temp2*denumerator_beta_function_)));
																					 

			//Calculate sum_second_derivative_conditionl_postponed_p_[n]
			//check
			la::ScaleOverwrite(( powtemp*(log(alpha_temp)*beta_fn_temp1-beta_fn_temp2) ), first_derivative_conditional_postpond_prob,
												&sum_second_derivative_conditionl_postponed_p_temp);
			la::AddTo(sum_second_derivative_conditionl_postponed_p_temp, &sum_second_derivative_conditionl_postponed_p_[n]);
			//sum_second_derivative_conditionl_postponed_p_[n]+=first_derivative_conditional_postpond_prob
			//																									*( powtemp
			//																									*(log(alpha_temp)*beta_fn_temp1
		  //																										- beta_fn_temp2) );
			
      //Calculate sum_second_derivative_conditionl_postponed_q_[n]
			//check
			la::ScaleOverwrite(( powtemp*(log(1-alpha_temp)*beta_fn_temp1- beta_fn_temp4) ), first_derivative_conditional_postpond_prob,
												&sum_second_derivative_conditionl_postponed_q_temp);
			la::AddTo(sum_second_derivative_conditionl_postponed_q_temp, &sum_second_derivative_conditionl_postponed_q_[n]);


		}	//l
		sum_first_derivative_p_beta_fn_[n]*=alpha_weight_;
		sum_second_derivative_p_beta_fn_[n]*=alpha_weight_;
		sum_first_derivative_q_beta_fn_[n]*=alpha_weight_;
		sum_second_derivative_q_beta_fn_[n]*=alpha_weight_;
		sum_second_derivative_p_q_beta_fn_[n]*=alpha_weight_;
		la::Scale(alpha_weight_, &sum_second_derivative_conditionl_postponed_p_[n]);
		la::Scale(alpha_weight_, &sum_second_derivative_conditionl_postponed_q_[n]);
	}	//n
}




double Objective::ComputeDerivativePTerm2_() {
	double derivative_p_term2=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
		
    if (first_stage_y_[n]<0) {
      continue;
    } else {
      derivative_p_term2+=(sum_first_derivative_p_beta_fn_[n]/(1-postponed_probability_[n]));
    }
  }
	derivative_p_term2*=-1;
	
  return derivative_p_term2;
	
}



double Objective::ComputeDerivativePTerm3_() {
	double derivative_p_term3=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (second_stage_y_[n]<0) {
      continue;
    } else {
      derivative_p_term3+=(sum_first_derivative_p_beta_fn_[n]/(postponed_probability_[n]));
    }
  }
	return derivative_p_term3;
}


double Objective::ComputeSecondDerivativePTerm2_() {
	double second_derivative_p_term2=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) {
      continue;
    } else {
      second_derivative_p_term2+=( sum_second_derivative_p_beta_fn_[n]/(1-postponed_probability_[n]))
																	 + pow( (sum_first_derivative_p_beta_fn_[n]/(1-postponed_probability_[n])), 2);
    }
  }
	second_derivative_p_term2*=-1;
  return second_derivative_p_term2;
}



double Objective::ComputeSecondDerivativePTerm3_() {
	double second_derivative_p_term3=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (second_stage_y_[n]<0) {
      continue;
    } else {
      second_derivative_p_term3+=( sum_second_derivative_p_beta_fn_[n]/(postponed_probability_[n]))
																	 - pow( (sum_first_derivative_p_beta_fn_[n]/(postponed_probability_[n])), 2);
    }
  }
	return second_derivative_p_term3;
}



double Objective::ComputeDerivativeQTerm2_() {
	double derivative_q_term2=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) {
      continue;
    } else {
      derivative_q_term2+=(sum_first_derivative_q_beta_fn_[n]/(1-postponed_probability_[n]));
    }
  }
	derivative_q_term2*=-1;
  return derivative_q_term2;
}


double Objective::ComputeDerivativeQTerm3_() {
	double derivative_q_term3=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (second_stage_y_[n]<0) {
      continue;
    } else {
      derivative_q_term3+=(sum_first_derivative_q_beta_fn_[n]/(postponed_probability_[n]));
    }
  }
	return derivative_q_term3;
}




double Objective::ComputeSecondDerivativeQTerm2_(){
	double second_derivative_q_term2=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) {
      continue;
    } else {
      second_derivative_q_term2+=( sum_second_derivative_q_beta_fn_[n]/(1-postponed_probability_[n]))
																	 + pow( (sum_first_derivative_q_beta_fn_[n]/(1-postponed_probability_[n])), 2);
    }
  }
	second_derivative_q_term2*=-1;
  return second_derivative_q_term2;
}



double Objective::ComputeSecondDerivativeQTerm3_() {
	double second_derivative_q_term3=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (second_stage_y_[n]<0) {
      continue;
    } else {
      second_derivative_q_term3+=( sum_second_derivative_q_beta_fn_[n]/(postponed_probability_[n]))
																	 - pow( (sum_first_derivative_q_beta_fn_[n]/(postponed_probability_[n])), 2);
    }
  }
	return second_derivative_q_term3;
}




void Objective::ComputeSecondDerivativePBetaTerm1_(Vector *p_beta_term1) {
	Vector second_derivative_p_beta_term1;
	second_derivative_p_beta_term1.Init(num_of_betas_);
	second_derivative_p_beta_term1.SetZero();

	p_beta_term1->Copy(second_derivative_p_beta_term1);

}



void Objective::ComputeSecondDerivativePBetaTerm2_(Vector *p_beta_term2) {
	Vector second_derivative_p_beta_term2;
	second_derivative_p_beta_term2.Init(num_of_betas_);
	second_derivative_p_beta_term2.SetZero();

	Vector temp1;
	temp1.Init(num_of_betas_);

	Vector temp2;
	temp2.Init(num_of_betas_);

	Vector temp3;
	temp3.Init(num_of_betas_);

	Vector temp4;
	temp4.Init(num_of_betas_);


	//second_derivative_p_beta_term2.SetZero();
	for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) {
      continue;
    } else {
			//temp1-first term
			la::ScaleOverwrite( pow((1-postponed_probability_[n]), -1), sum_second_derivative_conditionl_postponed_p_[n], &temp1);
			la::ScaleOverwrite( (sum_first_derivative_p_beta_fn_[n]/(pow((1-postponed_probability_[n]), 2))),
												 sum_first_derivative_conditional_postpond_prob_[n], &temp2);
			la::AddOverwrite( temp2, temp1, &temp3);
			la::AddTo(temp3, &second_derivative_p_beta_term2);
      //second_derivative_p_beta_term2+=(sum_first_derivative_q_beta_fn_[n]/(1-postponed_probability_[n]));
    }
  }
	la::Scale(-1, &second_derivative_p_beta_term2);
	p_beta_term2->Copy(second_derivative_p_beta_term2);
}



void Objective::ComputeSecondDerivativePBetaTerm3_(Vector *p_beta_term3) {
	Vector second_derivative_p_beta_term3;
	second_derivative_p_beta_term3.Init(num_of_betas_);
	second_derivative_p_beta_term3.SetZero();

	Vector temp1;
	temp1.Init(num_of_betas_);

	Vector temp2;
	temp2.Init(num_of_betas_);

	Vector temp3;
	temp3.Init(num_of_betas_);


	//second_derivative_p_beta_term2.SetZero();
	for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (second_stage_y_[n]<0) {
      continue;
    } else {
			//temp1-first term
			la::ScaleOverwrite((1/(postponed_probability_[n])), sum_second_derivative_conditionl_postponed_p_[n], &temp1);
			la::ScaleOverwrite( (sum_first_derivative_p_beta_fn_[n]/(pow((postponed_probability_[n]), 2))),
												 sum_first_derivative_conditional_postpond_prob_[n], &temp2);
			la::AddOverwrite( temp2, temp1, &temp3);
			la::AddTo(temp3, &second_derivative_p_beta_term3);
      //second_derivative_p_beta_term2+=(sum_first_derivative_q_beta_fn_[n]/(1-postponed_probability_[n]));
    }
  }
	p_beta_term3->Copy(second_derivative_p_beta_term3);
}


void Objective::ComputeSecondDerivativeQBetaTerm1_(Vector *q_beta_term1) {
	Vector second_derivative_q_beta_term1;
	second_derivative_q_beta_term1.Init(num_of_betas_);
	second_derivative_q_beta_term1.SetZero();

	q_beta_term1->Copy(second_derivative_q_beta_term1);
}


void Objective::ComputeSecondDerivativeQBetaTerm2_(Vector *q_beta_term2) {
	Vector second_derivative_q_beta_term2;
	second_derivative_q_beta_term2.Init(num_of_betas_);
	second_derivative_q_beta_term2.SetZero();

	Vector temp1;
	temp1.Init(num_of_betas_);

	Vector temp2;
	temp2.Init(num_of_betas_);

	Vector temp3;
	temp3.Init(num_of_betas_);


	//second_derivative_p_beta_term2.SetZero();
	for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) {
      continue;
    } else {
			//temp1-first term
			la::ScaleOverwrite((1/(1-postponed_probability_[n])), sum_second_derivative_conditionl_postponed_q_[n], &temp1);
			la::ScaleOverwrite( (sum_first_derivative_q_beta_fn_[n]/(pow((1-postponed_probability_[n]), 2))),
												 sum_first_derivative_conditional_postpond_prob_[n], &temp2);
			la::AddOverwrite( temp2, temp1, &temp3);
			la::AddTo(temp3, &second_derivative_q_beta_term2);
      //second_derivative_p_beta_term2+=(sum_first_derivative_q_beta_fn_[n]/(1-postponed_probability_[n]));
    }
  }
	la::Scale(-1, &second_derivative_q_beta_term2);
	q_beta_term2->Copy(second_derivative_q_beta_term2);

}


void Objective::ComputeSecondDerivativeQBetaTerm3_(Vector *q_beta_term3) {
	Vector second_derivative_q_beta_term3;
	second_derivative_q_beta_term3.Init(num_of_betas_);
	second_derivative_q_beta_term3.SetZero();

	Vector temp1;
	temp1.Init(num_of_betas_);

	Vector temp2;
	temp2.Init(num_of_betas_);

	Vector temp3;
	temp3.Init(num_of_betas_);


	//second_derivative_p_beta_term2.SetZero();
	for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (second_stage_y_[n]<0) {
      continue;
    } else {
			//temp1-first term
			la::ScaleOverwrite((1/(postponed_probability_[n])), sum_second_derivative_conditionl_postponed_q_[n], &temp1);
			la::ScaleOverwrite( (sum_first_derivative_q_beta_fn_[n]/(pow((postponed_probability_[n]), 2))),
												 sum_first_derivative_conditional_postpond_prob_[n], &temp2);
			la::AddOverwrite( temp2, temp1, &temp3);
			la::AddTo(temp3, &second_derivative_q_beta_term3);
      //second_derivative_p_beta_term2+=(sum_first_derivative_q_beta_fn_[n]/(1-postponed_probability_[n]));
    }
  }
	q_beta_term3->Copy(second_derivative_q_beta_term3);

}



double Objective::ComputeSecondDerivativePQTerm1_() {
	double second_derivative_p_q_term1=0;

	return second_derivative_p_q_term1;
}


double Objective::ComputeSecondDerivativePQTerm2_() {
	double second_derivative_p_q_term2=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) {
      continue;
    } else {
      second_derivative_p_q_term2+=( sum_second_derivative_p_q_beta_fn_[n]/(1-postponed_probability_[n]))
																		+ ((sum_first_derivative_p_beta_fn_[n]/(1-postponed_probability_[n]))
																		* (sum_first_derivative_q_beta_fn_[n]/(1-postponed_probability_[n])));
    }
  }
	second_derivative_p_q_term2*=-1;
  return second_derivative_p_q_term2;
}


double Objective::ComputeSecondDerivativePQTerm3_() {
	double second_derivative_p_q_term3=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (second_stage_y_[n]<0) {
      continue;
    } else {
      second_derivative_p_q_term3+=( sum_second_derivative_p_q_beta_fn_[n]/(postponed_probability_[n]))
																		- ((sum_first_derivative_p_beta_fn_[n]/(postponed_probability_[n]))
																		* (sum_first_derivative_q_beta_fn_[n]/(postponed_probability_[n])));
    }
  }
  return second_derivative_p_q_term3;
}







