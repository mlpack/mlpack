#include "objective2.h"
#include <cmath>
#include <iostream>
#include <iostream>
#include <algorithm> //sqrt

using namespace std;
  

/*
void Objective::Init(ArrayList<Matrix> &added_first_stage_x, 
										 ArrayList<Matrix> &added_second_stage_x, 
										 ArrayList<Matrix> &added_unknown_x_past, 
										 ArrayList<index_t> &added_first_stage_y,
										 Vector &ind_unknown_x) {


	index_t previous_num_selected_people=0;
	//current_num_selected_people
	num_of_betas_=added_first_stage_x[0].n_rows();
	index_t num_selected_people=added_first_stage_x.size();

	

	
	//first_stage_x_.Init(num_selected_people);
	//first_stage_x_.PushBackCopy(added_first_stage_x);
	//for(index_t i=0; i<num_selected_people; i++) {
    //first_stage_x_[i].Init(added_first_stage_x[i].n_rows(), added_first_stage_x[i].n_cols());
    //first_stage_x_[i].PushBackCopy(added_first_stage_x[i]);
    
  //}
	second_stage_x_.Init(num_selected_people);
	second_stage_x_.Copy(added_second_stage_x);

	
	unknown_x_past_.Init(num_selected_people);
	unknown_x_past_.Copy(added_unknown_x_past);

	first_stage_y_.Init(num_selected_people);
	first_stage_y_.Copy(added_first_stage_y);

	
	ind_unknown_x_.Copy(ind_unknown_x);









	exp_betas_times_x1_.Init(num_selected_people);
  exp_betas_times_x2_.Init(num_selected_people);
	postponed_probability_.Init(num_selected_people);
	for(index_t i=0; i<postponed_probability_.size(); i++) {
    postponed_probability_[i]=0;
  }

	denumerator_beta_function_=0;
	num_of_t_beta_fn_=0;
	t_weight_=0;
	num_of_alphas_=0;
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

	
	


	//previous_num_selected_people_=current_num_selected_people;
	//current_num_selected_people


	
}

*/


void Objective::Init2(Vector &ind_unknown_x, int count_init2) {

	if(count_init2 ==0) {
	first_stage_x_.Init();
	second_stage_x_.Init();
	unknown_x_past_.Init();
	first_stage_y_.Init();
	ind_unknown_x_.Copy(ind_unknown_x);
	}


	exp_betas_times_x1_.Init();
  exp_betas_times_x2_.Init();
	postponed_probability_.Init();
	//for(index_t i=0; i<postponed_probability_.size(); i++) {
  //  postponed_probability_[i]=0;
  //}

	denumerator_beta_function_=0;
	num_of_t_beta_fn_=100;
	t_weight_=1;
	num_of_alphas_=100;
	alpha_weight_=1;  

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
	

	second_stage_dot_logit_.Init();
	second_stage_ddot_logit_.Init();

	//for(index_t i=0; i<second_stage_dot_logit_.size(); i++) {
    //exp_betas_times_x2_[i]=0;
		//second_stage_dot_logit_[i].Init(first_stage_x_[i].n_cols());
		//second_stage_dot_logit_[i].SetZero();
		//second_stage_ddot_logit_[i].Init(first_stage_x_[i].n_cols(),first_stage_x_[i].n_cols());
		//second_stage_ddot_logit_[i].SetZero();

  //}

	sum_first_derivative_conditional_postpond_prob_.Init();
	sum_second_derivative_conditional_postpond_prob_.Init();

	//for(index_t n=0; n<first_stage_x_.size(); n++) {
		//sum_first_derivative_conditional_postpond_prob_[n].Init(num_of_betas_);
		//sum_first_derivative_conditional_postpond_prob_[n].SetZero();
		//sum_second_derivative_conditional_postpond_prob_[n].Init(num_of_betas_, num_of_betas_);
		//sum_second_derivative_conditional_postpond_prob_[n].SetZero();

	//}
	

	sum_first_derivative_p_beta_fn_.Init();
	sum_second_derivative_p_beta_fn_.Init();
	sum_first_derivative_q_beta_fn_.Init();
	sum_second_derivative_q_beta_fn_.Init();
	sum_second_derivative_p_q_beta_fn_.Init();
	sum_second_derivative_conditionl_postponed_p_.Init();
	sum_second_derivative_conditionl_postponed_q_.Init();


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





void Objective::Init3(int sample_size,
						ArrayList<Matrix> &added_first_stage_x,
						ArrayList<Matrix> &added_second_stage_x, 
						ArrayList<Matrix> &added_unknown_x_past, 
						ArrayList<index_t> &added_first_stage_y) {

	
	num_of_betas_=added_first_stage_x[0].n_rows();
	int num_selected_people=added_first_stage_x.size();

	for(index_t i=sample_size; i<num_selected_people; i++){
		first_stage_x_.PushBackCopy(added_first_stage_x[i]);
		second_stage_x_.PushBackCopy(added_second_stage_x[i]);
		unknown_x_past_.PushBackCopy(added_unknown_x_past[i]);
		first_stage_y_.PushBackCopy(added_first_stage_y[i]);

	}
	exp_betas_times_x1_.Destruct();
	exp_betas_times_x1_.Init(num_selected_people);

	exp_betas_times_x2_.Destruct();
  exp_betas_times_x2_.Init(num_selected_people);

	postponed_probability_.Destruct();
	postponed_probability_.Init(num_selected_people);
	for(index_t i=0; i<postponed_probability_.size(); i++) {
    postponed_probability_[i]=0;
  }

	 

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
	
	second_stage_dot_logit_.Destruct();
	second_stage_dot_logit_.Init(num_selected_people);
	second_stage_ddot_logit_.Destruct();
	second_stage_ddot_logit_.Init(num_selected_people);

	for(index_t i=0; i<second_stage_dot_logit_.size(); i++) {
    exp_betas_times_x2_[i]=0;
		second_stage_dot_logit_[i].Init(first_stage_x_[i].n_cols());
		second_stage_dot_logit_[i].SetZero();
		second_stage_ddot_logit_[i].Init(first_stage_x_[i].n_cols(),first_stage_x_[i].n_cols());
		second_stage_ddot_logit_[i].SetZero();

  }
	
	sum_first_derivative_conditional_postpond_prob_.Destruct();
	sum_first_derivative_conditional_postpond_prob_.Init(num_selected_people);
	sum_second_derivative_conditional_postpond_prob_.Destruct();
	sum_second_derivative_conditional_postpond_prob_.Init(num_selected_people);

	for(index_t n=0; n<first_stage_x_.size(); n++) {
		sum_first_derivative_conditional_postpond_prob_[n].Init(num_of_betas_);
		sum_first_derivative_conditional_postpond_prob_[n].SetZero();
		sum_second_derivative_conditional_postpond_prob_[n].Init(num_of_betas_, num_of_betas_);
		sum_second_derivative_conditional_postpond_prob_[n].SetZero();

	}

	sum_first_derivative_p_beta_fn_.Destruct();
	sum_second_derivative_p_beta_fn_.Destruct();
	sum_first_derivative_q_beta_fn_.Destruct();
	sum_second_derivative_q_beta_fn_.Destruct();
	sum_second_derivative_p_q_beta_fn_.Destruct();
	sum_second_derivative_conditionl_postponed_p_.Destruct();
	sum_second_derivative_conditionl_postponed_q_.Destruct();


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





	

	
	//first_stage_x_.Init(num_select
}


//void Objective::ComputeObjective(Matrix &x, double *objective) {
void Objective::ComputeObjective(double current_sample,
																 Vector &current_parameter, 
																 double *objective) { 
	
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

//cout<<"term1="<<ComputeTerm1_(betas)<<endl;
//cout<<"term2="<<ComputeTerm2_()<<endl;
//cout<<"term3="<<ComputeTerm3_()<<endl;
  /**objective = ComputeTerm1_(betas) 
               + ComputeTerm2_()
               + ComputeTerm3_();
*/

	
	*objective = (-1/current_sample)*(ComputeTerm1_(betas) 
               + ComputeTerm2_()
               + ComputeTerm3_());

//	cout<<"The objective="<<*objective<<endl;
	


	//*objective=2;

	
	
}

////////////////////////////////////////////////
////Calculate gradient
////////////////////////////////////////////////

void Objective::ComputeGradient(double current_sample,
																Vector &current_parameter, 
																Vector *gradient) { 
	
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
																

	/*
	double derivative_p_term2;
  derivative_p_term2=ComputeDerivativePTerm2_();
	//cout<<"ComputeDerivativePTerm2_()="<<derivative_p_term2<<endl;
	double derivative_p_term3;
  derivative_p_term3=ComputeDerivativePTerm3_();
	cout<<"ComputeDerivativePTerm3_()="<<derivative_p_term3<<endl;

	double derivative_p_term;
	derivative_p_term=derivative_p_term2+derivative_p_term3;
	cout<<"derivative_p_term="<<derivative_p_term<<endl;
  
	double derivative_p_term;
	derivative_p_term=ComputeDerivativePTerm2_()+ComputeDerivativePTerm3_();
	cout<<"derivative_p_term="<<derivative_p_term<<endl;


  //cout<<"num_of_betas_"<<num_of_betas_<<endl;
  dummy_gradient[num_of_betas_]=derivative_p_term;
	dummy_gradient[num_of_betas_+1]=(ComputeDerivativeQTerm2_()+ComputeDerivativeQTerm3_());
  */

	/*
	Matrix dummy_gradient;
	dummy_gradient.Init(num_of_betas_+2, 1);
  dummy_gradient.SetZero();


  cout<<"dummy gradient"<<endl;
	for(index_t i=0; i<dummy_gradient.n_rows(); i++){
		cout<<dummy_gradient.get(i,0)<<" ";
	}
	cout<<endl;




	for(index_t i=0; i<num_of_betas_; i++){
		dummy_gradient.set(i,0, (dummy_beta_term1[i]+dummy_beta_term2[i]+dummy_beta_term3[i]));
	}
	
  cout<<"dummy gradient 2"<<endl;
	for(index_t i=0; i<dummy_gradient.n_rows(); i++){
		cout<<dummy_gradient.get(i,0)<<" ";
	}
	cout<<endl;
	
  dummy_gradient.set(num_of_betas_, 0, (ComputeDerivativePTerm2_()+ComputeDerivativePTerm3_()));
	dummy_gradient.set(num_of_betas_+1, 0, (ComputeDerivativeQTerm2_()+ComputeDerivativeQTerm3_()));

	cout<<"dummy gradient 3"<<endl;
	for(index_t i=0; i<dummy_gradient.n_rows(); i++){
		cout<<dummy_gradient.get(i,0)<<" ";
	}
	cout<<endl;

	//handle minimization
	la::Scale(-1.0/current_sample, &dummy_gradient);

	Vector dummy_gradient2;
  dummy_gradient.MakeColumnVector(0, &dummy_gradient2);


	gradient->Copy(dummy_gradient2);
	*/

	la::Scale(-1.0/current_sample, &dummy_gradient);
	//la::Scale(+1.0/current_sample, &dummy_gradient);
	gradient->Copy(dummy_gradient);
												
}



////////////////////////////////////////////////
////Calculate hessian
////////////////////////////////////////////////

void Objective::ComputeHessian(double current_sample,
															 Vector &current_parameter, 
															 Matrix *hessian) { 
	
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
	////cout<<"ddot done"<<endl;
  ComputeSumDerivativeConditionalPostpondProb_(betas, p, q);
	////cout<<"sumDerivativeCondPostpondprob done"<<endl;
	
	
	ComputeSumDerivativeBetaFunction_(betas, p, q);
	////cout<<"SumDerivativeBetaFunction done"<<endl;

	

	Matrix dummy_second_beta_term1;
	Matrix dummy_second_beta_term2;
	Matrix dummy_second_beta_term3;
	

	ComputeSecondDerivativeBetaTerm1_(&dummy_second_beta_term1);
	ComputeSecondDerivativeBetaTerm2_(&dummy_second_beta_term2);
	ComputeSecondDerivativeBetaTerm3_(&dummy_second_beta_term3);
	

	
	//Vector dummy_p_beta_term1;
	Vector dummy_p_beta_term2;
	Vector dummy_p_beta_term3;
	//Vector dummy_q_beta_term1;
	Vector dummy_q_beta_term2;
	Vector dummy_q_beta_term3;


	
	//ComputeSecondDerivativePBetaTerm1_(&dummy_p_beta_term1);
	ComputeSecondDerivativePBetaTerm2_(&dummy_p_beta_term2);
	ComputeSecondDerivativePBetaTerm3_(&dummy_p_beta_term3);
  
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

	Vector dummy_p_beta_term;
	dummy_p_beta_term.Init(num_of_betas_);
	dummy_p_beta_term.SetZero();
	la::AddOverwrite(dummy_p_beta_term2, dummy_p_beta_term3, &dummy_p_beta_term);

	/*
	cout<<"p_beta_term: "<<endl;
	for(index_t i=0; i<dummy_p_beta_term.length(); i++){
		cout<<dummy_p_beta_term[i]<<" ";
	}
	cout<<endl;
	*/

	//ComputeSecondDerivativeQBetaTerm1_(&dummy_q_beta_term1);
	ComputeSecondDerivativeQBetaTerm2_(&dummy_q_beta_term2);
	ComputeSecondDerivativeQBetaTerm3_(&dummy_q_beta_term3);
	//cout<<"SecondDerivativeQBetaTerm3_ done"<<endl;
	Vector dummy_q_beta_term;
	dummy_q_beta_term.Init(num_of_betas_);
	dummy_q_beta_term.SetZero();
	la::AddOverwrite(dummy_q_beta_term2, dummy_q_beta_term3, &dummy_q_beta_term);

  Matrix dummy_hessian;
	dummy_hessian.Init(num_of_betas_+2, num_of_betas_+2);
	dummy_hessian.SetZero();

	Matrix dummy_hessian_beta;
  dummy_hessian_beta.Init(num_of_betas_,num_of_betas_);
  dummy_hessian_beta.SetZero();
	la::AddOverwrite(dummy_second_beta_term1, dummy_second_beta_term2, &dummy_hessian_beta);
	la::AddTo(dummy_second_beta_term3, &dummy_hessian_beta);
	 
	/*
  cout<<"Hessian matrix beta"<<endl;	
	for (index_t j=0; j<dummy_second_beta_term3.n_rows(); j++){
		for (index_t k=0; k<dummy_second_beta_term3.n_cols(); k++){
				cout<<dummy_hessian_beta.get(j,k) <<"  ";
		}
		cout<<endl;
	}
	*/


	for(index_t i=0; i<num_of_betas_+2; i++){
		for(index_t j=0; j<num_of_betas_+2; j++){

			if(i<num_of_betas_ && j<num_of_betas_){
				//dummy_hessian.set(i,j, (dummy_second_beta_term1.get(i,j)
				//											+dummy_second_beta_term2.get(i,j)
				//											+dummy_second_beta_term2.get(i,j)));
				dummy_hessian.set(i,j, dummy_hessian_beta.get(i,j));
				
			} else if(i<num_of_betas_ && j>=num_of_betas_){
				//cout<<"i="<<i<<endl;
				//cout<<"j="<<j<<endl;
				/*
				dummy_hessian.set(i,num_of_betas_, (dummy_p_beta_term1[i]
															+dummy_p_beta_term2[i]
															+dummy_p_beta_term2[i]));
				dummy_hessian.set(i,num_of_betas_+1, (dummy_q_beta_term1[i]
															+dummy_q_beta_term2[i]
															+dummy_q_beta_term2[i]));

															*/

				dummy_hessian.set(i,num_of_betas_, (dummy_p_beta_term[i]));
				dummy_hessian.set(i,num_of_betas_+1, (dummy_q_beta_term[i]));
															
			}	else if(j<num_of_betas_ && i>=num_of_betas_){
				/*
				dummy_hessian.set(num_of_betas_, j, (dummy_p_beta_term1[j]
															+dummy_p_beta_term2[j]
															+dummy_p_beta_term2[j]));
				dummy_hessian.set(num_of_betas_+1, j, (dummy_q_beta_term1[j]
															+dummy_q_beta_term2[j]
															+dummy_q_beta_term2[j]));
															*/
				dummy_hessian.set(num_of_betas_, j, (dummy_p_beta_term[j]));
				dummy_hessian.set(num_of_betas_+1, j, (dummy_q_beta_term[j]));
			} 

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

void Objective::ComputeChoiceProbability(Vector &current_parameter, 
																				 Vector *choice_probability) {

	Vector betas;
  //betas.Alias(x.ptr(), x.n_rows());
	
	

	betas.Alias(current_parameter.ptr(), num_of_betas_);
  //double p=first_stage_x_[1].get(0, 0);
  //double q=first_stage_x_[1].get(0, 1);
  double p=current_parameter[num_of_betas_];
	double q=current_parameter[num_of_betas_+1];

	choice_probability->Init(first_stage_x_.size());

	
	ComputeExpBetasTimesX1_(betas);

	
  ComputeDeumeratorBetaFunction_(p, q);
	
  ComputePostponedProbability_(betas, 
                               p, 
                               q);

	for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) { 
			(*choice_probability)[n]=log(postponed_probability_[n]);
			//(*choice_probability)[n]=(postponed_probability_[n]);
      
    } else {
      Vector temp;
      first_stage_x_[n].MakeColumnVector((first_stage_y_[n]-1), &temp);
			//(*choice_probability)[n]=exp(la::Dot(betas, temp))/exp_betas_times_x1_[n];
			//(*choice_probability)[n]=exp(la::Dot(betas, temp))/exp_betas_times_x1_[n]
			//												*(1-postponed_probability_[n]);
			(*choice_probability)[n]=la::Dot(betas, temp)-log(exp_betas_times_x1_[n])+log((1-postponed_probability_[n]));
			
		}
  }


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

double Objective::ComputeTerm2_() {
  double term2=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) {
      continue;
    } else {
      DEBUG_ASSERT(1-postponed_probability_[n]);
      term2+=log(1-postponed_probability_[n]);
			//cout<<"term2"<<endl;
			//cout<<(log(1-postponed_probability_[n]))<<endl;
    }
  }
	//cout<<"term2="<<term2<<endl;
  return term2;
}

double Objective::ComputeTerm3_() {
  double term3=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]>0) {
      continue;
    } else {
      DEBUG_ASSERT(postponed_probability_[n]>0);
      term3+=log(postponed_probability_[n]);
			//cout<<"term3"<<endl;
			//cout<<(log(postponed_probability_[n]))<<endl;
			//cout<<"postponed_prob="<<postponed_probability_[n]<<endl;
    }
  }
	//cout<<"term3="<<term3<<endl;
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
		exp_betas_times_x2_[n]=0;
		postponed_probability_[n]=0;
		for(index_t l=0; l<num_of_alphas_-1; l++){
			double alpha_temp;
	    double beta_function_temp;
      alpha_temp=(l+1)*(alpha_weight_);
			//cout<<"alpha_temp="<<alpha_temp<<endl;
			beta_function_temp=pow(alpha_temp, p-1)*
          pow((1-alpha_temp), q-1)/denumerator_beta_function_;
			//cout<<"beta_fn_temp="<<beta_function_temp<<endl;
		
			
			//Calculate x^2_{ni}(alpha_l)=alpha*z3(1st-stage)+alpha(1-alpha)Z2(back0)+(1-alpha)^2*Z1(back-1)
			//int count=0;
			double j=ind_unknown_x_[0];

			for(index_t i=0; i<first_stage_x_[n].n_cols(); i++){
				
				exponential_temp=alpha_temp*first_stage_x_[n].get(j-1, i)
													+(alpha_temp)*(1-alpha_temp)*unknown_x_past_[n].get(0, i)
													+pow((1-alpha_temp),2)*unknown_x_past_[n].get(1,i);
				second_stage_x_[n].set(j-1, i, exponential_temp);
				//count+=first_stage_x_[n].n_cols();
				

				/*for(index_t j=ind_unknown_x_[0]; j<=ind_unknown_x_[ind_unknown_x_.length()-1]; j++){
					exponential_temp=alpha_temp*first_stage_x_[n].get(j-1, i)
													+(alpha_temp)*(1-alpha_temp)*unknown_x_past_[n].get(0, i)
													+(alpha_temp)*pow((1-alpha_temp),2)*unknown_x_past_[n].get(1,i);
					second_stage_x_[n].set(j-1, i, exponential_temp);
					count+=first_stage_x_[n].n_cols();
				}	//j
				*/

			}	//i
			


		  for(index_t i=0; i<exp_betas_times_x2_.size(); i++) {
			  exp_betas_times_x2_[i]=0;
		  }

			for(index_t i=0; i<second_stage_x_[n].n_cols(); i++) {
				//exp_betas_times_x2_[n]=0;
				exp_betas_times_x2_[n]+=exp(la::Dot(betas.length(), 
																betas.ptr(),
																second_stage_x_[n].GetColumnPtr(i)));			}
			
			//conditional_postponed_probability_[n]
			
			postponed_probability_[n]+=( (exp_betas_times_x2_[n]/(exp_betas_times_x1_[n]
																  + exp_betas_times_x2_[n]) )
																		*beta_function_temp );	
			//cout<<"beta_fn_temp "<<beta_function_temp<<endl;
			//cout<<"postpond_prob "<<postponed_probability_[n]<<endl;
			//cout<<"denumerator_beta_function_ "<<denumerator_beta_function_<<endl;
   
    }	//alpha

		

		//cout<<"exp_betas_times_x2_"<<n<<" "<<exp_betas_times_x2_[n]<<endl;


		postponed_probability_[n]*=alpha_weight_;	
	}	//n

		/*cout<<"second_stage_x:"<<endl;
		for(index_t i=0; i<second_stage_x_[2].n_cols(); i++){
			cout<<second_stage_x_[2].get(2,i)<<" ";
		}
		cout<<endl;
		*/
	/*
	cout<<"exp_betas_times_x2_[n] part1:"<<endl;
	for(index_t n=0; n<exp_betas_times_x2_.size(); n++){
		cout<<exp_betas_times_x2_[n]<<" ";
	}
	cout<<endl;
	
		
	cout<<"postponed_probability_"<<endl;
	for(index_t i=0; i<postponed_probability_.size(); i++){
		cout<<postponed_probability_[i]<<" ";
	}
	cout<<endl;
  */

	//postponed_probability_check

}

void Objective::ComputeExpBetasTimesX1_(Vector &betas) {
  
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
	//cout<<"denumerator_beta_function_"<<denumerator_beta_function_<<endl;
	
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

			la::SubOverwrite(num_of_betas_, temp.ptr(), first_stage_x_[n].GetColumnPtr(first_stage_y_[n]-1), temp2.ptr());
			la::AddTo(temp2, &temp3);
																									
		}	//else
		//la::AddTo(temp2, &temp3);
		
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


	//double conditional_postponed_prob;
	Vector conditional_postponed_prob;
	conditional_postponed_prob.Init(first_stage_x_.size());

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

	conditional_postponed_prob.SetZero();

	for(index_t n=0; n<first_stage_x_.size(); n++){

		sum_first_derivative_conditional_postpond_prob_[n].SetZero();
		sum_second_derivative_conditional_postpond_prob_[n].SetZero();
		//conditional_postponed_prob=0;
		

		exp_betas_times_x2_[n]=0;

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
		
		/*
		cout<<"temp10"<<endl;
		for(index_t i=0; i<temp10.n_rows(); i++){
			for(index_t j=0; j<temp10.n_cols(); j++){
				cout<<temp10.get(i,j)<<" ";
			}
			cout<<endl;

		}
		*/

		
		

		for(index_t l=0; l<num_of_alphas_-1; l++){
			alpha_temp=(l+1)*(alpha_weight_);
		
			beta_function_temp=pow(alpha_temp, p-1)*pow((1-alpha_temp), q-1)/denumerator_beta_function_;

			//Calculate x^2_{ni}(alpha_l)
			//int count=0;
			double j=ind_unknown_x_[0];

			for(index_t i=0; i<first_stage_x_[n].n_cols(); i++){
				
				exponential_temp=alpha_temp*first_stage_x_[n].get(j-1, i)
													+(alpha_temp)*(1-alpha_temp)*unknown_x_past_[n].get(0, i)
													+pow((1-alpha_temp),2)*unknown_x_past_[n].get(1,i);
				second_stage_x_[n].set(j-1, i, exponential_temp);
				//count+=first_stage_x_[n].n_cols();
				

				/*for(index_t j=ind_unknown_x_[0]; j<=ind_unknown_x_[ind_unknown_x_.length()-1]; j++){
					exponential_temp=alpha_temp*first_stage_x_[n].get(j-1, i)
													+(alpha_temp)*(1-alpha_temp)*unknown_x_past_[n].get(0, i)
													+(alpha_temp)*pow((1-alpha_temp),2)*unknown_x_past_[n].get(1,i);
					second_stage_x_[n].set(j-1, i, exponential_temp);
					count+=first_stage_x_[n].n_cols();
				}	//j
				*/

			}	//i
			

			for(index_t i=0; i<exp_betas_times_x2_.size(); i++) {
			  exp_betas_times_x2_[i]=0;
				//sum_first_derivative_conditional_postpond_prob_[i].SetZero();
				//sum_second_derivative_conditional_postpond_prob_[i].SetZero();
		  }
		  

			for(index_t i=0; i<second_stage_x_[n].n_cols(); i++) {
				//exp_betas_times_x2_[n]=0;
				exp_betas_times_x2_[n]+=exp(la::Dot(betas.length(), 
																betas.ptr(),
																second_stage_x_[n].GetColumnPtr(i)));			
			}
			//cout<<"exp_betas_times_x2_a part1="<<exp_betas_times_x2_[0]<<endl;
			/*
			//Calculate x^2_{ni}(alpha_l)
			//int count=0;
			double j=ind_unknown_x_[0];

			for(index_t i=0; i<first_stage_x_[n].n_cols(); i++){
				
				exponential_temp=alpha_temp*first_stage_x_[n].get(j-1, i)
													+(alpha_temp)*(1-alpha_temp)*unknown_x_past_[n].get(0, i)
													+pow((1-alpha_temp),2)*unknown_x_past_[n].get(1,i);
				second_stage_x_[n].set(j-1, i, exponential_temp);
				//count+=first_stage_x_[n].n_cols();
				


			}	//i
			
			

			for(index_t i=0; i<second_stage_x_[n].n_cols(); i++) {
				exp_betas_times_x2_[n]+=exp(la::Dot(betas.length(), betas.ptr(),
											 second_stage_x_[n].GetColumnPtr(i) ));
			}	//i
			*/




			for(index_t i=0; i<second_stage_x_[n].n_cols(); i++) {
				//Calculate second_stage_dot_logit_
				second_stage_dot_logit_[n][i]=((exp(la::Dot(betas.length(), betas.ptr(),
																				 second_stage_x_[n].GetColumnPtr(i))))/
																				 exp_betas_times_x2_[n]);
																				 
				second_stage_ddot_logit_[n].set(i, i, second_stage_dot_logit_[n][i]);
			}	//i

			/*
			cout<<"second_stage_dot_logit[0]"<<endl;
			for(index_t i=0; i<second_stage_dot_logit_[0].length(); i++){
				cout<<second_stage_dot_logit_[0][i]<<" ";
			}
			cout<<endl;
			*/


		
			conditional_postponed_prob[n]=exp_betas_times_x2_[n]/(exp_betas_times_x1_[n]+exp_betas_times_x2_[n]);
			la::MulOverwrite(second_stage_x_[n], second_stage_dot_logit_[n], &temp2);
			la::SubOverwrite(temp1, temp2, &first_derivative_conditional_postpond_prob);
			/*
			cout<<"n="<<n<<"first_derivative_conditional_postpond_prob middle"<<endl;
			for(index_t i=0; i<first_derivative_conditional_postpond_prob.length(); i++){
				cout<<first_derivative_conditional_postpond_prob[i]<<" ";
			}
			cout<<endl;
			*/


			//Calculate SecondDerivativePostponedProb.
			//Matrix first_term_temp;
			//first_term_temp.Init(betas.length(), betas.length());

			
			////handle vector transpose				
			matrix_first_derivative_conditional_postpond_prob.Alias(first_derivative_conditional_postpond_prob.ptr(), 
																										first_derivative_conditional_postpond_prob.length(),
																										1);
			tmatrix_first_derivative_conditional_postpond_prob.Alias(first_derivative_conditional_postpond_prob.ptr(), 
																										1,
																										first_derivative_conditional_postpond_prob.length());



			la::MulOverwrite(matrix_first_derivative_conditional_postpond_prob, 
														 tmatrix_first_derivative_conditional_postpond_prob, 
														 &first_term_temp);
			
			
			la::Scale( (1-2*conditional_postponed_prob[n])*(conditional_postponed_prob[n])*(1-conditional_postponed_prob[n]),
								&first_term_temp);
			/*
			cout<<"n="<<n<<" "<<"first_term_temp"<<endl;
			for(index_t i=0; i<first_term_temp.n_rows(); i++){
					for(index_t j=0; j<first_term_temp.n_cols(); j++){
						cout<<first_term_temp.get(i,j)<<" ";
					}
					cout<<endl;
			}
			cout<<endl;
			
			*/


			

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

			/*
			if(n==0){
				cout<<"temp4"<<endl;
				for(index_t i=0; i<temp4.n_rows(); i++){
					for(index_t j=0; j<temp4.n_cols(); j++){
						cout<<temp4.get(i,j)<<" ";
					}
					cout<<endl;

				}
			}
			*/
			
			//Matrix temp5;	//dotX2(temp4)
			//temp5.Init(betas.length(), second_stage_x_[n].n_cols());

			//la::MulOverwrite(second_stage_dot_logit_[n], temp4, &temp5);
			la::MulOverwrite(second_stage_x_[n], temp4, &temp5);

			//Matrix temp6; //temp5*dotX2'
			//temp6.Init(betas.length(), betas.length());
			//la::MulTransBOverwrite(temp5, second_stage_dot_logit_[n], &temp6);
			la::MulTransBOverwrite(temp5, second_stage_x_[n], &temp6);
			/*
			if(n==0){
				cout<<"temp6"<<endl;
				for(index_t i=0; i<temp6.n_rows(); i++){
					for(index_t j=0; j<temp6.n_cols(); j++){
						cout<<temp6.get(i,j)<<" ";
					}
					cout<<endl;

				}
			}
			*/

			la::SubOverwrite(temp10, temp6, &second_term_temp);
			la::Scale( (conditional_postponed_prob[n])*(1-conditional_postponed_prob[n]), &second_term_temp);

			la::AddOverwrite(second_term_temp, first_term_temp, &second_derivative_conditional_postpond_prob);
			
			/*
			if(n==0){
				cout<<"first_term_temp"<<endl;
				for(index_t i=0; i<first_term_temp.n_rows(); i++){
					for(index_t j=0; j<first_term_temp.n_cols(); j++){
						cout<<first_term_temp.get(i,j)<<" ";
					}
					cout<<endl;

				}
			}
			*/

			/*
			if(n==0){
				

				cout<<"second_term_temp"<<endl;
				for(index_t i=0; i<second_term_temp.n_rows(); i++){
					for(index_t j=0; j<second_term_temp.n_cols(); j++){
						cout<<second_term_temp.get(i,j)<<" ";
					}
					cout<<endl;

				}
			}
			*/



			//end of calculation of second_derivative_conditional_postpond_prob
			
			//Scale with beta_function
			la::Scale( beta_function_temp, &second_derivative_conditional_postpond_prob );
			//check
			

			la::AddTo(second_derivative_conditional_postpond_prob, 
								&sum_second_derivative_conditional_postpond_prob_[n]);


			la::Scale( (conditional_postponed_prob[n])*(1-conditional_postponed_prob[n]), &first_derivative_conditional_postpond_prob);

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

		//test
		/*
		cout<<"first_derivative_conditional_postpond_prob"<<endl;
		for(index_t i=0; i<first_derivative_conditional_postpond_prob.length(); i++){
			cout<<first_derivative_conditional_postpond_prob[i]<<" ";
		}
		cout<<endl;
		*/

		

		la::Scale(alpha_weight_, &sum_first_derivative_conditional_postpond_prob_[n]);
		la::Scale(alpha_weight_, &sum_second_derivative_conditional_postpond_prob_[n]);
		

		if(n<first_stage_x_.size()-1) {
			matrix_first_stage_dot_logit.Destruct();
			tmatrix_first_stage_dot_logit.Destruct();
			
		}
		/*
		if(n==1){
			
			cout<<"temp7="<<temp7.get(1,1)<<endl;
			cout<<"temp8="<<temp8.get(1,1)<<endl;
			cout<<"temp9="<<temp9.get(1,1)<<endl;
			cout<<"temp3="<<temp3.get(1,1)<<endl;
			cout<<"temp4="<<temp4.get(1,1)<<endl;
			cout<<"temp5="<<temp5.get(1,1)<<endl;
			cout<<"temp6="<<temp6.get(1,1)<<endl;
			cout<<"temp10="<<temp10.get(1,1)<<endl;
			cout<<"temp1="<<temp1[1]<<endl;
			cout<<"temp2="<<temp2[1]<<endl;
			
		}
		*/



	}	//n
		//cout<<"test n end"<<endl;
	//cout<<"second_stage_ddot_logit="<<endl;
	//cout<<second_stage_ddot_logit_[1].get(1,1)<<endl;
	//cout<<"second_stage_x="<<second_stage_x_[1].get(1,1)<<endl;
	//cout<<"exp_betas_times_x2_="<<exp_betas_times_x2_[1]<<endl;
	//cout<<"conditional_postponed_prob="<<conditional_postponed_prob<<endl;
	//cout<<"beta_fn_temp="<<beta_function_temp<<endl;

	//cout<<"beta_fn_temp="<<beta_function_temp<<endl;
	
	/*
	cout<<"exp_betas_times_x2_[n]:"<<endl;
	for(index_t n=0; n<exp_betas_times_x2_.size(); n++){
		cout<<exp_betas_times_x2_[n]<<" ";
	}
	cout<<endl;

	cout<<"conditional_postponed_prob="<<endl;
	for(index_t n=0; n<conditional_postponed_prob.length(); n++){
		cout<<conditional_postponed_prob[n]<<" ";
	}
	cout<<endl;
	*/

  

  /*
	cout<<"sum_first_derivative_conditional_postpond_prob_[0]"<<endl;
	for(index_t i=0; i<sum_first_derivative_conditional_postpond_prob_[0].length(); i++){
		cout<<sum_first_derivative_conditional_postpond_prob_[0][i]<<" ";
	}
	cout<<endl;
	*/

	/*
	cout<<"sum_second_derivative_conditional_postpond_prob_[0]"<<endl;
	for(index_t i=0; i<sum_second_derivative_conditional_postpond_prob_[0].n_rows(); i++){
		for(index_t j=0; j<sum_second_derivative_conditional_postpond_prob_[0].n_cols(); j++){
			cout<<sum_second_derivative_conditional_postpond_prob_[0].get(i,j)<<" ";
		}
	cout<<endl;
	}
	*/


	



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
			la::AddTo(temp, &temp2);
			

		}	//if-else
		//la::AddTo(temp, &temp2);
		

	}	//n
	la::Scale(-1, &temp2);
	//return derivative_beta_term2;
	beta_term2->Copy(temp2);
	/*
	cout<<"derivative beta term2="<<endl;
	for(index_t i=0; i<num_of_betas_; i++){
		cout<<temp2[i]<<" ";
	}
	cout<<endl;

	
	cout<<"postponed_probability="<<endl;
	for(index_t i=0; i<5; i++){
		cout<<postponed_probability_[i]<<" ";
	}
	cout<<endl;
	
	cout<<"sum_first_derivative_conditional_postpond_prob_="<<endl;
	for(index_t i=0; i<sum_first_derivative_conditional_postpond_prob_[1].length(); i++){
		cout<<sum_first_derivative_conditional_postpond_prob_[1][i]<<" ";
	}
	cout<<endl;
	*/


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

	

	for(index_t n=0; n<first_stage_x_.size(); n++){
		  if (first_stage_y_[n]<0) { 
				continue;
			} else {
				/*
				cout<<"sum_second_derivative_conditional_postpond_prob_[n]"<<endl;
				for(index_t i=0; i<sum_second_derivative_conditional_postpond_prob_[n].n_rows(); i++){
					for(index_t j=0; j<sum_second_derivative_conditional_postpond_prob_[n].n_cols(); j++){
						cout<<sum_second_derivative_conditional_postpond_prob_[n].get(i,j)<<" ";
					}
					cout<<endl;
				}
				cout<<endl;
				*/



				la::ScaleOverwrite( (1/(1-postponed_probability_[n])), sum_second_derivative_conditional_postpond_prob_[n], &first_temp);

				//handle vector transpose

				Matrix matrix_sum_first_derivative_conditional_postpond_prob;
				//matrix_sum_first_derivative_conditional_postpond_prob.Init(num_of_betas_, 1);

				Matrix tmatrix_sum_first_derivative_conditional_postpond_prob;
				//tmatrix_sum_first_derivative_conditional_postpond_prob.Init(num_of_betas_, 1);

				
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

	/*
	cout<<"second_derivative_beta_term2:"<<endl;
	for(index_t i=0; i<second_derivative_beta_term2.n_rows(); i++){
		for(index_t j=0; j<second_derivative_beta_term2.n_cols(); j++){
			cout<<second_derivative_beta_term2.get(i,j)<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
	*/


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
		if (first_stage_y_[n]>0) {
      continue;
    } else {
			//check
			la::ScaleOverwrite( (1/postponed_probability_[n]), sum_first_derivative_conditional_postpond_prob_[n], &temp);
			//cout<<"postponed_probability_[n]="<<postponed_probability_[n]<<endl;
			//temp=SumFirstDerivativeConditionalPostpondProb_[n]/(postponed_probability_[n]);
			//check
			la::AddTo(temp, &temp2);


		}	//if-else
		//la::AddTo(temp, &temp2);

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

	
	for(index_t n=0; n<first_stage_x_.size(); n++){
		  if (first_stage_y_[n]>0) { 
				continue;
			} else {

				la::ScaleOverwrite( (1/(postponed_probability_[n])), sum_second_derivative_conditional_postpond_prob_[n], &first_temp);
				//la::Scale( (1/(postponed_probability_[n])), &first_temp);
	
				//handle vector transpose

				Matrix matrix_sum_first_derivative_conditional_postpond_prob;
				//matrix_sum_first_derivative_conditional_postpond_prob.Init(num_of_betas_,1);

				Matrix tmatrix_sum_first_derivative_conditional_postpond_prob;
				//tmatrix_sum_first_derivative_conditional_postpond_prob.Init(num_of_betas_,1);



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


				//la::Scale( 1/pow((1-postponed_probability_[n]), 2), &second_temp);
				la::Scale( 1/pow((postponed_probability_[n]), 2), &second_temp);

				//la::AddOverwrite(second_temp, first_temp, &second_derivative_beta_temp);
				la::SubOverwrite(second_temp, first_temp, &second_derivative_beta_temp);

				//check
				la::AddTo(second_derivative_beta_temp, &second_derivative_beta_term3);

							
			}	//else

	}	//n

	/*
	cout<<"second_derivative_beta_term3:"<<endl;
	for(index_t i=0; i<second_derivative_beta_term3.n_rows(); i++){
		for(index_t j=0; j<second_derivative_beta_term3.n_cols(); j++){
			cout<<second_derivative_beta_term3.get(i,j)<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
	*/

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
	//double conditional_postponed_prob=0;
	Vector conditional_postponed_prob;
	conditional_postponed_prob.Init(first_stage_x_.size());

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
		//cout<<"t_temp="<<t_temp<<endl;
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

	/*
	cout<<"beta_fn_temp1="<<beta_fn_temp1<<endl;
	cout<<"beta_fn_temp2="<<beta_fn_temp2<<endl;
	cout<<"beta_fn_temp3="<<beta_fn_temp3<<endl;
	cout<<"beta_fn_temp4="<<beta_fn_temp4<<endl;
	cout<<"beta_fn_temp5="<<beta_fn_temp5<<endl;
	cout<<"beta_fn_temp6="<<beta_fn_temp6<<endl;
	*/


	
	for(index_t i=0; i<first_stage_x_.size(); i++) {
		sum_first_derivative_p_beta_fn_[i]=0;
		sum_second_derivative_p_beta_fn_[i]=0;
		sum_first_derivative_q_beta_fn_[i]=0;
		sum_second_derivative_q_beta_fn_[i]=0;
		sum_second_derivative_p_q_beta_fn_[i]=0;
		//sum_second_derivative_conditionl_postponed_p_[i].Init(num_of_betas_);
		sum_second_derivative_conditionl_postponed_p_[i].SetZero();
		//sum_second_derivative_conditionl_postponed_q_[i].Init(num_of_betas_);
		sum_second_derivative_conditionl_postponed_q_[i].SetZero();
		conditional_postponed_prob[i]=0;

	}
	


	for(index_t n=0; n<first_stage_x_.size(); n++){
		//exp_betas_times_x2_[n]=0;
		

		la::MulOverwrite(first_stage_x_[n], first_stage_dot_logit_[n], &temp1);
		for(index_t l=0; l<num_of_alphas_-1; l++){
			alpha_temp=(l+1)*(alpha_weight_);

			//Calculate x^2_{ni}(alpha_l)
			//int count=0;
			double j=ind_unknown_x_[0];

			for(index_t i=0; i<first_stage_x_[n].n_cols(); i++){
				
				exponential_temp=alpha_temp*first_stage_x_[n].get(j-1, i)
													+(alpha_temp)*(1-alpha_temp)*unknown_x_past_[n].get(0, i)
													+pow((1-alpha_temp),2)*unknown_x_past_[n].get(1,i);
				second_stage_x_[n].set(j-1, i, exponential_temp);
				//count+=first_stage_x_[n].n_cols();
				

				/*for(index_t j=ind_unknown_x_[0]; j<=ind_unknown_x_[ind_unknown_x_.length()-1]; j++){
					exponential_temp=alpha_temp*first_stage_x_[n].get(j-1, i)
													+(alpha_temp)*(1-alpha_temp)*unknown_x_past_[n].get(0, i)
													+(alpha_temp)*pow((1-alpha_temp),2)*unknown_x_past_[n].get(1,i);
					second_stage_x_[n].set(j-1, i, exponential_temp);
					count+=first_stage_x_[n].n_cols();
				}	//j
				*/

			}	//i
			

			for(index_t i=0; i<exp_betas_times_x2_.size(); i++) {
			  exp_betas_times_x2_[i]=0;
		  }
		  

			for(index_t i=0; i<second_stage_x_[n].n_cols(); i++) {
				//exp_betas_times_x2_[n]=0;
				exp_betas_times_x2_[n]+=exp(la::Dot(betas.length(), 
																betas.ptr(),
																second_stage_x_[n].GetColumnPtr(i)));			
			}
	
			//cout<<"exp_betas_times_x2_a="<<exp_betas_times_x2_[0]<<endl;
			for(index_t i=0; i<second_stage_x_[n].n_cols(); i++) {
				//Calculate second_stage_dot_logit_
				second_stage_dot_logit_[n][i]=((exp(la::Dot(betas.length(), betas.ptr(),
																				 second_stage_x_[n].GetColumnPtr(i))))/
																				 exp_betas_times_x2_[n]);
																				 
				second_stage_ddot_logit_[n].set(i, i, first_stage_dot_logit_[n][i]);
			}	//i

			
			conditional_postponed_prob[n]=exp_betas_times_x2_[n]/(exp_betas_times_x1_[n]+exp_betas_times_x2_[n]);
			la::MulOverwrite(second_stage_x_[n], second_stage_dot_logit_[n], &temp2);
			la::SubOverwrite(temp1, temp2, &first_derivative_conditional_postpond_prob);
			la::Scale( (conditional_postponed_prob[n]*(1-conditional_postponed_prob[n])), &first_derivative_conditional_postpond_prob);
			/*
			cout<<"n="<<n<<"first_derivative_conditional_postpond_prob="<<endl;
			for(index_t i=0; i<first_derivative_conditional_postpond_prob.length(); i++){
				cout<<first_derivative_conditional_postpond_prob[i]<<" ";
			}
			cout<<endl;
			*/


			
			powtemp=pow(alpha_temp, p-1)*pow(1-alpha_temp, q-1);

			sum_first_derivative_p_beta_fn_[n]+=conditional_postponed_prob[n]
																					*( powtemp
																					*(log(alpha_temp)*beta_fn_temp1
																					- beta_fn_temp2) );
			sum_second_derivative_p_beta_fn_[n]+=conditional_postponed_prob[n]
																					 *( powtemp
																					 *( pow(log(alpha_temp), 2)*beta_fn_temp1
																					 -beta_fn_temp3
																					 -2*log(alpha_temp)*beta_fn_temp1*beta_fn_temp2*denumerator_beta_function_
																					 +2*pow(beta_fn_temp2,2)*denumerator_beta_function_));
																					 			
			sum_first_derivative_q_beta_fn_[n]+=conditional_postponed_prob[n]
																					*( powtemp
																					*(log(1-alpha_temp)*beta_fn_temp1
																					- beta_fn_temp4) );
			

			/*sum_second_derivative_q_beta_fn_[n]+=conditional_postponed_prob
																					 *( powtemp
																					 *( pow(log(1-alpha_temp), 2)*beta_fn_temp1
																					 -2*log(1-alpha_temp)*beta_fn_temp4
																					 +beta_fn_temp5));
			
			*/


			sum_second_derivative_q_beta_fn_[n]+=conditional_postponed_prob[n]
																					 *( powtemp
																					 *( pow( log(1-alpha_temp), 2)*beta_fn_temp1
																					 -beta_fn_temp5
																					 -2*log(1-alpha_temp)*beta_fn_temp1*beta_fn_temp4*denumerator_beta_function_
																					 +2*pow(beta_fn_temp4,2)*denumerator_beta_function_));
			

			sum_second_derivative_p_q_beta_fn_[n]+=conditional_postponed_prob[n]
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

	//cout<<"conditional_postponed_prob[0]="<<conditional_postponed_prob[0]<<endl;
	/*
	cout<<"sum_first_derivative_p_beta_fn_"<<endl;
	for(index_t i=0; i<sum_first_derivative_p_beta_fn_.size(); i++){
		cout<<sum_first_derivative_p_beta_fn_[i]<<" ";
	}
	cout<<endl;
	*/

	/*
	cout<<"sum_second_derivative_p_beta_fn_"<<endl;
	for(index_t i=0; i<sum_second_derivative_p_beta_fn_.size(); i++){
		cout<<sum_second_derivative_p_beta_fn_[i]<<" ";
	}
	cout<<endl;
  */

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

	//cout<<"derivative_p_term2="<<derivative_p_term2<<endl;
	
  return derivative_p_term2;
	
}



double Objective::ComputeDerivativePTerm3_() {
	double derivative_p_term3=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]>0) {
      continue;
    } else {
      derivative_p_term3+=(sum_first_derivative_p_beta_fn_[n]/(postponed_probability_[n]));
    }
  }
	//cout<<"derivative_p_term3="<<derivative_p_term3<<endl;
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

	//cout<<"second_derivative_p_term2="<<second_derivative_p_term2<<endl;
  return second_derivative_p_term2;
}



double Objective::ComputeSecondDerivativePTerm3_() {
	double second_derivative_p_term3=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]>0) {
      continue;
    } else {
      second_derivative_p_term3+=( sum_second_derivative_p_beta_fn_[n]/(postponed_probability_[n]))
																	 - pow( (sum_first_derivative_p_beta_fn_[n]/(postponed_probability_[n])), 2);
    }
  }
	//cout<<"second_derivative_p_term3="<<second_derivative_p_term3<<endl;
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
	//cout<<"derivative_q_term2="<<derivative_q_term2<<endl;
  return derivative_q_term2;
}


double Objective::ComputeDerivativeQTerm3_() {
	double derivative_q_term3=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]>0) {
      continue;
    } else {
      derivative_q_term3+=(sum_first_derivative_q_beta_fn_[n]/(postponed_probability_[n]));
    }
  }
	//cout<<"derivative_q_term3="<<derivative_q_term3<<endl;
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
	//cout<<"second_derivative_q_term2="<<second_derivative_q_term2<<endl;
  return second_derivative_q_term2;
}



double Objective::ComputeSecondDerivativeQTerm3_() {
	double second_derivative_q_term3=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]>0) {
      continue;
    } else {
      second_derivative_q_term3+=( sum_second_derivative_q_beta_fn_[n]/(postponed_probability_[n]))
																	 - pow( (sum_first_derivative_q_beta_fn_[n]/(postponed_probability_[n])), 2);
    }
  }
	//cout<<"second_derivative_q_term3="<<second_derivative_q_term3<<endl;
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
			temp1.SetZero();
			temp2.SetZero();
			temp3.SetZero();
			//temp1-first term
			la::ScaleOverwrite( (1.0/(1-postponed_probability_[n])), sum_second_derivative_conditionl_postponed_p_[n], &temp1);
			la::ScaleOverwrite( (sum_first_derivative_p_beta_fn_[n]/(pow((1-postponed_probability_[n]), 2))),
												 sum_first_derivative_conditional_postpond_prob_[n], &temp2);
			la::AddOverwrite( temp2, temp1, &temp3);
			
			/*
			cout<<"n="<<n<<" "<<"postponed_probability_[n]"<<postponed_probability_[n]<<endl;

			cout<<"temp2"<<endl;
			for(index_t i=0; i<temp2.length(); i++){
				cout<<temp2[i]<<" ";
			}
			cout<<endl;

			cout<<"temp1"<<endl;
			for(index_t i=0; i<temp1.length(); i++){
				cout<<temp1[i]<<" ";
			}
			cout<<endl;

			cout<<"n="<<n<<" "<<"sum_second_derivative_conditionl_postponed_p_"<<endl;
			for(index_t i=0; i<sum_second_derivative_conditionl_postponed_p_[n].length(); i++){
				cout<<sum_second_derivative_conditionl_postponed_p_[n][i]<<" ";
			}
			cout<<endl;
			*/




			la::AddTo(temp3, &second_derivative_p_beta_term2);
      //second_derivative_p_beta_term2+=(sum_first_derivative_q_beta_fn_[n]/(1-postponed_probability_[n]));
			/*
			cout<<"sum_second_derivative_conditionl_postponed_p_"<<endl;
			for(index_t i=0; i<sum_second_derivative_conditionl_postponed_p_[n].length(); i++){
				cout<<sum_second_derivative_conditionl_postponed_p_[n][i]<<" ";
			}
			cout<<endl;
			

			cout<<"second_derivative_p_beta_term2"<<endl;
			for(index_t i=0; i<second_derivative_p_beta_term2.length(); i++){
				cout<<second_derivative_p_beta_term2[i]<<" ";
			}
			cout<<endl;
			*/


		}
  }
	la::Scale(-1, &second_derivative_p_beta_term2);
	
	/*
	cout<<"second_derivative_p_beta_term2"<<endl;
	for(index_t i=0; i<second_derivative_p_beta_term2.length(); i++){
		cout<<second_derivative_p_beta_term2[i]<<" ";
	}
	cout<<endl;
	*/


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
    if (first_stage_y_[n]>0) {
      continue;
    } else {
			temp3.SetZero();
			temp1.SetZero();
			temp2.SetZero();
			//temp1-first term
			la::ScaleOverwrite((1/(postponed_probability_[n])), sum_second_derivative_conditionl_postponed_p_[n], &temp1);
			la::ScaleOverwrite( (sum_first_derivative_p_beta_fn_[n]/(pow((postponed_probability_[n]), 2))),
												 sum_first_derivative_conditional_postpond_prob_[n], &temp2);
			//la::AddOverwrite( temp2, temp1, &temp3);
			la::SubOverwrite(temp2, temp1, &temp3);

			la::AddTo(temp3, &second_derivative_p_beta_term3);
      //second_derivative_p_beta_term2+=(sum_first_derivative_q_beta_fn_[n]/(1-postponed_probability_[n]));
    }
  }

	/*
	cout<<"second_derivative_p_beta_term3"<<endl;
	for(index_t i=0; i<second_derivative_p_beta_term3.length(); i++){
		cout<<second_derivative_p_beta_term3[i]<<" ";
	}
	cout<<endl;
  */

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
			temp1.SetZero();
			temp2.SetZero();
			temp3.SetZero();

			la::ScaleOverwrite((1/(1-postponed_probability_[n])), sum_second_derivative_conditionl_postponed_q_[n], &temp1);
			la::ScaleOverwrite( (sum_first_derivative_q_beta_fn_[n]/(pow((1-postponed_probability_[n]), 2))),
												 sum_first_derivative_conditional_postpond_prob_[n], &temp2);
			la::AddOverwrite( temp2, temp1, &temp3);
			la::AddTo(temp3, &second_derivative_q_beta_term2);
      //second_derivative_p_beta_term2+=(sum_first_derivative_q_beta_fn_[n]/(1-postponed_probability_[n]));
    }
  }
	la::Scale(-1, &second_derivative_q_beta_term2);

	/*
	cout<<"second_derivative_q_beta_term2"<<endl;
	for(index_t i=0; i<second_derivative_q_beta_term2.length(); i++){
		cout<<second_derivative_q_beta_term2[i]<<" ";
	}
	cout<<endl;
  */


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
    if (first_stage_y_[n]>0) {
      continue;
    } else {
			temp1.SetZero();
			temp2.SetZero();
			temp3.SetZero();
			//temp1-first term
			la::ScaleOverwrite((1/(postponed_probability_[n])), sum_second_derivative_conditionl_postponed_q_[n], &temp1);
			la::ScaleOverwrite( (sum_first_derivative_q_beta_fn_[n]/(pow((postponed_probability_[n]), 2))),
												 sum_first_derivative_conditional_postpond_prob_[n], &temp2);
			//la::AddOverwrite( temp2, temp1, &temp3);
			la::SubOverwrite(temp2, temp1, &temp3);
			la::AddTo(temp3, &second_derivative_q_beta_term3);
      //second_derivative_p_beta_term2+=(sum_first_derivative_q_beta_fn_[n]/(1-postponed_probability_[n]));
    }
  }

	/*
	cout<<"second_derivative_q_beta_term3"<<endl;
	for(index_t i=0; i<second_derivative_q_beta_term3.length(); i++){
		cout<<second_derivative_q_beta_term3[i]<<" ";
	}
	cout<<endl;
  */

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

	
	//cout<<"second_derivative_p_q_term2="<<second_derivative_p_q_term2<<endl;
  return second_derivative_p_q_term2;
}


double Objective::ComputeSecondDerivativePQTerm3_() {
	double second_derivative_p_q_term3=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]>0) {
      continue;
    } else {
      second_derivative_p_q_term3+=( sum_second_derivative_p_q_beta_fn_[n]/(postponed_probability_[n]))
																		- ((sum_first_derivative_p_beta_fn_[n]/(postponed_probability_[n]))
																		* (sum_first_derivative_q_beta_fn_[n]/(postponed_probability_[n])));
    }
  }

	//cout<<"second_derivative_p_q_term3="<<second_derivative_p_q_term3<<endl;
  return second_derivative_p_q_term3;
}


//////////////////////////////////////////////////
///Finite differene derivative approximation
//////////////////////////////////////////////////
void Objective::CheckGradient(double current_sample,
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



	


void Objective::CheckHessian(double current_sample,
														 Vector &current_parameter, 
														 Matrix *approx_hessian){


//approx_hessian(i,j)=(gradi(x+epsilon*ej)-gradi(x-epsilon*ej))/(4*epsilon)
//									+(gradj(x+epsilon*ei)-gradj(x-epsilon*ei))/(4*epsilon)
//OR


	Vector ei;
	ei.Init(num_of_betas_+2);

	Vector ej;
	ej.Init(num_of_betas_+2);


	Vector u1i; //x+epsillon*ei
	u1i.Init(num_of_betas_+2);
	u1i.SetZero();

	Vector u2i; //x-epsillon*ei
	u2i.Init(num_of_betas_+2);
  u2i.SetZero();

	Vector u1j; //x+epsillon*ej
	u1j.Init(num_of_betas_+2);
	u1j.SetZero();

	Vector u2j; //x-epsillon*ej
	u2j.Init(num_of_betas_+2);
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
	dummy_approx_hessian.Init(num_of_betas_+2, num_of_betas_+2);
	dummy_approx_hessian.SetZero();

	for(index_t i=0; i<(num_of_betas_+2); i++){
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
		

		for(index_t j=0; j<(num_of_betas_+2); j++){
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
	for(index_t i=0; i<(num_of_betas_+2); i++){
		for(index_t j=0; j<(num_of_betas_+2); j++){

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

void Objective::CheckHessian3(double current_sample,
														 Vector &current_parameter, 
														 Matrix *approx_hessian){


	

	Matrix dummy_approx_hessian;
	dummy_approx_hessian.Init(num_of_betas_+2, num_of_betas_+2);
	dummy_approx_hessian.SetZero();

	for(index_t i=0; i<(num_of_betas_+2); i++){
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
 
void Objective::CheckHessian2(double current_sample, 
									  Vector &current_parameter, 
										Matrix *approx_hessian) {

//(i,j)=(f(x+epsilon*ei+epsilon*ej)-f(x+epsilon*ei)-f(x+epsilon*ej)+f(x))/(epsilon^2)
//(i,j)=(f(x+epsilon*ei+epsilon*ej)-f(x+epsilon*ei-epsilon*ej)
//															 -f(x-epsilon*ei+epsilon*ej)+f(x-epsilon*ei-epsilon*ej))/(4*epsilon^2)

	Vector ei;
	ei.Init(num_of_betas_+2);

	Vector ej;
	ej.Init(num_of_betas_+2);


	Vector u1i; //x+epsillon*ei
	u1i.Init(num_of_betas_+2);
	u1i.SetZero();

	Vector u1j; //x+epsillon*ej
	u1j.Init(num_of_betas_+2);
	u1j.SetZero();

	Vector uij; //x+epsillon*ei+epsilon*ej
	uij.Init(num_of_betas_+2);
  uij.SetZero();

	double epsilon;
	//epsilon=sqrt(1e-8);
	epsilon=1e-4;

		
	Matrix dummy_approx_hessian;
	dummy_approx_hessian.Init(num_of_betas_+2, num_of_betas_+2);
	dummy_approx_hessian.SetZero();

	for(index_t i=0; i<(num_of_betas_+2); i++){
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
		
		for(index_t j=0; j<(num_of_betas_+2); j++){
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
	for(index_t i=0; i<(num_of_betas_+2); i++){
		for(index_t j=0; j<(num_of_betas_+2); j++){

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





void Objective::ComputePredictionError(double current_sample, 
									  Vector &current_parameter,
										ArrayList<index_t> &true_decision,
										double *postponed_prediction_error,
										double *choice_prediction_error){



	Vector betas;
  betas.Alias(current_parameter.ptr(), num_of_betas_);
  double p=current_parameter[num_of_betas_];
	double q=current_parameter[num_of_betas_+1];

	ComputeExpBetasTimesX1_(betas);

  ComputeDeumeratorBetaFunction_(p, q);
	
  ComputePostponedProbability_(betas, 
                               p, 
                               q);

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
		if(postponed_probability_[n]>=0.5) {
			predicted_decision[n]=-1;
			predicted_choice_probability[n]=postponed_probability_[n];
		} else {
			for(index_t i=0; i<first_stage_x_[n].n_cols(); i++){
				Vector temp;
				first_stage_x_[n].MakeColumnVector(i, &temp);
				predicted_choice_probability_all[n][i]=
					exp(la::Dot(betas, temp))/exp_betas_times_x1_[n]
					*(1-postponed_probability_[n]);
			}
			//find the alternative with maximum choice_prob.
			index_t index_chosen_alternative=0;
			double max_choce_probability=-100;
			for(index_t i=0; i<first_stage_x_[n].n_cols(); i++){
				if(postponed_probability_[n]<0.5){
					if(max_choce_probability<predicted_choice_probability_all[n][i]){
						index_chosen_alternative=i;
						//cout<<"n="<<n<<"index_chosen_alternative"<<index_chosen_alternative<<endl;
						max_choce_probability=predicted_choice_probability_all[n][i];
					}
				}
	
			}
			predicted_decision[n]=index_chosen_alternative+1;
			predicted_choice_probability[n]=max_choce_probability;
		}
	}

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
	data::Save("main_predicted_choice_probability.csv",  mtx_predicted_choice_probability);

	cout<<"Save predicted decision..."<<endl;
	data::Save("main_predicted_decision.csv", mtx_predicted_decision);




	//error calculation
	double temp_postponed_prediction_error=0;
	double temp_choice_prediction_error=0;

	
	double count_correct_postponed=0;
	double count_correct_choice=0;

	for(index_t n=0; n<number_of_test; n++){
		//if(true_decision[n]==-1 &&predicted_decision[n]==-1){
	  if(true_decision[n]*predicted_decision[n]>0){
			count_correct_postponed=count_correct_postponed+1;
			//temp_postponed_prediction_error_prob;

		}
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
			predicted_prob_of_n_choose_i[n]=postponed_probability_[n];
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
		if(true_decision[n]<0) {
			temp_choice_prediction_error_prob+=(1-postponed_probability_[n]);
		}
		else {
			temp_choice_prediction_error_prob+=(1-predicted_prob_of_n_choose_i[n]);
		}

		temp_postpone_prediction_error_prob+=(1-postponed_probability_[n]);
	}
	temp_choice_prediction_error_prob*=2;
	temp_postpone_prediction_error_prob*=2;

	cout<<"choice_prediction_error_prob_rate="<<temp_choice_prediction_error_prob/number_of_test<<endl;
	cout<<"postpone_prediction_error_prob_rate="<<temp_postpone_prediction_error_prob/number_of_test<<endl;
	(*postponed_prediction_error)=temp_postponed_prediction_error;
	(*choice_prediction_error)=temp_choice_prediction_error;

	/*
	cout<<"predicted_decision"<<endl;
	for(index_t n=0; n<number_of_test; n++){
		cout<<predicted_decision[n]<<" ";
	}
	cout<<endl;
	*/






}










