#include "fastlib/fastlib.h"
#include "utils.h"
#include "ocas_smo.h"

void OCASSMO::get_primal_solution(Vector &return_val){
  
  // Asume the vector has already been initialized
  return_val.CopyValues(primal_solution_);
}

void OCASSMO::get_dual_solution(Vector &return_val){
  
  // Assume vector has already been initialized
  return_val.CopyValues(alpha_vec_);
}

//////////////*** SMO OPTIMIZATION PROBLEM//////////////////////////

/** This simply resets the position of the variable that we are
 *  currently examining. So that we recalculate its position.
 */

void OCASSMO::ResetPositions_(){

  position_of_i_in_I0_=-1;
  position_of_i_in_I1_=-1;
  position_of_i_in_I2_=-1;

  position_of_j_in_I0_=-1;
  position_of_j_in_I1_=-1;
  position_of_j_in_I2_=-1;
}


void OCASSMO::DeleteFromI0_(int position_of_variable_in_I0,int position){

  if(I0_indices_[position_of_variable_in_I0]!=position){
    
    printf("There is a mistake here4...\n");
    exit(0);
  }
  //printf("removing from I0, position of variable in I0 is %d..\n",position_of_variable_in_I0);
  I0_indices_.Remove(position_of_variable_in_I0,1);
}

void OCASSMO::DeleteFromI1_(int position_of_variable_in_I1,int position){
  
  
  if(I1_indices_[position_of_variable_in_I1]!=position){

    printf("There is a mistake here5...\n");
    exit(0);
  }
  I1_indices_.Remove(position_of_variable_in_I1,1);

}

/** This function is called if a variable initially in I0 after update
 * is either in I1 or in I2
 */

void OCASSMO::DeleteFromFiForI0_(int position_of_variable_in_I0){

  /*printf("Before deleting from Fi...\n");
  for(int i=0;i<I0_indices_.size();i++){
    
  printf("I0 position is %d..\n",I0_indices_[i]);
  }*/

  //printf("Position of variable in I0 is %d...\n",position_of_variable_in_I0);

  // printf("Will delete from Fi..\n");

  
  Fi_for_I0_.Remove(position_of_variable_in_I0,1);
  
  /*printf("After deleting from Fi in the function we have...\n");

  for(int i=0;i<I0_indices_.size();i++){
    
    printf("I0 position is %d..\n",I0_indices_[i]);
    }*/
  
}

void OCASSMO::DeleteFromI2_(int position_of_variable_in_I2,int position){

  
  if(I2_indices_[position_of_variable_in_I2]!=position){

    printf("There is a mistake here6...\n");
    exit(0);
  }
  I2_indices_.Remove(position_of_variable_in_I2,1);
}

void OCASSMO::AddToI0_(int position){

  /*  printf("Before I add an element to I0...\n");
  for(int i=0;i<I0_indices_.size();i++){
    
    printf("I0 position is %d..\n",I0_indices_[i]);
    }*/

  I0_indices_.PushBack(1);
  I0_indices_[I0_indices_.size()-1]=position;
  
  /*printf("After I have added an element to I0...\n");
  for(int i=0;i<I0_indices_.size();i++){
    
    printf("I0 position is %d..\n",I0_indices_[i]);
    }*/
}

void OCASSMO::AddToFiForI0_(double F_val){
  
  Fi_for_I0_.PushBack(1);
  Fi_for_I0_[Fi_for_I0_.size()-1]=F_val;
  
}

void OCASSMO::AddToI1_(int position){
  
  I1_indices_.PushBack(1);
  I1_indices_[I1_indices_.size()-1]=position;

}

void OCASSMO::AddToI2_(int position){

  I2_indices_.PushBack(1);
  I2_indices_[I2_indices_.size()-1]=position;

}



int OCASSMO::CheckIfInI1_(int position){
  
  // lets check if this position(w.r.t subgradient matrix) is in I0
  
  if(I1_indices_.size()==0){
    
   
    
    // This means there are no elements. Hence return -1
    return -1;
  }
  for(index_t i=0;i<I1_indices_.size();i++){
    
    if(I1_indices_[i]==position){
      
      // This element exists in I1. Hence return the 
      // position of the element in I1
      return i;
    }
  }
  return -1;
}

int OCASSMO::CheckIfInI2_(int position){
  
  // lets check if this position(w.r.t subgradient matrix) is in I0
 
  if(I2_indices_.size()==0){  

    // This means there are no elements. Hence return -1
    return -1;
  }
  for(index_t i=0;i<I2_indices_.size();i++){
    
    if(I2_indices_[i]==position){
      
      // This element exists in I2. Hence return the 
      // position of the element in I2
      return i;
    }
  }
  return -1;
}


/** This function simply checks if a certain element given by the
 * position w.r.t subgradient matrix is actually in I0 or not 
 */

int OCASSMO::CheckIfInI0_(int position){
 
  
  // lets check if this position(w.r.t subgradient matrix) is in I0
  
  if(I0_indices_.size()==0){
    
    // This means there are no elements. Hence return -1
    return -1;
  }
  for(index_t i=0;i<I0_indices_.size();i++){
    
    if(I0_indices_[i]==position){
      
      // This element exists in I0. Hence return the 
      // position of the element in I0
      return i;
    }
  }
  return -1;
}






// Remember this function might be updating certain elements that are
// no longer in I0, but that doesn't matter as we follow this routine
// by UpdateSets routine

void OCASSMO::UpdateFiForI0_(double alpha_i_new,double alpha_j_new){

  /*printf("In update Fi function. We have I0 as...\n");
  for(int i=0;i<I0_indices_.size();i++){
    
    printf("I0 position is %d..\n",I0_indices_[i]);
    }*/

  // Subtract off old contributions and add new contributions
  
  Vector a_i,a_j;
  a_i.Alias(subgradients_mat_[index_working_set_variables_1_]);
  a_j.Alias(subgradients_mat_[index_working_set_variables_2_]);

  double alpha_i_old=alpha_vec_[index_working_set_variables_1_];
  double alpha_j_old=alpha_vec_[index_working_set_variables_2_];
  
  //printf("Size of I0 indices is %d...\n",I0_indices_.size());

  /*for(int i=0;i<I0_indices_.size();i++){
    
    printf("I0 position is %d..\n",I0_indices_[i]);
    }*/
 

  /*for(int z=0;z<I0_indices_.size();z++){
    
    printf("I0 position is %d..\n",I0_indices_[z]);
  }
  printf("The size of Fi vector is %d...\n",Fi_for_I0_.size());*/

  //ArrayList<double> temp;
  //temp.Init(Fi_for_I0_.size());

  for(index_t z=0;z<Fi_for_I0_.size();z++){
   
    Vector a_z;
    int pos_in_subgrad_matrix=I0_indices_[z];
   
    // printf("z=%d,I0 holds the following position=%d..\n",z,pos_in_subgrad_matrix);
    a_z.Alias(subgradients_mat_[pos_in_subgrad_matrix]);
    
    double F_val=Fi_for_I0_[z];
    F_val-=
      (la::Dot(a_z,a_i)*alpha_i_old+la::Dot(a_z,a_j)*alpha_j_old)/
      lambda_reg_const_;

    F_val+=(la::Dot(a_z,a_i)*alpha_i_new+la::Dot(a_z,a_j)*alpha_j_new)/
      lambda_reg_const_;


    /*printf("Before updating Fi for I0 we have...\n");
    for(int z=0;z<I0_indices_.size();z++){
      
      printf("I0 position is %d..\n",I0_indices_[z]);
      }*/
    // Store it back.  Remember to uncomment it out.
    Fi_for_I0_[z]=F_val;

    /*printf("After one round of iteration we have...\n");
    for(int z=0;z<I0_indices_.size();z++){
      
      printf("I0 position is %d..\n",I0_indices_[z]);
      }*/
  }  
}

void OCASSMO::UpdateSets_(double new_value, int position_I0,int position_I1,
			  int position_I2, int position, int which_variable){
  
  // The argument which_variable tells us the variable for which we
  // are making these set updates
  // Remember we have the positions the
  // working set variable before the updates in the sets I0,I1,I2

  // This flag will tell us from which set the element was deleted from


  /*printf("position I0:%d...\n",position_I0);
  printf("position I1:%d...\n",position_I1);
  printf("position I2:%d...\n",position_I2);
  printf("position:%d...\n",position);
  printf("new value=%f..\n",new_value);*/

  // Delete flag will tell us from which set we need to delete an
  // element, during the set update process.
  int delete_flag=-1;
  
  if(fabs(new_value)<SMALL){
    // This implies the new value is 0. That is it should be in the
    // set I2
    
    if(position_I0!=-1){
      
      //This means this element was originally in I0
      
      // First delete from I0 and then add it to I2. We need to pass
      // in the index in I0 from where we should delete and the
      // original index of this variable w.r.t the subgradient matrix
      
      //printf("will delete from I0..\n");
      DeleteFromI0_(position_I0,position);
      // Since Fi values are the cache values, hence delete it from
      // the cache too
      

      /*printf("Having deleted from I0...\n");
      for(int i=0;i<I0_indices_.size();i++){
	
      printf("I0 position is %d..\n",I0_indices_[i]);
      }*/
      
      DeleteFromFiForI0_(position_I0);

      /*printf("Having deleted from Fi...\n");
      for(int i=0;i<I0_indices_.size();i++){
	
	printf("I0 position is %d..\n",I0_indices_[i]);
	}*/

      delete_flag=0;
      
      //Finally add this element to I2
      AddToI2_(position);
    }
    else{

      // Check if this element was originally in I1
      if(position_I1!=-1){   
	
	//So we delete from I1 and add it to I2
	DeleteFromI1_(position_I1,position);

	//Finally add this element to I2
	AddToI2_(position);
	
	delete_flag=1;
      }
      else{
	
	// this element was originally in I2. 
	// In this case dont do anything

	if(position_I2==-1){

	  printf("There is a mistake here16..\n");
	  exit(0);
	}
	
      }
    }     
  }
  
  else{
    
    if(fabs(1-new_value)<SMALL){
      // This means the new value is =1.
      if(position_I0!=-1){
	
	//This means this element was originally in I0
	
	// First delete from I0 and then add it to I1. We need to pass
	// in the index in I0 from where we should delete and the
	// original index of this variable w.r.t the sungradient matrix
	
	DeleteFromI0_(position_I0,position);
	DeleteFromFiForI0_(position_I0);

	delete_flag=0;
	// Finally add to I1
	AddToI1_(position);
      }
      else{
	
	// Check if this element was originally in I2
	if(position_I2!=-1){
	  
	  //So we delete from I2 and add it to I1
	  DeleteFromI2_(position_I2,position);

	  delete_flag=2;

	  
	  // Finally add to I1
	  AddToI1_(position);
	}
	else{ // This element was originally in I1.  Hence nothing
	      // needs to be done

	  if(position_I1==-1){
	    
	    printf("There is a mistake here17..\n");
	    exit(0);
	  }
	}
      }
    }
    else{  // The new value is strictly between 0 and 1

      /*printf("Before I verify how to update the sets we have  I0 as...\n");
      for(int i=0;i<I0_indices_.size();i++){
	
	printf("I0 position is %d..\n",I0_indices_[i]);
	}*/
     
      if(position_I0!=-1){
	
	// Previously the element was in I0. Also since the cache was
	// updated before updating sets, we have nothing more to do
      }
      else{
	// Check if this element was in I1	

	if(position_I1!=-1){
	  
	  // This means previously the variable was in I1
	  DeleteFromI1_(position_I1,position);
	  delete_flag=1;

	  //printf("Adding to I0 the position=%d*********************\n",position);
	  AddToI0_(position);
	}
	else{
	  // This element was in I2. hence delete from I2
	  
	  if(position_I2==-1){
	    
	    printf("There was a mistake here3..........\n");
	    exit(0);
	  }
	  DeleteFromI2_(position_I2,position);
	  delete_flag=2;

	  //printf("Adding to I0 the position=%d*********************\n",position);
	  AddToI0_(position);

	}
       
	if(which_variable==1){
	  
	  AddToFiForI0_(F_working_set_variables_1_);
	}
	else{

	  AddToFiForI0_(F_working_set_variables_2_);
	}

	/*printf("After adding an element to I0 we have...\n");
	for(int i=0;i<I0_indices_.size();i++){
	  
	  printf("I0 position is %d..\n",I0_indices_[i]);
	  }*/
      }
    }
  }
  
  if(delete_flag==1){
    // The element was deleted from I1

    if(which_variable==1){
      
      if(position_of_j_in_I1_!=-1&&
	 position_of_i_in_I1_<position_of_j_in_I1_){
	
	// j is behind i in I1
	position_of_j_in_I1_--;
      }
      
      else{
	//Dont care
	
      }
    }
  }
  else{
    if(delete_flag==2){
      // The element was deleted from I2
      
      if(which_variable==1){
      
	if(position_of_j_in_I2_!=-1&&
	   position_of_i_in_I2_<position_of_j_in_I2_){
	  
	  position_of_j_in_I2_--;
	}
	else{
	  //Dont care
	  
	}
      }
    }
    else{
      if(delete_flag==0){
	if(which_variable==1){
	
	  if(position_of_j_in_I0_!=-1&&
	     position_of_i_in_I0_<position_of_j_in_I0_){

	    // j is behind i	    
	    position_of_j_in_I0_--;
	  }
	  else{
	  //Dont care
	    
	  }
	}
      }
    }
  }
  
  // The reason why we dont do anything if which_variable=2 is because
  // after updating sets we never ever again use the position of
  // working set variables.
}



void OCASSMO::UpdateFiValuesOfWorkingSetVariables_(double alpha_i_new,
						   double alpha_j_new){
  
  Vector a_i,a_j;
  a_i.Alias(subgradients_mat_[index_working_set_variables_1_]);
  a_j.Alias(subgradients_mat_[index_working_set_variables_2_]);
  
  double alpha_i_old=alpha_vec_[index_working_set_variables_1_];
  double alpha_j_old=alpha_vec_[index_working_set_variables_2_];

  
  
  double F_val=F_working_set_variables_1_;
  F_val-=
    (la::Dot(a_i,a_i)*alpha_i_old+la::Dot(a_i,a_j)*alpha_j_old)/
    lambda_reg_const_;
  
  F_val+=(la::Dot(a_i,a_i)*alpha_i_new+la::Dot(a_i,a_j)*alpha_j_new)/
    lambda_reg_const_;
  
  // Store it back
  F_working_set_variables_1_=F_val;
  
  
  F_val=F_working_set_variables_2_;
  F_val-=
    (la::Dot(a_j,a_i)*alpha_i_old+la::Dot(a_j,a_j)*alpha_j_old)/
    lambda_reg_const_;
  F_val+=(la::Dot(a_j,a_i)*alpha_i_new+la::Dot(a_j,a_j)*alpha_j_new)/
    lambda_reg_const_;
  
  // Store it back
  F_working_set_variables_2_=F_val;

}




/** This function calculates beta_up and beta_low using elements in I0 and
 *  the working set variables only.
 */

void OCASSMO::UpdateBetaUpAndBetaLowUsingI0_(double alpha_i_new_value, 
					     double alpha_j_new_value){
  
  // Iterate over all elements in I0


  // Reset beta_up and beta_low and i_up and i_low
  beta_up_=DBL_MAX;
  beta_low_=-DBL_MAX;

  i_up_=-1;
  i_low_=-1;

  // beta_up=min{F_i| i\in I0 \or I2}
  
  for(int i=0;i<I0_indices_.size();i++){
   
    if(beta_up_>Fi_for_I0_[i]){
      
      beta_up_=Fi_for_I0_[i];
      i_up_=I0_indices_[i];
    } 

    if(beta_low_<Fi_for_I0_[i]){
      
      beta_low_=Fi_for_I0_[i];
      i_low_=I0_indices_[i];
    }
  }

  /*printf("After using I0 only...\n");
  printf("i_low=%d..\n",i_low_);
  printf("i_up=%d..\n",i_up_);*/
  
  // Take a look at the working_set_variable_1_
  
  if(fabs(alpha_i_new_value)<SMALL){
    
    // This implies the working set variable is now in I2.
    
    if(beta_up_>F_working_set_variables_1_){
      
      beta_up_=F_working_set_variables_1_;
      i_up_=index_working_set_variables_1_;
    }     
  }
  else{
    if(fabs(1-alpha_i_new_value)<SMALL){
      
      // This implies this working set variable is in I1.
      
      if(beta_low_<F_working_set_variables_1_){
	
	beta_low_=F_working_set_variables_1_;
	i_low_=index_working_set_variables_1_;
      }
    }
    else{
      
      //This implies the working set variable is now in I0

      if(beta_up_>F_working_set_variables_1_){
	
	beta_up_=F_working_set_variables_1_;
	i_up_=index_working_set_variables_1_;
      }

      if(beta_low_<F_working_set_variables_1_){
	
	beta_low_=F_working_set_variables_1_;
	i_low_=index_working_set_variables_1_;
      }
    }
  }
  
  // Working set variable 2
  
  if(fabs(alpha_j_new_value)<SMALL){
    
    // This implies the working set variable is now in I2.
    
    if(beta_up_>F_working_set_variables_2_){
      
      beta_up_=F_working_set_variables_2_;
      i_up_=index_working_set_variables_2_;
    }
  }
  else{
    if(fabs(1-alpha_j_new_value)<SMALL){
      
      // This implies this working set variable is in I1.
      
      if(beta_low_<F_working_set_variables_2_){
	
	beta_low_=F_working_set_variables_2_;
	i_low_=index_working_set_variables_2_;
      }
    }
    else{
      
      //This implies the working set variable is now in I0
      
      if(beta_up_>F_working_set_variables_2_){
	
	beta_up_=F_working_set_variables_2_;
	i_up_=index_working_set_variables_2_;
      }
      if(beta_low_<F_working_set_variables_2_){
	
	beta_low_=F_working_set_variables_2_;
	i_low_=index_working_set_variables_2_;
      }
    }
  }
  /*printf("After updates we have...\n");
  printf("i_up=%d...\n",i_up_);
  printf("i_low=%d...\n",i_low_);*/
}




/** This updates the values of beta_hi and beta_low value using the
  * calculated F_value of the element whose position in the
  * subgradient matrix is given. Remember beta-up and beta_low are
  * simply the values F_up and F_low
  */


void OCASSMO::UpdateBetaUpAndBetaLow_(double F_val,int position){
  
  if(position_of_j_in_I2_!=-1&&F_val<beta_up_){

    // We have found an element in 
    // I2 with a smaller beta value
    
    beta_up_=F_val;
    i_up_=position;
    return;
  }  
  
  if(position_of_j_in_I1_!=-1&& F_val>beta_low_){
    
    
    //We have found an element which has a better F-value than the
    //present beta_low
    
    beta_low_=F_val;
    i_low_=position;
  }
  
   if(position_of_j_in_I0_!=-1&&F_val<beta_up_){

    // We have found an element in 
    // I2 with a smaller beta value
    
    beta_up_=F_val;
    i_up_=position;
   }  
    
   if(position_of_j_in_I0_!=-1&& F_val>beta_low_){
     
     
     //We have found an element which has a better F-value than the
     //present beta_low
     
     beta_low_=F_val;
     i_low_=position;
   }
}

/** Fi value needs to be calculate from scratch as it is not available
 * in the cache
 */

double OCASSMO::ComputeFiValue_(int position){
  
  // As usual this position is w.r.t the subgradient matrix.
    
  //F_i=\frac{1}{\lambda} \sum_{j=1}^{t+1} <a_i,a_j>\alpha_j -b_i

  //printf("The position is %d and alpha value is %f..\n",position,alpha_vec_[position]);
  
  Vector subgrad_i;
  subgrad_i.Alias(subgradients_mat_[position]);

  double sum=0;

  for (int j=0;j<num_subgradients_available_;j++){

    Vector subgrad_j;

    subgrad_j.Alias(subgradients_mat_[j]);
    // Calculate <a_i,a_j>

    double a_i_dot_a_j=la::Dot(subgrad_i,subgrad_j);
    sum+=a_i_dot_a_j*alpha_vec_[j];
  }

  // Divide the sum by \lambda and subtract b_i
  
  double val=(sum/lambda_reg_const_)-intercepts_vec_[position];
  return val;
}



//  We now have the indices of working set variables

int OCASSMO::TakeStep_(){

    
  if(index_working_set_variables_1_==index_working_set_variables_2_){

    return 0;
  }

  // \gamma1=\frac{1}{2\lambda} <a_i,a_i>

  Vector a_i=subgradients_mat_[index_working_set_variables_1_];
  double gamma1=la::Dot(a_i,a_i)/(2*lambda_reg_const_);
  
  //\gamma2=\frac{1}{2\lambda} <a_j,a_j>

  Vector a_j=subgradients_mat_[index_working_set_variables_2_];
  double gamma2=la::Dot(a_j,a_j)/(2*lambda_reg_const_);


  // gamma5=\frac{1}{\lambda} <a_i,a_j>
  double gamma5=la::Dot(a_i,a_j)/lambda_reg_const_;

  
  // \gamma3=\frac{1}{2\lambda} \sum_{l\neq i,j} \alpha_l <a_l,a_i> -b_i

  // Alternatively gamma3 and gamma4 can be expressed in terms of F values

  // double gamma3=(F_working_set_variables_1_-intercepts_vec_[index_working_set_variables_1_])/2.0;
  //gamma3-=(alpha_vec_[index_working_set_variables_1_]*gamma1+
  //   0.5*alpha_vec_[index_working_set_variables_2_]*gamma5);
  

  double gamma3=F_working_set_variables_1_-
  (2*gamma1*alpha_vec_[index_working_set_variables_1_]+
   gamma5*alpha_vec_[index_working_set_variables_2_]);
  

  //double gamma4=(F_working_set_variables_2_-intercepts_vec_[index_working_set_variables_2_])/2.0;
  //gamma4-=(alpha_vec_[index_working_set_variables_2_]*gamma2+
  //   0.5*alpha_vec_[index_working_set_variables_1_]*gamma5);
    
  
  double gamma4=F_working_set_variables_2_-
    (2*alpha_vec_[index_working_set_variables_2_]*gamma2+
     alpha_vec_[index_working_set_variables_1_]*gamma5);
  
  // gamma6=1-\sum_{l\neq i,j}\alpha_l=alpha_i+alpha_j [This is
  // because \sum alpha_l=1]
  
 
  double gamma6=alpha_vec_[index_working_set_variables_1_]+
    alpha_vec_[index_working_set_variables_2_];
 
  // \chi=2\gamma_1+2\gamma_2-2\gamma5

  double chi=2*gamma1+ 2*gamma2- 2*gamma5;

  //\psi =gamma4-gamma3+2gamma6gamma2-gamma6gamma5
  
  double psi=gamma4-gamma3+2*gamma6*gamma2-gamma6*gamma5;
 
  // L=max(0,gamma6-1)

  double L=max(0.0,gamma6-1);
  double H=min(1.0,gamma6);

  //alpha_i_new=min(max(L,\xi^-1 \chi),H)
  
  double alpha_i_new;
  double alpha_j_new;
  if(chi>0){
    alpha_i_new=min(max(L,psi/chi),H);
    alpha_j_new=gamma6-alpha_i_new; 
  }
  else{
    
    if(psi>0){

      alpha_i_new=H;
      alpha_j_new=gamma6-alpha_i_new;
    }
    else{
      
      alpha_i_new=L;
      alpha_j_new=gamma6-alpha_i_new;
    }
  }
  
  //Check if enough progress was made or not
  if(fabs(alpha_j_new-alpha_vec_[index_working_set_variables_2_]) < 
     eps_*(alpha_j_new+alpha_vec_[index_working_set_variables_2_]+eps_)){

    return 0;
  }

  //  printf("Previously we had ...\n");
  //alpha_vec_.PrintDebug();
 
  

  // Since alpha values have changed, modify the F values for $i\in
  // I0$ 
  
  /*printf("Before updating Fi's we have ...\n");

  for(int i=0;i<I0_indices_.size();i++){
    
    printf("I0 position is %d..\n",I0_indices_[i]);
    }*/

  //printf("Will now update Fi's...\n");

  UpdateFiForI0_(alpha_i_new,alpha_j_new);

  // Now compute the updated F values for the variables involved in
  // TakeStep. Remember these variables need not be in I0.
  // This is important to be done before set updates
  
  UpdateFiValuesOfWorkingSetVariables_(alpha_i_new,alpha_j_new);


  /*printf("Before updating I0 for first set we have...\n");
  for(int i=0;i<I0_indices_.size();i++){
    
    printf("I0 position is %d..\n",I0_indices_[i]);
    }*/

  // Update sets for working set variable 1
  UpdateSets_(alpha_i_new,position_of_i_in_I0_,position_of_i_in_I1_,
	      position_of_i_in_I2_,index_working_set_variables_1_,1);
  //printf("Updated sets for the first variable....\n");

  /*printf("After updating I0 for first set we have...\n");
  for(int i=0;i<I0_indices_.size();i++){
    
  printf("I0 position is %d..\n",I0_indices_[i]);
  }*/

  //  printf("will now update sets for the second variable....\n");
  
    // Now update sets for working set variable 2
    
    
  UpdateSets_(alpha_j_new,position_of_j_in_I0_,position_of_j_in_I1_,
	      position_of_j_in_I2_,index_working_set_variables_2_,2);
  //printf("Updated sets for 2nd variable....\n");
  
   // Finally update beta_up and beta_low using indices in I0 and
  // i,j(the working set variables).
  
  /*printf("before updating beta up and beta low using I0...\n");
  printf("i_up=%d..\n",i_up_);
  printf("i_low=%d..\n",i_low_);*/
  
  UpdateBetaUpAndBetaLowUsingI0_(alpha_i_new,alpha_j_new);
  
  /*printf("After updating beta using I0...\n");
  printf("i_up=%d..\n",i_up_);
  printf("i_low=%d..\n",i_low_);*/
  
    
  //Finally update the alpha vector

  alpha_vec_[index_working_set_variables_1_]=alpha_i_new;
  alpha_vec_[index_working_set_variables_2_]=alpha_j_new;

  

  /*printf("After update in the take step function we have...\n");
    alpha_vec_.PrintDebug();*/

  // Lets do some sanity checks

  if(Fi_for_I0_.size()!=I0_indices_.size()){
    
    printf("Size of Fi is %d...\n",Fi_for_I0_.size());
    printf("Size of I0 indices  is %d..\n",I0_indices_.size());
    exit(0);
  }
  if(I0_indices_.size()+I1_indices_.size()+
     I2_indices_.size()!=num_subgradients_available_){
    
    printf("There are some missing elements....\n");
    exit(0);
  }
   // Since this step was a success return 1
  return 1;
  
}

void OCASSMO::CalculatePrimalSolution_(){

  for(int i=0;i<num_subgradients_available_;i++){

    double alpha_i=alpha_vec_[i];

    Vector alpha_i_a_i;
    la::ScaleInit(alpha_i,subgradients_mat_[i],&alpha_i_a_i);

    //Add this to primal solution
    la::AddTo(alpha_i_a_i,&primal_solution_);
  }

  // Finally scale it with -1/\lambda

  la::Scale(-1.0/lambda_reg_const_,&primal_solution_);
}


/*void OCASSMO::UpdatePrimalSolution_(double alpha_i_new,double alpha_j_new){

  // Subtract off the old contribution

  // w <-w- [-1/lambda (\alpha_i a_i+\alpha_j a_j)]
  
  double alpha_i_old=alpha_vec_[index_working_set_variables_1_];
  double alpha_j_old=alpha_vec_[index_working_set_variables_2_];
  
  Vector alpha_i_old_a_i;
  la::ScaleInit(alpha_i_old,subgradients_mat_[index_working_set_variables_1_],
  &alpha_i_old_a_i);
  
  Vector alpha_j_old_a_j; 
  la::ScaleInit(alpha_j_old,subgradients_mat_[index_working_set_variables_2_],
		&alpha_j_old_a_j);
   
  Vector temp;
  la::AddInit(alpha_i_old_a_i,alpha_j_old_a_j,&temp);

 
  la::Scale (-1.0/lambda_reg_const_,&temp);
  
  // Subtract of temp from primal_solution

  la::SubFrom(temp,&primal_solution_);

  Vector alpha_i_new_a_i;
  la::ScaleInit(alpha_i_new,subgradients_mat_[index_working_set_variables_1_],
		&alpha_i_new_a_i);

  Vector alpha_j_new_a_j;
  la::ScaleInit(alpha_j_new,subgradients_mat_[index_working_set_variables_2_],
		&alpha_j_new_a_j);

  Vector new_add;
  la::AddInit(alpha_i_new_a_i,alpha_j_new_a_j,&new_add);
  la::Scale(-1.0/lambda_reg_const_,&new_add);

  // Add to primal_solution
  la::AddTo(new_add,&primal_solution_);
  }*/


/** This procedure checks for optimality w.r.t i_up and i_low
 */
int OCASSMO::CheckForOptimality_(int position,double F_val){


//   printf("F_val=%f..\n",F_val);
//   printf("beta_up=%f..\n",beta_up_);
//   printf("beta_low=%f..\n",beta_low_);
  
  // We already know if this element is in I0 or not.

  bool optimality=true;

  if(position_of_j_in_I0_!=-1){
    // This element is in I0
    
    if(F_val<beta_low_-2*tau_){
      
      // There is violation. When seen as a part of I2
      optimality=false;
    }
    if(F_val>beta_up_+2*tau_){

      
      // violation of optimality. When seen as a part of I1
      optimality=false;
    }
    // If there was no violation just return
    
    if(optimality==true){

      return 0;
    }
    else{ // The variable has violated. Lets see what is the best
	  // variable to work with

      // Pick the best pair to optimize with
      if(beta_low_-F_val>F_val-beta_up_){
	
	// The other variable will be i_low_
	index_working_set_variables_1_=i_low_;

	position_of_i_in_I2_=-1;
	position_of_i_in_I1_=CheckIfInI1_(i_low_);

	//printf("i_low_ is =%d\n",i_low_);
	if(position_of_i_in_I1_==-1){

	  // So this i is in I0 
	  position_of_i_in_I0_=CheckIfInI0_(i_low_);
	  

	  //printf("position of i in I0 is %d...\n",position_of_i_in_I0_);
	  //printf("i_low=%d..\n",i_low_);
	  if(position_of_i_in_I0_==-1){

	    printf("There is a mistake here8..\n");
	    exit(0);
	  }
	}
	else{
	  
	  position_of_i_in_I0_=-1;
	}

	// Also dont forget to cache the value of F_working_set_variables_1_

	F_working_set_variables_1_=beta_low_;
      }
      else{
	
	
	index_working_set_variables_1_=i_up_;
	
	position_of_i_in_I1_=-1;
	position_of_i_in_I2_=CheckIfInI2_(i_up_);
	if(position_of_i_in_I2_==-1){
	  
	  // So this i is in I0 
	  position_of_i_in_I0_=CheckIfInI0_(i_up_);
	  if(position_of_i_in_I0_==-1){
	    
	    printf("There is a mistake here9...\n");
	    exit(0);
	  }
	}
	else{
	  
	  position_of_i_in_I0_=-1;
	}

	// Also dont forget to cache the value of F_working_set_variables_1_
	
	F_working_set_variables_1_=beta_up_;

      }
    }
  }
  
  else{ //either the variable is in I1 or I2
    
    if(position_of_j_in_I1_!=-1){
      // This variable is in I1
      
      if(F_val>beta_up_+2*tau_){
	
	index_working_set_variables_1_=i_up_;
	optimality=false;
	
	position_of_i_in_I1_=-1;
	position_of_i_in_I2_=CheckIfInI2_(i_up_);

	if(position_of_i_in_I2_==-1){
	  
	  // So this i is I0 
	  position_of_i_in_I0_=CheckIfInI0_(i_up_);
	  if(position_of_i_in_I0_==-1){
	    
	    printf("There is a mistake here10...\n");
	    exit(0);
	  }
	}
	else{
	  
	  position_of_i_in_I0_=-1;
	}

	// Also dont forget to cache the value of F_working_set_variables_1_

	F_working_set_variables_1_=beta_up_;
      }
    }
    else{
      
      if(position_of_j_in_I2_==-1){

	printf("There is a mistake. THIS PARTICULAR VARIABLE IS NOWHERE...\n");
	exit(0);
      }
      // The variable j is in I2
      if(F_val<beta_low_-2*tau_){
	
	// There is violation.
	
	index_working_set_variables_1_=i_low_;
	optimality=false;
	
	position_of_i_in_I2_=-1;
	position_of_i_in_I1_=CheckIfInI1_(i_low_);
	if(position_of_i_in_I1_==-1){
	  
	  // So this i is I0 
	  position_of_i_in_I0_=CheckIfInI0_(i_low_);
	  if(position_of_i_in_I0_==-1){
	    
	    printf("There is a mistake here11...\n");
	    exit(0);
	  }
	}
	else{
	  
	  position_of_i_in_I0_=-1;
	}

	// Also dont forget to cache the value of F_working_set_variables_1_
	
	F_working_set_variables_1_=beta_low_;
	
      }
    }
  }
  if(optimality){

    //    printf("This pair is already optimal...\n");
    return 0;
  }
  else{
    
    // Check if it is beneficial to work with the above derived pairs
    // of variables

    index_working_set_variables_2_=position;

    
    int ret_val= TakeStep_();
    
    return ret_val;
  }
  
}



int OCASSMO::ExamineSubgradient_(int position){
  
  
  // In this routine we examine this particular subgradient and see 
  
  // Pull out the value of F_i. In case i\in I0 then this value will
  // be in the cache itself.

  position_of_j_in_I0_=CheckIfInI0_(position);

 
  double F2_value;
  if(position_of_j_in_I0_!=-1){
  
    // This element is in I0. Hence pull out its F_i value from the
    // cache


    F2_value=Fi_for_I0_[position_of_j_in_I0_];
    //Also mark the position in I1 and I2 as -1
  
    position_of_j_in_I1_=-1;
    position_of_j_in_I2_=-1;
    F_working_set_variables_2_=F2_value;

  }

  else{
    //The F_i value is not in cache and hence compute it from scratch.
    // Remember the position is w.r.t the subgradient matrix
    
    F2_value=ComputeFiValue_(position);

    // Cache this value
    F_working_set_variables_2_=F2_value;
    
    // Find out the position of j in I1 and I2

    position_of_j_in_I0_=-1;
    position_of_j_in_I1_=CheckIfInI1_(position);
    
    //printf("position of j in I1 is %d...\n",position_of_j_in_I1_);
    if(position_of_j_in_I1_==-1){

      position_of_j_in_I2_=CheckIfInI2_(position);
      if(position_of_j_in_I2_==-1){
	
	printf("There is a mistake here12...\n");
	exit(0);
      }
    }
    else{
      // The element is in I1
      position_of_j_in_I2_=-1;
    }
  }
  
  // Firstly update
  UpdateBetaUpAndBetaLow_(F2_value,position);
  
  //Finally check for optimality
  int ret_val= CheckForOptimality_(position,F2_value); 

  return ret_val;
}

void OCASSMO::PrintSubgradientsAndIntercepts_(){
  
  printf("The subgradients are...\n");
  for(int z=0;z<num_subgradients_available_;z++){

    for(int j=0;j<num_dims_;j++){

      printf("%f,",subgradients_mat_[z][j]);
    }
    printf("\n");
  }
  printf("The intercepts are...\n");
  for(int k=0;k<num_subgradients_available_;k++){
    
    printf("%f,",intercepts_vec_[k]);
    
  }
}

void OCASSMO::SMOMainRoutine_(){
  
  // variables that keep track of KKT conditions
  int num_changed=0;
  bool examine_all_subgradients=true;

  int num_iterations=0;
  int MAX_NUM_ITERATIONS=200000;
  while((num_changed>0||examine_all_subgradients)&&num_iterations<
	MAX_NUM_ITERATIONS){

    //printf("hi....\n");
    
    // Set num_changed to 0

    num_changed=0;
    
    if(examine_all_subgradients){
      
      //loop over all subgradients.
    
      for(int i=0;i<num_subgradients_available_;i++){

	int position=i;
	
	num_changed+=ExamineSubgradient_(position);
	//printf("alpha vector returned is ...\n");
	//alpha_vec_.PrintDebug();

	//printf("Before going for next iteration the beta values are...\n");
	//printf("beta_up=%f,beta_low=%f...\n",beta_up_,beta_low_);
	
	// Reset all positions
	//	ResetPositions_();
      }
    }
    else{
      //Now we shall loop over swubgradients that are in I0.
      
      for(int i=0;i<I0_indices_.size();i++){
	
	int position=I0_indices_[i];
	position_of_j_in_I0_=i;
	position_of_j_in_I1_=-1;
	position_of_j_in_I2_=-1;
	num_changed+=ExamineSubgradient_(position);
	//ResetPositions_();
	// May be we can quit if optimality looks satisfied. However
	// I am omitting this piece of code for now.
	
      }
    }
    if(examine_all_subgradients==true){

      examine_all_subgradients=false;
    }
    else{
      if(num_changed==0){

	examine_all_subgradients=true;
      }
    }

    num_iterations++;
  }

  if(num_iterations==MAX_NUM_ITERATIONS){
    printf("CAUTION: SMO used all iterations\n");

    printf("lambda=%f,bandwidth=%f...\n",
	   lambda_reg_const_,smoothing_kernel_bandwidth_);
    //PrintSubgradientsAndIntercepts_(); 
  }
    //  printf("Finished everything. Bailing out.....\n");
}

void OCASSMO::InitializeIndexSets_(){
  
  //Iterate over each element and set up I0,I1,I2

  for(int i=0;i<num_subgradients_available_;i++){
    
    // I_2=\{i|\alpha_i=0\}
    if(fabs(alpha_vec_[i])<SMALL){

      I2_indices_.PushBack(1);
      int size=I2_indices_.size();
      I2_indices_[size-1]=i;
    }
    else{

      if(fabs(1-alpha_vec_[i])<SMALL){

	// I_1=\{i|\alpha_i=1\}
	
	I1_indices_.PushBack(1);
	int size=I1_indices_.size();
	I1_indices_[size-1]=i;
      }
      else{
	
	// This element is in I0
	I0_indices_.PushBack(1);
	int size=I0_indices_.size();
	I0_indices_[size-1]=i;
      }
    }
  }
}

void OCASSMO::InitializeFiForI0_(){
  int size=I0_indices_.size();  

  for(int i=0;i<size;i++){

    int position=I0_indices_[i];
    double val=ComputeFiValue_(position);
    Fi_for_I0_.PushBack(1);
    Fi_for_I0_[i]=val;
  }
}


void OCASSMO::InitializeUsingWarmStart_(Vector &initial_alpha){

  //Lets do a quick check and make sure that the initial_alpha is
  //feasible

  double total_sum=0.0;
  for(int i=0;i<initial_alpha.length();i++){
    
    if(initial_alpha[i]>1.0 || initial_alpha[i]<-SMALL){

      printf("The seed alpha vector is infeasbile..\n");
      printf("initial_alpha[%d]=%f..\n",i,initial_alpha[i]);
      printf("Exiting....\n");
      exit(0);
    }
    total_sum+=initial_alpha[i];
  }

  // Check if this sum=1.0
  
  if(fabs(total_sum-1)>SMALL){

    printf("The seed alpha is infeasible as it's components don't sum to 1...\n");

    printf("The seed vector was...\n");
    initial_alpha.PrintDebug();

    printf("Exiting......\n");
    exit(0);
  }

  alpha_vec_.Alias(initial_alpha);  
  InitializeIndexSets_();
  InitializeFiForI0_();
}

void OCASSMO::SolveOCASSMOProblem_(){
  
 
  SMOMainRoutine_();

  // Now calculate primal solution
  
  CalculatePrimalSolution_();
 
  //printf("Checking if KKT conditions are satisfied before returning..\n");
  //printf("Lambda_=%f..\n",lambda_reg_const_);
  for(int i=0;i<num_subgradients_available_;i++){
    
    double F_val=ComputeFiValue_(i);
    if(fabs(alpha_vec_[i])<SMALL){
      
      // This is in I2
      if(F_val<beta_up_){
	
	printf("There is a mistake here18...\n");
	printf("F_val=%f..\n",F_val);
	printf("beta_up=%f..\n",beta_up_);
	//exit(0);
      }
    }
    else{
      
      if(fabs(1-alpha_vec_[i])<SMALL){
	
	// This is in I1
	if(F_val>beta_low_){
	  
	  printf("There is a mistake here13...\n");
	  // exit(0);
	}
      }
      
      else{
	
	// This element is in I0
	if(fabs(F_val-(beta_up_+beta_low_)/2.0)>SMALL){
	  
	  // There is a violation
	  printf("F_val calculated from scratch=%f...\n",F_val);
	  printf("beta_up=%f...\n",beta_up_);
	  printf("beta_low=%f...\n",beta_low_);
	  
	  printf("There is a mistake here14, lambda_reg_const_=%f,smoothing kernel bandwidth=%f...\n",lambda_reg_const_,smoothing_kernel_bandwidth_);

	  //PrintSubgradientsAndIntercepts_();
	
	  /*   printf("The subgradients are...\n");
	       for(int j=0;j<subgradients_mat_.size();j++){
	       printf("Subgradient is..\n");
	       for(int k=0;k<subgradients_mat_[j].length();k++){
	       
	       
	       printf("%f,",subgradients_mat_[j][k]);
	       
	    }
	  }
	  printf("...\n");

	  printf("Intercepts are...\n");

	  for(int j=0;j<intercepts_vec_.size();j++){

	    printf("%f,",intercepts_vec_[j]);
	  }
	  printf("....\n");
	  
	  
	  
	  int pos=CheckIfInI0_(i);
	  printf("alpha value=%f and position in I0=%d...\n",alpha_vec_[i],pos);
	  printf("F value from the cache is =%f\n",Fi_for_I0_[pos]);
	  
	  
	  printf("I0 is ...\n");
	  for(int i=0;i<I0_indices_.size();i++){
	    printf("I0_[%d]=%d,Fi_for_I0_[%d]=%f..\n",
		   i,I0_indices_[i],i,Fi_for_I0_[i]);
	    
	  }
	  printf("I1 is ...\n");
	    
	  for(int i=0;i<I1_indices_.size();i++){
	    
	    double val=ComputeFiValue_(I1_indices_[i]);
	    printf("I1_[%d],Fi[%d]=%d,%f..\n",
		   i,I1_indices_[i],i,val);
	    
	  }
	  
	  printf("I2 is...\n");
	  
	  for(int i=0;i<I2_indices_.size();i++){
	    
	    double val=ComputeFiValue_(I2_indices_[i]);
	    printf("I2_[%d]=%d,Fi[%d]=%f..\n",
		   i,I2_indices_[i],i,val);
	    
	  }
	  
	  }*/
	}
      }
    }
  }
}

void OCASSMO::Init(ArrayList <Vector> &subgradients_mat, 
		   ArrayList <double> &intercepts_vec, 
		   double lambda_reg_const,
		   Vector &initial_alpha,
		   double bandwidth){

  // Copy the value of the regularization constant

  lambda_reg_const_=lambda_reg_const;  
  smoothing_kernel_bandwidth_=bandwidth;

  subgradients_mat_.InitAlias(subgradients_mat);
  intercepts_vec_.InitAlias(intercepts_vec);
  
  // Assume that the 0 subgradients that we always have is already added
  num_subgradients_available_=subgradients_mat.size();
  num_dims_=subgradients_mat_[0].length(); 
   
  I0_indices_.Init(0);
  I1_indices_.Init(0);
  I2_indices_.Init(0);
  Fi_for_I0_.Init(0);    

  // The number of dimensions (actually after having appended the
  // dataset with 1).

  InitializeUsingWarmStart_(initial_alpha);

  //Initialize the primal solution. The primal solution is
  //w=-\frac{1}{\lambda}\sum \alpha_l a_l. Initialize this to 0.

  primal_solution_.Init(num_dims_);
  primal_solution_.SetZero();
  
  // Initialize beta_up and beta_low
  
  beta_up_=DBL_MAX;
  beta_low_=-DBL_MAX;
  i_up_=1;
  i_low_=0;
  tau_=pow(10,-6);
  eps_=pow(10,-6);

  position_of_i_in_I0_=-1;
  position_of_i_in_I1_=-1;
  position_of_i_in_I2_=-1;

  position_of_j_in_I0_=-1;
  position_of_j_in_I1_=-1;
  position_of_j_in_I2_=-1;

  
  /* printf("number of subgradients available are %d....\n",num_subgradients_available_);
  
  printf("Printing the sizes and capacity..\n");
  printf("I0:size=%d, capacity=%d..\n",I0_indices_.size(),I0_indices_.capacity());
  printf("I1:size=%d, capacity=%d..\n",I1_indices_.size(),I1_indices_.capacity());
  printf("I2:size=%d, capacity=%d..\n",I2_indices_.size(),I2_indices_.capacity());
  printf("Fi_for_I0: size=%d, capacity=%d...\n",Fi_for_I0_.size(),Fi_for_I0_.capacity());

  printf("Initial alpha vector is ...\n");
  alpha_vec_.PrintDebug();*/
  
}
