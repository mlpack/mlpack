#ifndef REGRESSION_LL2_H
#define REGRESSION_LL2_H
#include "fastlib/fastlib_int.h"


// This function checks for the prunability of B^TWY

template <typename TKernel>
index_t FastRegression<TKernel>::PrunableB_TWY_(Tree *qnode, Tree *rnode, 
						Matrix &dl, Matrix &du){
  
  //I shall see if B^TWY is prunable or not and accordingly return 0 or 1
  
  // try pruning after bound refinement: first compute distance/kernel
  // value bounds

  //printf("Came to check for the prunability of B^TWY..\n");
  DRange dsqd_range;
  dsqd_range.lo = qnode->bound ().MinDistanceSq (rnode->bound ());
  dsqd_range.hi = qnode->bound ().MaxDistanceSq (rnode->bound ());
  DRange kernel_value_range = kernel_.RangeUnnormOnSq (dsqd_range);
  
  // the new lower bound after scaling the B^TY matrix with the scalar the 
  //minimum value of kernel
  
  double m=kernel_value_range.lo;
  
  la::ScaleOverwrite(m, rnode->stat().b_ty, &dl);
  
  //The new upper bound

  
  m = -1+kernel_value_range.hi;
  la::ScaleOverwrite (m, rnode->stat().b_ty, &du);
    
  //Lets find the maximum error that one can commit due to pruning 
  // This is nothing but the B^TY matrix scaled by a value m defined below
    
  m=0.5*(kernel_value_range.hi-kernel_value_range.lo);
  Matrix max_error;
    
  //max_error <- m*(b_ty)
  la::ScaleInit (m, rnode->stat().b_ty, &max_error); 
    
  //Now lets calculate the allowed error. 
  //allowed_error=tau*(rnode->stat().b_ty/rroot->stat().b_ty)(b_twy_mass_l+dl)
    
    
  //Lets define a matrix new_mass_l as shown below.This matrix takes 
  //into account the addditional dl that will be added if pruning takes 
  //place
    
    
  Matrix new_mass_l;
   
  la::AddInit(qnode->stat().b_twy_mass_l, dl, &new_mass_l);
    
  //Now it is easy to see that the allowed error is nothing but the 
  //scaled version of new_mass_l
    
 

  //Now we can prune only if max_error is componentwise lesser than allowed_error
    
  for(index_t col=0;col<qnode->stat().b_twy_mass_u.n_cols();col++){
      
    for(index_t row=0;row<qnode->stat().b_twy_mass_u.n_rows();row++){
	
      //We will prune only if the matrix is prunable componenet wise
      //This is what test now

      //printf("tau_ is %f\n",tau_);
      
      
      double alpha=tau_*
	(double)(rnode->stat().b_ty.get(row,col))/
	(rroot_->stat().b_ty.get(row,col));

      if(rroot_->stat().b_ty.get(row,col)==0){
	printf("BTY Wrongly estimated..\n");
	rroot_->stat().b_ty.PrintDebug();
	exit(0);
      }
          
      if(max_error.get(row,col)>=alpha*new_mass_l.get(row,col)){
	//pruning failed
	//Set everything back to 0 and return
	
	dl.SetAll(0);
	du.SetAll(0);
	return 0;
      }
    }
  }
 
  printf("dl beign sent from BTWY is ...\n");
  dl.PrintDebug();

  printf("du being sent from BTWY is ..\n");
  du.PrintDebug();
  return 1;
}
  
template <typename TKernel>
index_t FastRegression<TKernel>:: PrunableB_TWB_(Tree *qnode, Tree *rnode, Matrix &dl, Matrix &du){
    
  //I shall see if B^TWB is prunable or not and accordingly return 0 or 1
    
  //printf("Came to check for the prunability of B^TWB..\n");
    
  DRange dsqd_range;
  // try pruning after bound refinement: first compute distance/kernel
  // value bounds
  dsqd_range.lo = qnode->bound ().MinDistanceSq (rnode->bound ());
  dsqd_range.hi = qnode->bound ().MaxDistanceSq (rnode->bound ());
  DRange kernel_value_range = kernel_.RangeUnnormOnSq (dsqd_range);
    
    
    
  // the new lower bound after scaling the B^TB matrix with the scalar the 
  //minimum value of kernel
    
  double m=kernel_value_range.lo;
 
  //printf("kernel value lo is %f\n",m);
  la::ScaleOverwrite(m, rnode->stat().b_tb, &dl);
    
  //The new upper bound
    
  m = -1+kernel_value_range.hi;
  la::ScaleOverwrite (m, rnode->stat().b_tb, &du);

  //Lets find the maximum error that one can commit due to pruning 
  // This is nothing but the B^TB matrix scaled by a value m defined below
    
  m=0.5*(kernel_value_range.hi-kernel_value_range.lo);
  Matrix max_error;
    
    
  //max_error <- m*(b_tb)
  la::ScaleInit (m, rnode->stat().b_tb, &max_error); 
 
  //Now lets calculate the allowed error. 
    
  //allowed_error=tau*(rnode->stat().b_tb/rrot->stat().b_tb)(b_twb_mass_l+dl)
  //Lets define a matrix new_mass_l as shown below.This matrix takes 
  //into account the addditional dl that will be added if pruning takes 
  //place
    
  //printf("dl in B^TWB is..\n");
  //dl.PrintDebug();
    
  Matrix new_mass_l;
  la::AddInit(qnode->stat().b_twb_mass_l,dl, &new_mass_l);
    
  //Now it is easy to see that the allowed error is nothing but the 
  //scaled version of new_mass_l
    

 
  //Now we can prune only if max_error is componentwise lesser than allowed_error
    
  for(index_t col=0;col<qnode->stat().b_twb_mass_u.n_cols();col++){
      
    for(index_t row=0;row<qnode->stat().b_twb_mass_u.n_rows();row++){

      //printf("It is %f\n",(rroot_->stat().b_tb.get(row,col)));

      //We shall prune only if the matrix is componenet
      //wise prunable
	
      //Lets define a constant alpha as shown below.
     	double alpha=
	  tau_*(double)(rnode->stat().b_tb.get(row,col))/
	  (rroot_->stat().b_tb.get(row,col));
	if(rroot_->stat().b_tb.get(row,col)==0){

	  printf("btb.is not properly defined\n");
	}
	//printf("alpha is %f\n",alpha);
      


	// printf("the max error is %f\n",max_error.get(row,col));
	//printf("allowed error is %f\n",alpha*new_mass_l.get(row,col));
      if(max_error.get(row,col)>=alpha*new_mass_l.get(row,col)){
	//pruning failed
	  
	dl.SetAll(0);
	du.SetAll(0);
	return 0;
	  
      }
    }
  }
  printf("dl beign sent from BTWB is ...\n");
  dl.PrintDebug();
  
  printf("du being sent from BTWB is ..\n");
  du.PrintDebug();
  return 1;
}
  
template <typename TKernel>
void FastRegression<TKernel>::
UpdateBoundsForPruningB_TWY_(Tree *qnode, Matrix &dl_b_twy, Matrix &du_b_twy){
    
 


  //In this function we shall  update bounds of the quantity that is prunable.
  //This is similar to the UpdateBounds_(..) function. However this will be
  //called to incoprorate changes in bounds due to pruning
   
  //b_twy_mass_l <- b_twy_mass_l+dl
  //b_twy_mass_u <- b_twy_mass_u+du
    
 
  la::AddTo(dl_b_twy, &qnode->stat().b_twy_mass_l);
  la::AddTo(du_b_twy, &qnode->stat().b_twy_mass_u);
    
    
  // for a leaf node, incorporate the lower and upper bound changes into
  // its additional offset
    
  if (qnode->is_leaf ()){
      
    //more_l <- more_l + dl
    //more_u <-more_u + du
   
    la::AddTo (dl_b_twy, &(qnode->stat().b_twy_more_l));
    la::AddTo (du_b_twy, &(qnode->stat().b_twy_more_u));   

  }
    
  // otherwise, incorporate the bound changes into the owed slots of
  // the immediate descendants
  else{
      
    
    //Transmit the owed values to the left child
      
    la::AddTo (dl_b_twy, &(qnode->left()->stat().b_twy_owed_l));
    la::AddTo (du_b_twy, &(qnode->left()->stat().b_twy_owed_u));
      
    //transmission of the owed values to the right children
      
    la::AddTo (dl_b_twy, &(qnode->right()->stat().b_twy_owed_l));
    la::AddTo (du_b_twy, &(qnode->right()->stat().b_twy_owed_u));
      
  }
}
  
template <typename TKernel>
void FastRegression<TKernel>::UpdateBoundsForPruningB_TWB_(Tree
							   *qnode, Matrix &dl_b_twb,
							   Matrix &du_b_twb){
    
 
  
  //In this function we shall  update bounds of the quantity that is prunable.
  //This is similar to the UpdateBounds_(..) function. However this will be
  //called to incoprorate changes in bounds due to pruning
    
  // Changes the upper and lower bounds.
    
  //b_twb_mass_l <- b_twb_mass_l+dl
  //b_twb_mass_u <- b_twb_mass_u+du
    
 

  la::AddTo (dl_b_twb ,&(qnode->stat().b_twb_mass_l));
  la::AddTo(du_b_twb, &(qnode->stat().b_twb_mass_u));


    
  // for a leaf node, incorporate the lower and upper bound changes into
  // its additional offset
    
  if (qnode->is_leaf ()){
      
    //more_l <- more_l + dl
    //more_u <-more_u + du
      
    la::AddTo (dl_b_twb, &(qnode->stat().b_twb_more_l));
    la::AddTo (du_b_twb, &(qnode->stat().b_twb_more_u));   
  }
    
  // otherwise, incorporate the bound changes into the owed slots of
  // the immediate descendants
  else{
      
    //Transmit the owed values to the left child
      
    la::AddTo (dl_b_twb, &(qnode->left()->stat().b_twb_owed_l));
    la::AddTo (du_b_twb, &(qnode->left()->stat().b_twb_owed_u));
      
    //transmission of the owed values to the right children
      
    la::AddTo (dl_b_twb, &(qnode->right()->stat().b_twb_owed_l));
    la::AddTo (du_b_twb, &(qnode->right()->stat().b_twb_owed_u));
      
  }
}
 
template <typename TKernel>

void FastRegression<TKernel>:: 
MergeChildBoundsB_TWB_(  FastRegression<TKernel>::
			 FastRegressionStat *left_stat,
			 FastRegression<TKernel>::
			 FastRegressionStat *right_stat,
			 FastRegression<TKernel>::
			 FastRegressionStat &parent_stat){

  //This means we want to merge the bounds of b_twb
  //b_twb_mass_l_parent= 
  //max(min(b_twb_mass_l,left_child,b_twb_mass_l_right_child),
  //    parent
  
  //So lets find the componentwise minimum and maximum 

  //The first thing to do is check if qnode is a leaf. if it is then
  //return
  Matrix max_children;
  Matrix min_children;
  max_children.Init(parent_stat.b_twb_mass_l.n_rows(), 
		    parent_stat.b_twb_mass_l.n_cols());

  min_children.Init(parent_stat.b_twb_mass_l.n_rows(), 
		    parent_stat.b_twb_mass_l.n_cols());


  for(index_t col=0;col<parent_stat.b_twb_mass_l.n_cols();col++){
    for(index_t row=0;row<parent_stat.b_twb_mass_l.n_rows();row++){

      if(left_stat->b_twb_mass_l.get(row,col) <= 
	 right_stat->b_twb_mass_l.get(row,col)){

	//left child has lesser mass_l value

	min_children.set(row,col,
			 left_stat->b_twb_mass_l.get(row,col));

	max_children.set(row,col,
			 right_stat->b_twb_mass_l.get(row,col));
      }

      else{
	//right child has lesser mass-l value
	  
	min_children.set(row,col,
			 right_stat->b_twb_mass_l.get(row,col));
	  
	max_children.set(row,col,
			 left_stat->b_twb_mass_l.get(row,col));

      }

    }
      
  }

 
  //Now compare with parent...
 
   for(index_t col=0;col<parent_stat.b_twb_mass_l.n_cols();col++){
    for(index_t row=0;row<parent_stat.b_twb_mass_l.n_rows();row++){

      if(parent_stat.b_twb_mass_l.get(row,col)<min_children.get(row,col)){
	//parents value is less than the minimum of children. Hence update the 
	//value of the parent 
	parent_stat.b_twb_mass_l.set(row,col,min_children.get(row,col));
      }

      if(parent_stat.b_twb_mass_u.get(row,col)>max_children.get(row,col)){
	//parents value is greater than the maximmum of children. 
	//Hence update the value of the parent
	parent_stat.b_twb_mass_u.set(row,col,max_children.get(row,col));
      }      
    }
   }
}


template <typename TKernel>
void FastRegression<TKernel>:: 
MergeChildBoundsB_TWY_( FastRegression<TKernel>::
			FastRegressionStat *left_stat,
		        FastRegression<TKernel>::
			FastRegressionStat *right_stat,
		        FastRegression<TKernel>::
			FastRegressionStat &parent_stat){

  //This means we want to merge the bounds of b_twb
  //b_twb_mass_l_parent= 
  //max(min(b_twy_mass_l,left_child,b_twy_mass_l_right_child),
  //    parent)
  
  //So lets find the componentwise minimum and maximum 

  Matrix max_children;
  Matrix min_children;
  max_children.Init(parent_stat.b_twy_mass_l.n_rows(), 
		    parent_stat.b_twy_mass_l.n_cols());

  min_children.Init(parent_stat.b_twy_mass_l.n_rows(), 
		    parent_stat.b_twy_mass_l.n_cols());


  for(index_t col=0;col<parent_stat.b_twy_mass_l.n_cols();col++){
    for(index_t row=0;row<parent_stat.b_twy_mass_l.n_rows();row++){

      if(left_stat->b_twy_mass_l.get(row,col) <= 
	 right_stat->b_twy_mass_l.get(row,col)){

	//left child has lesser mass_l value

	min_children.set(row,col,
			 left_stat->b_twy_mass_l.get(row,col));

	max_children.set(row,col,
			 right_stat->b_twy_mass_l.get(row,col));
      }

      else{
	//right child has lesser mass-l value
	  
	min_children.set(row,col,
			 right_stat->b_twy_mass_l.get(row,col));
	  
	max_children.set(row,col,
			 left_stat->b_twy_mass_l.get(row,col));

      }
    }   
  }

  //Now compare with parent...
 
  for(index_t col=0;col<parent_stat.b_twy_mass_l.n_cols();col++){
    for(index_t row=0;row<parent_stat.b_twy_mass_l.n_rows();row++){

      if(parent_stat.b_twy_mass_l.get(row,col)<min_children.get(row,col)){
	//parents value is less than the minimum of children. Hence update the 
	//value of the parent
	parent_stat.b_twy_mass_l.set(row,col,min_children.get(row,col));
      }

      if(parent_stat.b_twy_mass_u.get(row,col)> max_children.get(row,col)){
	//parents value is more than the maximmum of children. 
	//Hence update the value of the parent

	parent_stat.b_twy_mass_u.set(row,col,max_children.get(row,col));
      }      
    }
  }
}



/* This function merges child bounds with that of the parent
 * and the children

*/  
template <typename TKernel>
void FastRegression<TKernel>::
MergeChildBounds_( Tree *qnode,check_for_prune_t flag){
 
  //We will merge bounds depending on the value of the flag

  //But firstly we will check if the qnode is a leaf node. if it is
  //then we will simply return

  if(qnode->is_leaf()){
    return;
  }

  //The statistics of the parent and the children node
  FastRegressionStat &parent_stat=qnode->stat();
  FastRegressionStat *left_stat=&(qnode->left()->stat());
  FastRegressionStat *right_stat=&(qnode->right()->stat());
  if(flag==CHECK_FOR_PRUNE_B_TWB){

    MergeChildBoundsB_TWB_(left_stat,right_stat,parent_stat);
   
  }
  else{
    if(flag==CHECK_FOR_PRUNE_B_TWY){

      MergeChildBoundsB_TWY_(left_stat,right_stat,parent_stat);
    }
    else{
      //printf("will merge both the bounds...\n");
      MergeChildBoundsB_TWB_(left_stat,right_stat,parent_stat);
      MergeChildBoundsB_TWY_(left_stat,right_stat,parent_stat);

    }
  }
}
 
/** This is the base case of regression. Here depending on the flag the 
 *  qunatities will be calculated exhaustively 
 */


template <typename TKernel>

void FastRegression<TKernel>::FRegressionBaseB_TWY_(Tree *qnode, Tree *rnode){

  //subtract along each dimension as we are now doing exhaustive calculations

  //more_u <- more_u - rnode->stat().b_ty

  printf("qnode start is %d and qnode end id %d\n",qnode->begin(),qnode->end());
 
 printf("rnode start is %d and rnode end id %d\n",rnode->begin(),rnode->end());
  
  printf("Hit base regression of BTWY..\n");

  la::SubFrom (rnode->stat().b_ty, &qnode->stat().b_twy_more_u);

  //Having subtracted calculate B^TWY exhaustively
  //One can do this by using linear algebra routines available 
  //in LaPack. however we shall not use them because B can be a 
  //very large matrix
  //printf("Came to regression BTWY base..\n");
 
  //printf("In base regression of BTWY..\n");
  //printf("qnode->start=%d\n",qnode->begin());
  //printf("qnode->end=%d\n",qnode->end());

  //printf("In base regression of BTWY..\n");
  //printf("rnode->start=%d\n",qnode->begin());
  //printf("rnode->end=%d\n",qnode->end());


  for(index_t q=qnode->begin();q<qnode->end();q++){
    
    //Get Query point
    const double *q_col=qset_.GetColumnPtr(q);
    
    for(index_t row=0;row<b_twy_l_estimate_[q].n_rows();row++){ 

      for(index_t col=0;col<b_twy_u_estimate_[q].n_cols();col++){

	//Along all dimensions
	for(index_t r=rnode->begin();r<rnode->end();r++){

	  //Get reference point
	  const double *r_col = rset_.GetColumnPtr (r);

	  // pairwise distance and kernel value
	  double dsqd =
	    la::DistanceSqEuclidean (qset_.n_rows (), q_col, r_col);
	  double ker_value = kernel_.EvalUnnormOnSq (dsqd);

	   //This is nothing but B^TWY being evaluated dimension wise
	  if (row != 0){

	    double val=b_twy_l_estimate_[q].get(row,col)+ 
	      ker_value * rset_weights_[old_from_new_r_[r]] * rset_.get (row- 1, r);

	    b_twy_l_estimate_[q].set(row,col,val);

	    val=b_twy_u_estimate_[q].get(row,col)+ 
	      ker_value * rset_weights_[old_from_new_r_[r]] * rset_.get (row- 1, r);
	    
	    b_twy_u_estimate_[q].set(row,col,val);
	    
	  }
	  else{
	    double val= b_twy_l_estimate_[q].get(row,col) + 
	      ker_value * rset_weights_[old_from_new_r_[r]];

	    b_twy_l_estimate_[q].set(row,col,val);

	    val= b_twy_u_estimate_[q].get(row,col) + 
	      ker_value * rset_weights_[old_from_new_r_[r]];
	   
	    b_twy_u_estimate_[q].set(row,col,val);

	  }
	
	}
      }
    }
  }
  //Loop over each point and set the max and min 

  Matrix min_l;
  min_l.Init(rset_.n_rows()+1,1);

  Matrix max_u;
  max_u.Init(rset_.n_rows()+1,1);

  min_l.SetAll(DBL_MAX);
  max_u.SetAll(DBL_MIN);


  //Iterate over each row 
  for(index_t row=0;row<min_l.n_rows();row++){

    //iterate over each column
    for(index_t col=0;col<min_l.n_cols();col++){

      //iterate over each query point
      for(index_t q = qnode->begin (); q < qnode->end (); q++){

	if(b_twy_l_estimate_[q].get(row,col) + 
	   qnode->stat().b_twy_more_l.get(row,col) < min_l.get(row,col)){
	  
	  double val=(b_twy_l_estimate_[q].get(row,col) + 
		      qnode->stat().b_twy_more_l.get(row,col));
	  min_l.set(row,col,val);
	  
	}
	
	if(b_twy_u_estimate_[q].get(row,col) + 
	   qnode->stat().b_twy_more_u.get(row,col) > max_u.get(row,col)){
	  
	  double val=b_twy_u_estimate_[q].get(row,col) + 
	    qnode->stat().b_twy_more_u.get(row,col);
	  max_u.set(row,col,val);
	  
	}
      }
    }
  }

  //having looped over each point

  qnode->stat().b_twy_mass_u.CopyValues(max_u);
  qnode->stat().b_twy_mass_l.CopyValues(min_l);

}

template <typename TKernel>
void FastRegression<TKernel>::FRegressionBaseB_TWB_(Tree *qnode,Tree *rnode){

  printf("qnode start is %d and qnode end id %d\n",qnode->begin(),qnode->end());

  printf("rnode start is %d and rnode end id %d\n",rnode->begin(),rnode->end());
 
  printf("Hit base regression of BTWB........\n\n");

  la::SubFrom (rnode->stat().b_tb, &qnode->stat().b_twb_more_u);


 for (index_t q = qnode->begin(); q < qnode->end(); q++){	//for each query point
    
    const double *q_col = qset_.GetColumnPtr (q);

    for (index_t r = rnode->begin(); r < rnode->end(); r++){ //for each reference point

      //Get reference point
      const double *r_col = rset_.GetColumnPtr (r);

      // pairwise distance and kernel value
      double dsqd =
	la::DistanceSqEuclidean (qset_.n_rows (), q_col, r_col);
      double ker_value = kernel_.EvalUnnormOnSq (dsqd);
      
      for(index_t col = 0; col < rset_.n_rows () + 1; col++){	//along each direction
	
	for(index_t row=0; row< rset_.n_rows () + 1; row++){	 

	  	
	  if(col==0){

	    //For this column 
	    if (row != 0){

	      double val1=b_twb_l_estimate_[q].get(row,col)+ 
		ker_value * rset_.get(row-1,r);
	      b_twb_l_estimate_[q].set(row,col,val1);

	      double val2=b_twb_u_estimate_[q].get(row,col)+ 
		ker_value * rset_.get(row-1,r);
	      b_twb_u_estimate_[q].set(row,col,val2);
	      
	    }

	    else{
	      
	      double val1=b_twb_l_estimate_[q].get(row,col) 
		+ ker_value ;
	      
	      b_twb_l_estimate_[q].set(row,col,val1);
	      
	      double val2=b_twb_u_estimate_[q].get(row,col) 
		+ ker_value ;
	      
	      b_twb_u_estimate_[q].set(row,col,val2);

	    }
	  }//end of col 0...............

	  //Column!=0

	  else{
	    if(row!=0){

	      double val1=b_twb_l_estimate_[q].get(row,col) 
		+ ker_value* rset_.get(row-1,r)* rset_.get(col-1,r);
	      
	      b_twb_l_estimate_[q].set(row,col,val1);

	      double val2=b_twb_u_estimate_[q].get(row,col) 
		+ ker_value* rset_.get(row-1,r)* rset_.get(col-1,r);
	      
	      b_twb_u_estimate_[q].set(row,col,val2);
	    }

	    else{
	      double val1=b_twb_l_estimate_[q].get(row,col) 
		+ ker_value* rset_.get(col-1,r);
	      b_twb_l_estimate_[q].set(row,col,val1);

	      double val2=b_twb_u_estimate_[q].get(row,col) 
		+ ker_value* rset_.get(col-1,r);
	      b_twb_u_estimate_[q].set(row,col,val2);

	    }
	  }
	}
      }
    }
 }

 //Loop over each point and set the max and min 

  Matrix min_l;
  min_l.Init(rset_.n_rows()+1,rset_.n_rows()+1);

  Matrix max_u;
  max_u.Init(rset_.n_rows()+1,rset_.n_rows()+1);

  min_l.SetAll(DBL_MAX);
  max_u.SetAll(DBL_MIN);


  //Iterate over each row 
  for(index_t row=0;row<min_l.n_rows();row++){

    //iterate over each column
    for(index_t col=0;col<min_l.n_cols();col++){

      //iterate over each query point
      for(index_t q = qnode->begin (); q < qnode->end (); q++){

	if(b_twb_l_estimate_[q].get(row,col) + 
	   qnode->stat().b_twb_more_l.get(row,col) < min_l.get(row,col)){
	  
	  double val=(b_twb_l_estimate_[q].get(row,col) + 
		      qnode->stat().b_twb_more_l.get(row,col));
	  min_l.set(row,col,val);
	  
	}
	
	if(b_twb_u_estimate_[q].get(row,col) + 
	   qnode->stat().b_twb_more_u.get(row,col) > max_u.get(row,col)){
	  
	  double val=b_twb_u_estimate_[q].get(row,col) + 
	    qnode->stat().b_twb_more_u.get(row,col);
	  max_u.set(row,col,val);
	  
	}
      }
    }
  }

  //having looped over each point

  qnode->stat().b_twb_mass_u.CopyValues(max_u);
  qnode->stat().b_twb_mass_l.CopyValues(min_l);

}

   


template <typename TKernel>

void FastRegression<TKernel>::FRegressionBase_(Tree *qnode, Tree *rnode, check_for_prune_t flag){

  //printf("Came to FRegressionBase_.....\n");

  if(flag==CHECK_FOR_PRUNE_B_TWB){

   
    FRegressionBaseB_TWB_(qnode,rnode);
  }

  else{
    if(flag==CHECK_FOR_PRUNE_B_TWY){
      
      //So we need to calculate only B^TWB exhaustively
      
      FRegressionBaseB_TWY_(qnode,rnode);
    }
    else{

      //We need to calculate both B^TWB and B^TWY exhaustively
      FRegressionBaseB_TWY_(qnode,rnode);
      FRegressionBaseB_TWB_(qnode,rnode);
    }
  }
}


template <typename TKernel>

void FastRegression<TKernel>::PostProcess_(Tree *qnode){

  //Add owed to mass, and then postprocess the left and right children

  //mass_l <-owed_l+mass_l for B_TWB
  //mass_u <- owed_u+mass_u





  if(!qnode->is_leaf()){

    //Having completed all the tree calculations we now update bounds for both
    //the qunatities. this can be done by calling the UpdateBounds_ function
    //with the flag set to CHECK_FOR_PRUNE_BOTH
      
    UpdateBounds_(qnode, qnode->stat().b_twb_owed_l, 
		  qnode->stat().b_twb_owed_u, 
		  qnode->stat().b_twy_owed_l, 
		  qnode->stat().b_twy_owed_u, CHECK_FOR_PRUNE_BOTH);
    PostProcess_(qnode->left());
    PostProcess_(qnode->right());

    printf("Mass lower bound of bTWB is ..\n");
    qnode->stat().b_twb_mass_l.PrintDebug();

    printf("mass upper bound of BTWB is ..\n");
    qnode->stat().b_twb_mass_u.PrintDebug();


    printf("Mass lower bound of bTWY is ..\n");
    qnode->stat().b_twy_mass_l.PrintDebug();
    
    printf("mass upper bound of BTWY is ..\n");
    qnode->stat().b_twy_mass_u.PrintDebug();



  }

  else{

  UpdateBounds_(qnode, qnode->stat().b_twb_owed_l, 
		  qnode->stat().b_twb_owed_u, 
		  qnode->stat().b_twy_owed_l, 
		  qnode->stat().b_twy_owed_u, CHECK_FOR_PRUNE_BOTH);
    // b_twb_e_estimate= 
    //0.5*(b_twy_l_estimate+b_twy_u_estimate+b_twy_more_l+b_twy_more_u)
    for(index_t q=qnode->begin();q<qnode->end();q++){
     
      FastRegressionStat qstat=qnode->stat();
      Matrix estimate_mean1;
      la::AddInit(b_twb_l_estimate_[q],
		  b_twb_u_estimate_[q],&estimate_mean1);
      
      la::Scale(0.50,&estimate_mean1);
      
      Matrix more_mean1;
      la::AddInit(qstat.b_twb_more_l,qstat.b_twb_more_u,&more_mean1);
      la::Scale(0.50,&more_mean1);

      //Final estimate is just the sum of estimate_mean and more_mean

      la::AddOverwrite(estimate_mean1,more_mean1,&b_twb_e_estimate_[q]);


      // b_twy_e_estimate= 
      //0.5*(b_twy_l_estimate+b_twy_u_estimate+b_twy_more_l+b_twy_more_u)

      Matrix estimate_mean2;
      la::AddInit(b_twy_l_estimate_[q],
		  b_twy_u_estimate_[q],&estimate_mean2);

      la::Scale(0.50,&estimate_mean2);

      Matrix more_mean2;
      la::AddInit(qstat.b_twy_more_l,qstat.b_twy_more_u,&more_mean2);
      la::Scale(0.50,&more_mean2);

      //Final estimate is just the sum of estimate_mean and more_mean
     
      la::AddOverwrite(estimate_mean2,more_mean2,&b_twy_e_estimate_[q]);
    }
  }
}


template <typename TKernel>

void FastRegression<TKernel>::Print_(){
  
  //printf("Came to print function......\n");
  //This is just a test function and will be removed some time later

  FILE *fp;
  fp=fopen("fast_b_twy.txt","w+");

  for(index_t q=0;q<qset_.n_cols();q++){
    
    
    // printf("The BTWY estimate for q is..\n");
    b_twy_e_estimate_[q].PrintDebug(NULL,fp);
  }
  fclose(fp);

  FILE *gp;
  gp=fopen("fast_b_twb.txt","w+");

  for(index_t q=0;q<qset_.n_cols();q++){
    
    
    // printf("The BTWY estimate for q is..\n");
    b_twb_e_estimate_[q].PrintDebug(NULL,gp);
  }
  fclose(gp);
}


#endif











