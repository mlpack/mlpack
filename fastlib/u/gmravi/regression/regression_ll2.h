#ifndef REGRESSION_LL2_H
#define REGRESSION_LL2_H
#include "fastlib/fastlib_int.h"
#include "pseudo_inverse.h"

template <typename TKernel>
double FastRegression<TKernel>::Compute1NormLike_(Matrix &a){

  //This function computes treats the natrix as a vector and computes
  //it's 1-norm
  double value=0;
  for(index_t col=0;col<a.n_cols();col++){
    for(index_t row=0;row<a.n_rows();row++){
      value+=fabs(a.get(row,col));
    }
  }
  return value;
}

template <typename TKernel>
double FastRegression<TKernel>::SquaredFrobeniusNorm_(Matrix &a){

  double sqd_frobenius_norm=0.0;
  for(index_t col=0;col<a.n_cols();col++){
    //Along each column
    for(index_t row=0;row<a.n_rows();row++){
      //Alon each row

      sqd_frobenius_norm+=a.get(row,col)*a.get(row,col);
    }
  }
  return sqd_frobenius_norm;
}


// This function checks for the prunability of B^TWY
//This function depends on the pruning criteria
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
  
  //Set up the dl
  double m=kernel_value_range.lo;
  
  la::ScaleOverwrite(m, rnode->stat().b_ty, &dl);
  
  //The new upper bound. Set up the du

  m = -1+kernel_value_range.hi;
  la::ScaleOverwrite (m, rnode->stat().b_ty, &du);
    
  //Lets find the maximum error that one can commit due to pruning 
  // This is nothing but the B^TY matrix scaled by a value m defined below
    
  m=0.5*(kernel_value_range.hi-kernel_value_range.lo);
  Matrix max_error;
    
  //max_error <- m*(b_ty)
  la::ScaleInit (m, rnode->stat().b_ty, &max_error); 
    
  //Now lets calculate the allowed error. this depends on the 
  //allowed_error=tau*(rnode->stat().b_ty/rroot->stat().b_ty)(b_twy_mass_l+dl)
    
    
  //Lets define a matrix new_mass_l as shown below.This matrix takes 
  //into account the addditional dl that will be added if pruning takes 
  //place
    
    
  Matrix new_mass_l;
   
  la::AddInit(qnode->stat().b_twy_mass_l, dl, &new_mass_l);
    
  //Now it is easy to see that the allowed error is nothing but the 
  //scaled version of new_mass_l
    
  //Now depending on the pruning criteria we shall have a different
  //pruning condition Now we can prune only if max_error is

  //component wise lesser than allowed_error

  //This is for componentwise bounding.........................................
  if(pruning_criteria==CRITERIA_FOR_PRUNE_COMPONENT){
    
    for(index_t col=0;col<qnode->stat().b_twy_mass_u.n_cols();col++){
      
      for(index_t row=0;row<qnode->stat().b_twy_mass_u.n_rows();row++){
	
	//We will prune only if the matrix is prunable componenet wise
	//This is what test now
	
	//printf("tau_ is %f\n",tau_);
	
	
	double alpha=tau_*
	  (double)(rnode->stat().b_ty.get(row,col))/
	  (rroot_->stat().b_ty.get(row,col));

	
          
	if(max_error.get(row,col)>=alpha*new_mass_l.get(row,col)){
	  //pruning failed
	  //Set everything back to 0 and return
	  
	  dl.SetAll(0);
	  du.SetAll(0);
	  return 0;
	}
      }
    }
    //successfully pruned for component wise calculations
    return 1;
  }

  //THIS Means we are interested in the frobenius norm pruning................
  
  //Maximum allowed error is the sqaured frobenius norm of the matrix max_error

  double squared_frobenius_norm_of_max_error=SquaredFrobeniusNorm_(max_error);
  double ratio_of_1norms=
    Compute1NormLike_(rnode->stat().b_ty)/Compute1NormLike_(rroot_->stat().b_ty);  

  //Now lets calculate the squared frobenius norm of new_mass_l
  double sqd_frobenius_norm_of_new_mass_l=SquaredFrobeniusNorm_(new_mass_l);
 
  double allowed_error= ratio_of_1norms*tau_*sqd_frobenius_norm_of_new_mass_l;
  if(squared_frobenius_norm_of_max_error<=allowed_error)
    {
      //prune
      return 1;
    }
  else{
    //Dont prune. So set dl and du to 0
    
    dl.SetAll(0);
    du.SetAll(0);
    return 0;
  }

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

  if(pruning_criteria==CRITERIA_FOR_PRUNE_COMPONENT){
    
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
    //This means the matrix is compoenent wise prunable. hence return 1
    return 1;
  }
  //THIS MEANS WE ARE INTERESTED IN PRUNING BY FROBENIUS NORM
  //First lets calculate the squared frobenius norm of max_error.

  double squared_frobenius_norm_of_max_error=SquaredFrobeniusNorm_(max_error);
  double ratio_of_1norms=
    Compute1NormLike_(rnode->stat().b_tb)/Compute1NormLike_(rroot_->stat().b_tb); 
  //Now lets calculate the squared frobenius norm of new_mass_l

  double squared_frobenius_norm_of_new_mass_l=SquaredFrobeniusNorm_(new_mass_l);

  double allowed_error=tau_*ratio_of_1norms*squared_frobenius_norm_of_new_mass_l;

  if(squared_frobenius_norm_of_max_error > allowed_error){

    //then this matrix is NOT runable
    dl.SetAll(0);
    du.SetAll(0);
    return 0;
  }

  //This means that the quantity is prunable
  return 1;
}
  

/* The Update Boundws function is independent of the pruning criteria */

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
  

/* The Update Boundws function is independent of the pruning criteria */
template <typename TKernel>
void FastRegression<TKernel>::UpdateBoundsForPruningB_TWB_(Tree
							   *qnode, Matrix &dl_b_twb,
							   Matrix &du_b_twb){
  //In this function we shall update bounds of the quantity that is
  //prunable.  This is similar to the UpdateBounds_(..)
  //function. However this will be called to incoprorate changes in
  //bounds due to pruning
    
  // Changes the upper and lower bounds.
    
  //b_twb_mass_l <- b_twb_mass_l+dl
  //b_twb_mass_u <- b_twb_mass_u+du
    
 

  la::AddTo (dl_b_twb ,&(qnode->stat().b_twb_mass_l));
  la::AddTo(du_b_twb, &(qnode->stat().b_twb_mass_u));


    
  // for a leaf node, incorporate the lower and upper bound changes
  // into its additional offset
    
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
 

//This function depends on the pruning criteria........

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

  Matrix max_children;
  Matrix min_children;
  max_children.Init(parent_stat.b_twb_mass_l.n_rows(), 
		    parent_stat.b_twb_mass_l.n_cols());

  min_children.Init(parent_stat.b_twb_mass_l.n_rows(), 
		    parent_stat.b_twb_mass_l.n_cols());

  if(pruning_criteria==CRITERIA_FOR_PRUNE_COMPONENT){
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

  //THe pruning criteria is Frobenius norm pruning...

  else{
    //Will update the frobenius norm as follows
    // lower_of_parent=max (parent, min (children))  
    // upper_of_parent=min(parent, max(children))
    
    //lets first find out the squared frobenius norms of the lower
    //bound masses of both the children

    double sqd_frobenius_norm_of_left_child= SquaredFrobeniusNorm_(left_stat->b_twb_mass_l);
    double sqd_frobenius_norm_of_right_child=SquaredFrobeniusNorm_(right_stat->b_twb_mass_l);

    //Now compare it with that of the parent
    
    double sqd_frobenius_norm_of_parent=SquaredFrobeniusNorm_(parent_stat.b_twb_mass_l);

    if(sqd_frobenius_norm_of_left_child <sqd_frobenius_norm_of_right_child)
      {
	//This means the left child has lower frobenius norm compared
	//to the right child

	if(sqd_frobenius_norm_of_parent < sqd_frobenius_norm_of_left_child){

	  // Change the parents mass_l matrix
	  parent_stat.b_twb_mass_l.CopyValues(left_stat->b_twb_mass_l);
	}
	
      }
    else{

      //The right child has lesser valued frobenius norm for mass_l
      if(sqd_frobenius_norm_of_parent < sqd_frobenius_norm_of_right_child){
	
	// Change the parents mass_l matrix
	parent_stat.b_twb_mass_l.CopyValues(right_stat->b_twb_mass_l);
      }
     
    }

    //A similar logic holds for mass_u values

    sqd_frobenius_norm_of_left_child=SquaredFrobeniusNorm_(left_stat->b_twb_mass_u);
    sqd_frobenius_norm_of_right_child=SquaredFrobeniusNorm_(right_stat->b_twb_mass_u);

    //Now compare it with that of the parent

    sqd_frobenius_norm_of_parent=SquaredFrobeniusNorm_(parent_stat.b_twb_mass_u);

    if(sqd_frobenius_norm_of_left_child > sqd_frobenius_norm_of_right_child){

      //So the left child has higher frobenius norm
      if(sqd_frobenius_norm_of_parent > sqd_frobenius_norm_of_left_child){
	//the parent has a higher frobenius norm. So decrease it
	parent_stat.b_twb_mass_u.CopyValues(left_stat->b_twb_mass_u);
      }
      
    }
    else{
      //So the right child has higher frobenius norm.
      if(sqd_frobenius_norm_of_parent > sqd_frobenius_norm_of_right_child){
	//the parent has higher frobenius norm. Hence decrease it

	parent_stat.b_twb_mass_u.CopyValues(right_stat->b_twb_mass_u);

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


  if(pruning_criteria==CRITERIA_FOR_PRUNE_COMPONENT){
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
  //IF Pruning criteria is Frobenius Norm Pruning criteria.....
  else{

    //Will update the frobenius norm as follows
    // lower_of_parent=max (parent, min (children))  
    // upper_of_parent=min(parent, max(children))

    //lets first frind out the squared frobenius norms of the lower
    //bound masses of both the children

    double sqd_frobenius_norm_of_left_child=SquaredFrobeniusNorm_(left_stat->b_twy_mass_l);
    double sqd_frobenius_norm_of_right_child=SquaredFrobeniusNorm_(right_stat->b_twy_mass_l);

    //Now compare it with that of the parent
    
    double sqd_frobenius_norm_of_parent=SquaredFrobeniusNorm_(parent_stat.b_twy_mass_l);

    if(sqd_frobenius_norm_of_left_child <sqd_frobenius_norm_of_right_child)
      {
	//This means the left child has lower frobenius norm compared
	//to the right child

	if(sqd_frobenius_norm_of_parent < sqd_frobenius_norm_of_left_child){

	  // Change the parents mass_l matrix
	  parent_stat.b_twy_mass_l.CopyValues(left_stat->b_twy_mass_l);
	}
	else{
	  //Do nothing
	}
      }
    else{

      //The right child has lesser valued frobenius norm for mass_l
      if(sqd_frobenius_norm_of_parent < sqd_frobenius_norm_of_right_child){
	
	// Change the parents mass_l matrix
	parent_stat.b_twy_mass_l.CopyValues(right_stat->b_twy_mass_l);
      }
      else{
	//Do nothing
      } 
    }

    //A similar logic for mass_u values
    sqd_frobenius_norm_of_left_child=SquaredFrobeniusNorm_(left_stat->b_twy_mass_u);
    sqd_frobenius_norm_of_right_child=SquaredFrobeniusNorm_(right_stat->b_twy_mass_u);

    //Now compute it with that of the parent

    sqd_frobenius_norm_of_parent=SquaredFrobeniusNorm_(parent_stat.b_twy_mass_u);

    if(sqd_frobenius_norm_of_left_child < sqd_frobenius_norm_of_right_child){
      //This means that the right child has higher mass_u
      if(sqd_frobenius_norm_of_parent > sqd_frobenius_norm_of_right_child){

	//Decrease the mass_u value
	parent_stat.b_twy_mass_u.CopyValues(right_stat->b_twy_mass_u);

      }
    }
    else{
      //the left child has higher mass_u value
      if(sqd_frobenius_norm_of_parent > sqd_frobenius_norm_of_left_child){

	//Decrease the mass_u value
	parent_stat.b_twy_mass_u.CopyValues(left_stat->b_twy_mass_u);

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

//This function depends on the pruning criteria.......................
template <typename TKernel>

void FastRegression<TKernel>::FRegressionBaseB_TWY_(Tree *qnode, Tree *rnode){

  //subtract along each dimension as we are now doing exhaustive calculations

  //more_u <- more_u - rnode->stat().b_ty

  la::SubFrom (rnode->stat().b_ty, &qnode->stat().b_twy_more_u);

  //Having subtracted calculate B^TWY exhaustively
  //One can do this by using linear algebra routines available 
  //in LaPack. however we shall not use them because B can be a 
  //very large matrix
  
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

  if(pruning_criteria==CRITERIA_FOR_PRUNE_COMPONENT){
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
  else{

    //We are doing frobenius norm pruning
    /** get a tighter lower and upper boiounf by looping over each query
     *  point to find that particular query point that has the least sqd
     *  frobenius norm of the matrix b_twy_l_estimate_[q]+qnode->stat().b_twy_more_l
     */
    double min_norm=DBL_MAX;
    double max_norm=DBL_MIN;
    Matrix temp;
    
    //The min_pointer and the max_pointer store the index number of the
    //query point which possibly has the least and the highest squared frobenius norm
    
    index_t min_pointer;
    index_t max_pointer;
    
    temp.Init(rset_.n_rows()+1,1);
    for(index_t i=qnode->begin();i<qnode->end();i++){
      
      //look for the lower bound
      la::AddOverwrite(b_twy_l_estimate_[i],qnode->stat().b_twy_more_l,&temp);
      
      double var=SquaredFrobeniusNorm_(temp);
      if(var< min_norm){
	
	min_pointer=i;
	min_norm=var;
      }
      
      //Look for the upper bound
      la::AddOverwrite(b_twy_u_estimate_[i],qnode->stat().b_twy_more_u,&temp);
      var=SquaredFrobeniusNorm_(temp);
      if(var>max_norm){
	
	max_pointer=i;
	max_norm=var;
      }
      
    }
    
    //Once done with the looping process set up the mass_l and mass_u values
    
    la::AddOverwrite(b_twy_l_estimate_[min_pointer],qnode->stat().b_twy_more_l,&qnode->stat().b_twy_mass_l);
    la::AddOverwrite(b_twy_u_estimate_[max_pointer],qnode->stat().b_twy_more_u,&qnode->stat().b_twy_mass_u);
    
  }
}



template <typename TKernel>
void FastRegression<TKernel>::FRegressionBaseB_TWB_(Tree *qnode,Tree *rnode){


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

  if(pruning_criteria==CRITERIA_FOR_PRUNE_COMPONENT){
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
  //We are interested in frobenius norm pruning
  else{
    
    

 //We are doing frobenius norm pruning
  /** get a tighter lower and upper boiounf by looping over each query
   *  point to find that particular query point that has the least sqd
   *  frobenius norm of the matrix b_twy_l_estimate_[q]+qnode->stat().b_twy_more_l
  */
  double min_norm=DBL_MAX;
  double max_norm=DBL_MIN;
  Matrix temp;

  //The min_pointer and the max_pointer store the index number of the
  //query point which possibly has the least and the highest squared frobenius norm

  index_t min_pointer;
  index_t max_pointer;

  temp.Init(rset_.n_rows()+1,rset_.n_rows()+1);
  for(index_t i=qnode->begin();i<qnode->end();i++){

    //look for the lower bound
    la::AddOverwrite(b_twb_l_estimate_[i],qnode->stat().b_twb_more_l,&temp);
    
    double var=SquaredFrobeniusNorm_(temp);
    if(var< min_norm){

      min_pointer=i;
      min_norm=var;
    }

    //Look for the upper bound
    la::AddOverwrite(b_twb_u_estimate_[i],qnode->stat().b_twb_more_u,&temp);
    var=SquaredFrobeniusNorm_(temp);
    if(var>max_norm){
      
      max_pointer=i;
      max_norm=var;
    }
    
  }

  //Once done with the looping process set up the mass_l and mass_u values

  la::AddOverwrite(b_twb_l_estimate_[min_pointer],qnode->stat().b_twb_more_l,&qnode->stat().b_twb_mass_l);
  la::AddOverwrite(b_twb_u_estimate_[max_pointer],qnode->stat().b_twb_more_u,&qnode->stat().b_twb_mass_u);

  }
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


template <typename TKernel>
void FastRegression<TKernel>::ObtainRegressionEstimate_(){

  //With this we have the BTWB estimate for each query point by
  //calling the invert function I can invert them

  for(index_t q=0;q<qset_.n_cols();q++){

    PseudoInverse::FindPseudoInverse(qset_.n_cols(),b_twb_e_estimate_[q]);
  }

  //We now have both (B^TWB)-1 and (B^TWY). The Regression Estimates
  //will be obtained by simply multiplying these 2 matrices


  //So we now have for each query point the (B^TWB)-1 matrix and B^TWY
  //matrix. In order to get the regression estimates we perform
  //y_hat= [1,q] (B^TWB)^-1 (B^TWY) where q are the coordinates of the query
  //point

  //lets perform these multiplications by using LaPack utilities

  //The vector temp will hold the multiplication of (B^TWB)^-1 and
  //(B^TWY)


  ArrayList<Matrix> temp;
  temp.Init(qset_.n_cols());
  
  for(index_t q=0;q<qset_.n_cols();q++){

    la::MulInit (b_twb_e_estimate_[q],b_twy_e_estimate_[q], &temp[q]);
  }

  printf("The arraylist temp formed...\n");

  //Now lets form a matrix q_vector using the coordinates of the
  //query point
  
  
  Matrix q_matrix;
  
  //Initialize q matrix and set it up
  q_matrix.Init(1,qset_.n_rows()+1);

  //The first element of the q matrix is a 1
  q_matrix.set(0,0,1);
  
  
  for(index_t q=0;q<qset_.n_cols();q++){
    
    for(index_t col=0;col<qset_.n_rows();col++){
      
      q_matrix.set(0,col+1,qset_.get(col,q));
    }

      //Now lets multiply the resulting q_matrix with temp to get the
    //regression estimate

    Matrix temp2;
    la::MulInit(q_matrix,temp[q],&temp2);

    //This temp2 is the regression estimate for the query point q
    regression_estimate_[q]=temp2.get(0,0);
    printf("regression estimate is %f\n",regression_estimate_[q]);
  }
}

template <typename TKernel>
void FastRegression<TKernel>::PrintRegressionEstimate_(){

  FILE *gp;
  gp=fopen("regression_estimate.txt","w+");

  for(index_t q=0;q<qset_.n_cols();q++){

    for(index_t dim=0;dim<qset_.n_rows();dim++){ 

      fprintf(gp,"%f, ",qset_.get(dim,q));
    }
    fprintf(gp,"fast:%f\n",regression_estimate_[q]); 
  }
  fclose(gp);
}


#endif











