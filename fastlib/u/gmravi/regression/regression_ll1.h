#ifndef REGRESSION_LL1_H
#define REGRESSION_LL1_H
#include "regression_ll2.h"


/** In this code we shalll evaluate (B^TWB)^-1 by first calculating B^TWB
 *  and then inverting it by SVD. Note B^TWB is a D+1 * D+1 matrix where D
 *  is the dimensionality of the dataset. We calculate this matrix by
 *  naive method and by using dual tree recursion.
 *  Below we mention how to evaluate B^TWY by using naive matrix calculations
 */

//These are the Init functions of FastRegressionStat

template <typename TKernel>
void FastRegression<TKernel>::
FastRegressionStat::
Init (const Matrix & dataset, index_t & start, index_t & count){

  /* Init function of leaf node */
  
  int num_of_dimensions=dataset.n_rows();
  
  /*First initalize the matrices*/
  
  /* Lets first allocate memory to all those variables
   * which appear in the statistic of the tree
   */
  
  b_tb.Init(num_of_dimensions+1,num_of_dimensions+1);
  b_twb_mass_l.Init(num_of_dimensions+1,num_of_dimensions+1);
  
  
  b_twb_owed_l.Init(num_of_dimensions+1,num_of_dimensions+1);
  b_twb_owed_u.Init(num_of_dimensions+1,num_of_dimensions+1);
  
  
  /**all lower bound matices will be all 0's **/
  
  b_twb_mass_l.SetAll(0);
  b_twb_owed_l.SetAll(0);
  
  
  /**Set the upper bound matrix owed_u to 0.
   *The upper bound matrix mass_u will be set later
   */
  
  b_twb_owed_u.SetAll(0);   
  
  
  // And since it is a leaf node we have 2 more fields namely b_twb_more_l
  //and b_twb_more_u_. Set them up too
  
  
  b_twb_more_l.Init(num_of_dimensions+1,num_of_dimensions+1);
  b_twb_more_u.Init(num_of_dimensions+1,num_of_dimensions+1);
  
  
  b_twb_more_l.SetAll(0);
  b_twb_more_u.SetAll(0);
  
  
  b_ty.Init(num_of_dimensions+1,1);
  b_twy_mass_l.Init(num_of_dimensions+1,1);
  
  
  b_twy_owed_l.Init(num_of_dimensions+1,1);
  b_twy_owed_u.Init(num_of_dimensions+1,1);
  
  
  /**all lower bound matices will be all 0's **/
  
  b_twy_mass_l.SetAll(0);
  b_twy_owed_l.SetAll(0);
  
  
  /**Set the upper bound matrix owed_u to 0.
   *The upper bound matrix mass_u will be set later
   */
  
  b_twy_owed_u.SetAll(0);   
  
  
  // And since it is a leaf node we have 2 more fields namely b_twy_more_l_
  //and b_twy_more_u_. Set them up too
  
  
  b_twy_more_l.Init(num_of_dimensions+1,1);
  b_twy_more_u.Init(num_of_dimensions+1,1);
  
  
  b_twy_more_l.SetAll(0);
  b_twy_more_u.SetAll(0);
  
  
  //So now we have initialized all variables except b_tb.
  //Since this is a leaf node we do this explicitly
  
  // Lets do B^T B operation by using LaPack
  //We shall later change this code
  
  //The temporary apporach is to form a temp matrix called temp
  //which will hold all points in the reference set from index
  //node->start to node->end
  
  Matrix temp;
  dataset.MakeColumnSlice(start,count,&temp);
  
  Matrix B_t;
  B_t.Init(dataset.n_rows()+1,count);
  
  //Now I form the B^T matrix from temp.
  //B^T matrix has first rows full of 1's and all the rows are same as the
  //rows in temp
  
  for(index_t row=0;row<dataset.n_rows()+1;row++){
    
    for(index_t col=0;col<count;col++){
      
      if(row==0){
	/*Set the first row to to all 1's */
	
	B_t.set(row,col,1);
      }
      else
	{
	  B_t.set(row,col,temp.get(row-1,col));
	}
    }
  }
  
  Matrix temp2; //temp2 hold B
  la::TransposeInit(B_t,&temp2);
  la:: MulOverwrite(B_t,temp2,&b_tb);
  
  
}
    

/** This is the Init function of an internal node */

template <typename TKernel>

void FastRegression <TKernel>::
FastRegressionStat::
Init (const Matrix & dataset, index_t & start, index_t & count, 
      const FastRegressionStat &left_stat, const FastRegressionStat &right_stat){
     

  int num_of_dimensions=dataset.n_rows();
      
  /*First initalize the matrices*/
      
  /* Lets first allocate memory to all those variables
   * which appear in the statistic of the tree
   */
      
  b_tb.Init(num_of_dimensions+1,num_of_dimensions+1);
  b_twb_mass_l.Init(num_of_dimensions+1,num_of_dimensions+1);
  //b_twb_mass_u will be initialized later.....
      
  b_twb_owed_l.Init(num_of_dimensions+1,num_of_dimensions+1);
  b_twb_owed_u.Init(num_of_dimensions+1,num_of_dimensions+1);
      
      
  /**all lower bound matices will be all 0's **/
      
  b_twb_mass_l.SetAll(0);
  b_twb_owed_l.SetAll(0);
      
      
  /**Set the upper bound matrix owed_u to 0. The upper bound matrix mass_u
   * will be set later*/
      
  b_twb_owed_u.SetAll(0);   


  b_ty.Init(num_of_dimensions+1,1);
  b_twy_mass_l.Init(num_of_dimensions+1,1);
      
  //b_twy_mass_u will be initialized later
  b_twy_owed_l.Init(num_of_dimensions+1,1);
  b_twy_owed_u.Init(num_of_dimensions+1,1);
      
      
  /**all lower bound matices will be all 0's **/
      
  b_twy_mass_l.SetAll(0);
  b_twy_owed_l.SetAll(0);
      
      
  /**Set the upper bound matrix owed_u to 0. The upper bound matrix mass_u
   * will be set later*/
      
  b_twy_owed_u.SetAll(0);

  //Note the b^T B of the parent is the sum of B^T B values of the children
  //We use the LAPACk's AddOVerwrite function to get the B^TB of the parent
     
  la::AddOverwrite (left_stat.b_tb, right_stat.b_tb, &b_tb);
 
}

template <typename TKernel>
FastRegression<TKernel>::FastRegressionStat::FastRegressionStat (){
  
  
}

template <typename TKernel>
FastRegression <TKernel>::FastRegressionStat::~FastRegressionStat (){
 
  
}


//Intersting *Private* functions...................................

template <typename TKernel>
void FastRegression<TKernel>::ScaleDataByMinmax_ (){
  int num_dims = rset_.n_rows ();
  DHrectBound < 2 > qset_bound;
  DHrectBound < 2 > rset_bound;
  qset_bound.Init (qset_.n_rows ());
  rset_bound.Init (qset_.n_rows ());

  // go through each query/reference point to find out the bounds
  for (index_t r = 0; r < rset_.n_cols (); r++){
    Vector ref_vector;
    rset_.MakeColumnVector (r, &ref_vector);
    rset_bound |= ref_vector;
  }
  for (index_t q = 0; q < qset_.n_cols (); q++){
    Vector query_vector;
    qset_.MakeColumnVector (q, &query_vector);
    qset_bound |= query_vector;
  }

  for (index_t i = 0; i < num_dims; i++){
    DRange qset_range = qset_bound.get (i);
    DRange rset_range = rset_bound.get (i);
    double min_coord = min (qset_range.lo, rset_range.lo);
    double max_coord = max (qset_range.hi, rset_range.hi);
    double width = max_coord - min_coord;
      
    //printf ("Dimension %d range: [%g, %g]\n", i, min_coord, max_coord);
      
    for (index_t j = 0; j < rset_.n_cols (); j++){
      rset_.set (i, j, (rset_.get (i, j) - min_coord) / width);
    }

    if (strcmp (fx_param_str (NULL, "query", NULL),
		fx_param_str_req (NULL, "data"))){
      for (index_t j = 0; j < qset_.n_cols (); j++){
	qset_.set (i, j, (qset_.get (i, j) - min_coord) / width);
      }
    }
  }
}


 
/** This function finds the order in which a node nd should
 * be pruned when being comparee with nd1 and nd2. It is simply
 * based on the distance metric. Hence if nd1 is neared to node nd
 * then we investigate nd and nd1 else we investigate nodes nd and nd2

*/
template <typename TKernel>
void FastRegression<TKernel>::BestNodePartners_ (Tree * nd, Tree * nd1,
						 Tree * nd2,
						 Tree ** partner1,
						 Tree ** partner2){
    
  double d1 = nd->bound ().MinDistanceSq (nd1->bound ());
  double d2 = nd->bound ().MinDistanceSq (nd2->bound ());
    
  if (d1 <= d2){
    *partner1 = nd1;
    *partner2 = nd2;
  }
  else{
    *partner1 = nd2;
    *partner2 = nd1;
  }
}


// So lets preprocess the tree. We shall allocate memory for  
//all the variables defined in the statistic of the tree and 
//initialize them to proper values


/* getter functions */
 
template <typename TKernel>
Matrix & FastRegression<TKernel>::get_query_dataset(){

  return qset_;
} 
template <typename TKernel>
Matrix& FastRegression<TKernel>::get_reference_dataset(){

  return rset_;
}

template <typename TKernel>
ArrayList<index_t>& FastRegression<TKernel>::get_old_from_new_r(){
  
  return old_from_new_r_;
}

/** This function will be called after update bounds has been called in the 
 *  function Prunable. This will flush the owed values as they have already 
 * been incorporated in the mass values by the function Update Bounds

*/

template <typename TKernel>
Matrix& FastRegression<TKernel>::get_b_twy_estimates(index_t q){

  return b_twy_e_estimate_[q];
}
template <typename TKernel>
void FastRegression<TKernel>::FlushOwedValues_(Tree *qnode, 
					       check_for_prune_t flag){
    
  //We use the value of flag to flush the bounds of 
  //the appropriate quantity
    
  /**  flag=PRUNE_BOTH =>Flush owed values of B^TWB, B^TWY.
   *   flag=PRUNE_B_TWB =>Flush owed values of B^TWB
   *   flag=PRUNE_B_TWY =>Flush owed values of B^TWY
   */
  if(flag==CHECK_FOR_PRUNE_BOTH){

    //Flush owed values of B^TWB first

    qnode->stat().b_twb_owed_l.SetAll(0);
    qnode->stat().b_twb_owed_u.SetAll(0);

    //Now flush owed values of B^TWY

    qnode->stat().b_twy_owed_l.SetAll(0);
    qnode->stat().b_twy_owed_u.SetAll(0);

  }

  else{


    if(flag==CHECK_FOR_PRUNE_B_TWB){

      //Flush owed values of B^TWB 
	
      qnode->stat().b_twb_owed_l.SetAll(0);
      qnode->stat().b_twb_owed_u.SetAll(0);
	
    }

    else{

      //Flush owed values of B^TWY
	
      qnode->stat().b_twy_owed_l.SetAll(0);
      qnode->stat().b_twy_owed_u.SetAll(0);
    }
  }
}

template <typename TKernel>
void FastRegression<TKernel>::UpdateBounds_(Tree *qnode,  
					    Matrix &dl_b_twb, 
					    Matrix &du_b_twb, 
					    Matrix &dl_b_twy, 
					    Matrix &du_b_twy, 
					    check_for_prune_t flag){

  //We use the value of flag to update the bounds of the 
  //appropriate quantity

  /**  flag=PRUNE_BOTH => Update bounds of B^TWB, B^TWY.
   *   flag=PRUNE_B_TWB => Update Bounds of B^TWB
   *   flag=PRUNE_B_TWY => Update Bounds of B^TWY
   */
   
  

 

  if(flag==CHECK_FOR_PRUNE_BOTH){

      
    //Update B^TWY and B^TWB both

    //Add dl_b_twb and du_b_twb to mass_l and mass_u of B^TWB 

     
    la::AddTo(dl_b_twb, &qnode->stat().b_twb_mass_l);
    la::AddTo(du_b_twb, &qnode->stat().b_twb_mass_u);

    //Add dl_b_twy  and du_b_twy  to mass_l and mass_u of B^TWY

    la::AddTo(dl_b_twy, &qnode->stat().b_twy_mass_l);
    la::AddTo(du_b_twy, &qnode->stat().b_twy_mass_u);

       
    if(!qnode->is_leaf()){

     
	
      // transmit these values to the children node

      //Transmit B^TWB first 
	
      la::AddTo(dl_b_twb,&(qnode->left()->stat().b_twb_owed_l));
      la::AddTo(du_b_twb,&(qnode->left()->stat().b_twb_owed_u));
	
      la::AddTo(dl_b_twb,&(qnode->right()->stat().b_twb_owed_l));
      la::AddTo(du_b_twb,&(qnode->right()->stat().b_twb_owed_u));

      //Transmit B^TWY 

      la::AddTo(dl_b_twy,&(qnode->left()->stat().b_twy_owed_l));
      la::AddTo(du_b_twy,&(qnode->left()->stat().b_twy_owed_u));
	
      la::AddTo(dl_b_twy,&(qnode->right()->stat().b_twy_owed_l));
      la::AddTo(du_b_twy,&(qnode->right()->stat().b_twy_owed_u));

    }

    else{
	
      //printf("A child node..\n");
      /* in case of leaf nodes add these values to more_l and more_u */

      la::AddTo(dl_b_twb,&(qnode->stat().b_twb_more_l));
      la::AddTo(du_b_twb,&(qnode->stat().b_twb_more_u));
     

      la::AddTo(dl_b_twy,&(qnode->stat().b_twy_more_l));
      la::AddTo(du_b_twy,&(qnode->stat().b_twy_more_u));
     
    }
  }

  else{
    if(flag==CHECK_FOR_PRUNE_B_TWB){

      //Add dl_b_twb and du_b_twb to mass_l and mass_u of B^TWB 
	
      la::AddTo(dl_b_twb, &qnode->stat().b_twb_mass_l);
      la::AddTo(du_b_twb, &qnode->stat().b_twb_mass_u);

      if(!qnode->is_leaf()){
	
	// transmit these values to the children node
	    
	la::AddTo(dl_b_twb,&(qnode->left()->stat().b_twb_owed_l));
	la::AddTo(du_b_twb,&(qnode->left()->stat().b_twb_owed_u));
	  
	la::AddTo(dl_b_twb,&(qnode->right()->stat().b_twb_owed_l));
	la::AddTo(du_b_twb,&(qnode->right()->stat().b_twb_owed_u));
      }
	
      else{

	// in case of leaf nodes add these values to more_l and more_u 
	la::AddTo(dl_b_twb,&(qnode->stat().b_twb_more_l));
	la::AddTo(du_b_twb,&(qnode->stat().b_twb_more_u));
      }
    }

    else{
      //the flag is CHECK_FOR_PRUNE_B_TWY
	
      //Add dl_b_twy and du_b_twy to mass_l and mass_u of B^TWY
	
      la::AddTo(dl_b_twy, &qnode->stat().b_twy_mass_l);
      la::AddTo(du_b_twy, &qnode->stat().b_twy_mass_u);
	
      if(!qnode->is_leaf()){
	  
	// transmit these values to the children node
	  
	la::AddTo(dl_b_twy,&(qnode->left()->stat().b_twy_owed_l));
	la::AddTo(du_b_twy,&(qnode->left()->stat().b_twy_owed_u));
	  
	la::AddTo(dl_b_twy,&(qnode->right()->stat().b_twy_owed_l));
	la::AddTo(du_b_twy,&(qnode->right()->stat().b_twy_owed_u));
      }
	
      else{

	/* in case of leaf nodes add these values to more_l and more_u */
	la::AddTo(dl_b_twy,&(qnode->stat().b_twy_more_l));
	la::AddTo(du_b_twy,&(qnode->stat().b_twy_more_u));
      }

    }
  }

 
}



/** This function will check if the quantity that is represented by the 
 * value of flag is prunable or not. 
 */

template <typename TKernel>
typename FastRegression<TKernel>::prune_t 
FastRegression<TKernel>::Prunable_(Tree *qnode, Tree *rnode, Matrix &dl_b_twb, 
				   Matrix &du_b_twb, 
				   Matrix &dl_b_twy, 
				   Matrix &du_b_twy, 
				   check_for_prune_t flag){

  /*legend: 

  * flag=PRUNE_BOTH => Check for the prunability of B^TWB, B^TWY.
  * flag=PRUNE_B_TWB => Check for prunability of B^TWB
  * flag=PRUNE_B_TWY => Check for prunability of B^TWY
  */

  if(flag==CHECK_FOR_PRUNE_BOTH){

    //Lets check if B^TWY and B^TWB are both prunable
    //flag1 will be 1 if B^TWY is pruable
    // and flag2 will be 1 if B^TWB is prunable


    index_t flag1 = 
      FastRegression<GaussianKernel>::
      PrunableB_TWY_(qnode, rnode, dl_b_twy, du_b_twy);

    index_t flag2 = 
      FastRegression<GaussianKernel>::
      PrunableB_TWB_(qnode, rnode, dl_b_twb, du_b_twb);

    if(flag1==1&&flag2==1){

      /** this means both the quantities are prunable
       */ 

         
      return PRUNE_BOTH; 
    }
    else{
      if(flag1==1){
	  
	/** This means that only B_TWY is prunable
	 */
	  
	return PRUNE_B_TWY;
      }
      else{
	 
	if(flag2==1){ 
	  /** This means that only B_TWB is prunable
	   */
	  return PRUNE_B_TWB;
	}
	else{
	  /** None of the quantities is prunable
	   */

	  return PRUNE_NONE;
	}
	
      }
      
    }
  }
  else
    {
      if(flag==CHECK_FOR_PRUNE_B_TWB){
	  
	//lets  check if B^TWB is prunable or not
	//if flag1 is 1 then it is else it is not
	 
	index_t flag1 = FastRegression<GaussianKernel>::
	  PrunableB_TWB_(qnode,rnode,dl_b_twb,du_b_twb);

	if(flag1==1){
	  return PRUNE_B_TWB;

	}
	else{
	  //it is not prunable. 
	  return PRUNE_NOT_THIS;
	}
	
      }
      else{
	//flag is now set to CHECK_FOR_PRUNE_B_TWY

	index_t flag2=FastRegression<GaussianKernel>::PrunableB_TWY_(qnode,rnode,dl_b_twy,du_b_twy);

	if(flag2==1){
	  //this means B_TWY is prunable
	  return PRUNE_B_TWY;

	}
	else{
	  //it is not prunable. 

	  return PRUNE_NOT_THIS;
	}

      }
    }
}



template <typename TKernel>
void FastRegression<TKernel>::FRegression_(Tree *qnode, Tree *rnode, 
					   typename 
					   FastRegression<TKernel>::
					   check_for_prune_t flag){

  //The first thing to do is to update bounds.
  //We now have to use the flag to decide what
  //quantity we are checking if it is prunable.
  //
   UpdateBounds_(qnode, qnode->stat().b_twb_owed_l, 
		qnode->stat().b_twb_owed_u, 
		qnode->stat().b_twy_owed_l, 
		qnode->stat().b_twy_owed_u, flag);
    
  /** With this we have updated the appropriate quantity
   * Lets flush the owed values now as they were incorporated 
   * by the update bounds function 
   */
    
  
  FlushOwedValues_(qnode,flag);
 
  //With this the appropriate owed values have been flushed and the 
  //appropriate bounds updated with the call to update bounds 
  //function. 
    
  //Now lets check for the prunability depending on the value of flag
  //Before we do that we need to send in parameters. Depending on the 
  //value of flag only the appropriate parameters will be set up
    
  Matrix dl_b_twb;
  Matrix du_b_twb;

  Matrix dl_b_twy;
  Matrix du_b_twy;
    
  dl_b_twb.Init(rset_.n_rows()+1,rset_.n_rows()+1);
  du_b_twb.Init(rset_.n_rows()+1,rset_.n_rows()+1);
    
  dl_b_twy.Init(rset_.n_rows()+1,1);
  du_b_twy.Init(rset_.n_rows()+1,1);
    
  dl_b_twb.SetAll(0);
  du_b_twb.SetAll(0);
    
  dl_b_twy.SetAll(0);
  du_b_twy.SetAll(0);
    
 
  //Lets send in all the above defined parameters, though depending on the 
  //value of flag not all parameters are necessary. however I am not able 
  //to think of anything better. the function prunable will tell us  
  //which quantity to prune
  //This is exactly what the enum variable what_is_prunable will tell us

  prune_t what_is_prunable = Prunable_(qnode, rnode, dl_b_twb, du_b_twb,
				       dl_b_twy, du_b_twy, flag);

 
 
  if(what_is_prunable==PRUNE_BOTH){
      
    //This menas both B^TWB and B^TWY are prunable. 
    //we first UpdateBounds for 
    //pruning by calling the function as declared below

    printf("qnode start is %d\n",qnode->begin());
    printf("Both qunats are prunable is prunable..\n");
    printf("Hence exiting..\n");
    exit(0);
    
      
    UpdateBoundsForPruningB_TWB_(qnode, dl_b_twb, du_b_twb);
    UpdateBoundsForPruningB_TWY_(qnode, dl_b_twy, du_b_twy);
    
    //Check if qnode is a leaf or not. if it is not call mergebounds
    if(!qnode->is_leaf()){
      FastRegressionStat &stat=qnode->stat();
      
      FastRegressionStat *left_stat=NULL;
      FastRegressionStat *right_stat=NULL;
       
      right_stat= &(qnode->right()->stat());
      left_stat= &(qnode->left()->stat());
      MergeChildBounds_(left_stat,right_stat,stat,flag);
    }
    return;
  }

  else{
    if(what_is_prunable==PRUNE_B_TWB){
      
      printf("qnode start is %d\n",qnode->begin());
      printf("BTWB is prunable..\n");
      printf("Hence exiting..\n");
      printf("flag is %d\n",flag);
     
   
      //Since B^TWB was prunable therefore prune this
      UpdateBoundsForPruningB_TWB_(qnode, dl_b_twb, du_b_twb);
      printf("Updated bounds..\n");
      //Check if qnode is a leaf or not. if it is not call mergebounds
      if(!qnode->is_leaf()){
	FastRegressionStat &stat=qnode->stat();
	
	FastRegressionStat *left_stat=NULL;
	FastRegressionStat *right_stat=NULL;
	
	right_stat= &(qnode->right()->stat());
	left_stat= &(qnode->left()->stat());

	MergeChildBounds_(left_stat,right_stat,stat,flag);
	printf("Merged child bounds..\n");
      }

      //Now if the variable *flag* passed to the function is
      //CHECK_FOR_PRUNE_B_TWB then our job is done and hence we return
      //However if it is CHECK_FOR_PRUNE_BOTH then our job is still 
      //not done and we will need to recurse
      
      if(flag==CHECK_FOR_PRUNE_B_TWB){
	
	return;
      }
      else{

	//the flag is CHECK_FOR_PRUNE_BOTH. 
	//AS B^TWY is still not prunable we recurse

	flag=CHECK_FOR_PRUNE_B_TWY;
	//CallRecursively_(qnode,rnode,flag);
	printf("will do recursion for flag=check-for-prune-btwy..\n");
	FRegression_(qnode, rnode, flag);


	//Check if qnode is a leaf or not. if it is not call mergebounds
	if(!qnode->is_leaf()){
	  FastRegressionStat &stat=qnode->stat();
	  
	  FastRegressionStat *left_stat=NULL;
	  FastRegressionStat *right_stat=NULL;
	  
	  right_stat= &(qnode->right()->stat());
	  left_stat= &(qnode->left()->stat());
	  MergeChildBounds_(left_stat,right_stat,stat,flag);
	}
	return; 
      }
    }

    else{ //Now what is prunable!=B_TWB
      
      if(what_is_prunable==PRUNE_B_TWY){

	printf("qnode start is %d\n",qnode->begin());
	printf("BTWB is prunable..\n");
	printf("Hence exiting..\n");
	exit(0);
	  
	//Since B^TWY was prunable therefore prune this
	  
	UpdateBoundsForPruningB_TWY_(qnode, dl_b_twy, du_b_twy);

	//Check if qnode is a leaf or not. if it is not call mergebounds
	if(!qnode->is_leaf()){
	  FastRegressionStat &stat=qnode->stat();
	  
	  FastRegressionStat *left_stat=NULL;
	  FastRegressionStat *right_stat=NULL;
	  
	  right_stat= &(qnode->right()->stat());
	  left_stat= &(qnode->left()->stat());
	  MergeChildBounds_(left_stat,right_stat,stat,flag);
	}

	  
	//Now if the variable *flag* passed to the function is
	//CHECK_FOR_PRUNE_B_TWY then our job is done and hence we return
	//However if it is CHECK_FOR_PRUNE_BOTH then our job is still 
	//not done and we will need to recurse
	  
	if(flag==CHECK_FOR_PRUNE_B_TWY){
	    
	  return;
	}
	else{
	    
	  //the flag is CHECK_FOR_PRUNE_BOTH. 
	  //AS BTWB is still not prunable we recurse
	    
	  flag=CHECK_FOR_PRUNE_B_TWB;
	  //CallRecursively_(qnode,rnode,flag);
	  FRegression_(qnode, rnode, flag);

	  //Check if qnode is a leaf or not. if it is not call mergebounds
	  if(!qnode->is_leaf()){
	    FastRegressionStat &stat=qnode->stat();
	    
	    FastRegressionStat *left_stat=NULL;
	    FastRegressionStat *right_stat=NULL;
	    
	    right_stat= &(qnode->right()->stat());
	    left_stat= &(qnode->left()->stat());
	    MergeChildBounds_(left_stat,right_stat,stat,flag);
	  }
	}	  
      }

      else{//prunable is not!=B_TWB and !=B_TWY
	if(what_is_prunable==PRUNE_NONE){

	  //printf("foun none to be prunable...\n");

	  //This means the flag was CHECK_FOR_PRUNE_BOTH and 
	  //none of B^TWB and B^TWY were prunable.
	  //Hence we need to recuse

	  flag=CHECK_FOR_PRUNE_BOTH;
	  //CallRecursively_(qnode,rnode,flag);
	}
	else{
	    
	  //this means what is prunable is PRUNE_NOT_THIS
	  //This means that the flag was either CHECK_FOR_PRUNE_B_TWB
	  //or CHECK_FOR_PRUNE_B_TWY, and that particular quantity 
	  //was not prunable
	  //hence we continue recursing.
	  //Note here the flag remains just the same thing, 
	  //as the qunatity is still
	  //not prunable

	  //printf("B_NOT THIS found\n");
	 
      
	  //CallRecursively_(qnode,rnode,flag);
	    
	}

	//So the pruning failed on atleast 1 qunatity. Hence we shall
	//now test for the exhaustive case or go for pairwise
	//recursion

	FastRegressionStat &stat=qnode->stat();
	
	FastRegressionStat *left_stat=NULL;
	FastRegressionStat *right_stat=NULL;
	
	if(!qnode->is_leaf()){
	  
	  right_stat= &(qnode->right()->stat());
	  left_stat= &(qnode->left()->stat());
	}

	if(qnode->is_leaf()){
	  
	  if(rnode->is_leaf()){
	    
	    /* This is the Base Case */
	    //printf("Hey man I hit the leaf nodes..\n");
	    FRegressionBase_(qnode,rnode,flag);
	    return;
	  }
	  else{
	    //printf("qnode is a leaf and rnode is not leaf...\n");
	    /* rnode is not a leaf node */
	    Tree *rnode_first = NULL, *rnode_second = NULL;
	    BestNodePartners_(qnode, rnode->left (), rnode->right (),
			      &rnode_first, &rnode_second);
	    FRegression_(qnode, rnode_first,flag);
	    FRegression_(qnode, rnode_second,flag);
	    return;
	  }
	}
	
	/* qnode is not a leaf node */
	else{
	  
	  if(rnode->is_leaf()){
	    //printf("qnode is not a leaf but rnode is ...\n");
	    Tree *qnode_first = NULL, *qnode_second = NULL;
	    BestNodePartners_(rnode, qnode->left (), qnode->right (),
			      &qnode_first, &qnode_second);
	    FRegression_(qnode_first,rnode,flag);
	    FRegression_(qnode_second,rnode,flag);
	  }
	  else{
	    //printf("Both are non-leaf nodes...\n");
	    /* Both are non-leaf nodes */
	    Tree *rnode_first = NULL, *rnode_second = NULL;
	    
	    BestNodePartners_(qnode->left (), rnode->left (), rnode->right (),
			      &rnode_first, &rnode_second);
	    FRegression_(qnode->left (), rnode_first,flag);
	    FRegression_(qnode->left (), rnode_second,flag);
	    
	    BestNodePartners_(qnode->right (), rnode->left (),rnode->right (), 
			      &rnode_first, &rnode_second);
	    FRegression_(qnode->right (), rnode_first,flag);
	    FRegression_(qnode->right (), rnode_second,flag);
	  }
	  
	  /* this will now update the bounds of the parent by using 
	   * the values of the children node
	   */
	  MergeChildBounds_(left_stat,right_stat,stat,flag);
	}
      }
    }
  }
}

template <typename TKernel>

void FastRegression<TKernel>::Compute(){
    
  //So having initialized all values we now call the *main" function
  //Which calculates the matrix B^TWB and B^TWY

  //At the start of the program we would like to check if both the 
  //quantities B^TWB and B^TWY are prunable
  
  check_for_prune_t flag=CHECK_FOR_PRUNE_BOTH;
  FRegression_(qroot_,rroot_,flag);
  PostProcess_(qroot_);
  Print_();

  printf("At the end of regression...\n");
  printf("BTB is...\n");
  rroot_->stat().b_tb.PrintDebug();

  printf("BTY is......\n");
  rroot_->stat().b_ty.PrintDebug();
 
}
 
template <typename TKernel>
void  FastRegression<TKernel>::CalculateB_TYRecursively_(Tree *rnode){

  /** The base case */

  if(rnode->is_leaf()){
   
    Matrix temp;
    rset_.MakeColumnSlice(rnode->begin(),rnode->end()-rnode->begin(),&temp);
      
    Matrix B_t;
    B_t.Init(rset_.n_rows()+1,rnode->end()-rnode->begin());
      
    //Now I form the B^T matrix from temp.
    //B^T matrix has first rows full of 1's and all the rows are same as the
    //rows in temp

    for(index_t row=0;row<rset_.n_rows()+1;row++){
    
      for(index_t col=0;col<rnode->end()-rnode->begin();col++){

	if(row==0){
	  /*Set the first row to to all 1's */

	  B_t.set(row,col,1);
	}
	else
	  {
	    B_t.set(row,col,temp.get(row-1,col));
	  }
      }
    }
    

    //Matrix Y is a column which has its elements as the weights of the
    //reference points in the node. Hence I form Y as a 1-column matrix
    //by copying the reference weights from the vector rset_weights_
    Matrix Y;
    Y.Init(rnode->end()-rnode->begin(),1);
    index_t j=0;

    for(index_t i=rnode->begin();i<rnode->end();i++){
	
	
      Y.set(j,0,rset_weights_[old_from_new_r_[i]]);
      j++;
    }
     
    //Finally Multiply B^T with Y
    la:: MulOverwrite(B_t,Y,&rnode->stat().b_ty);

      
  }
  /*node is not a leaf node */
  else{

    //B^T Y is nothing but the sum of the B^T Y values of the children
    CalculateB_TYRecursively_(rnode->left());
    CalculateB_TYRecursively_(rnode->right());
    la::AddOverwrite (rnode->left()->stat().b_ty,
		      rnode->right()->stat().b_ty, &(rnode->stat().b_ty));

      
  }

}

template <typename TKernel>
void FastRegression<TKernel>::SetUpperBounds_(Tree *node){



  if(node->is_leaf()){
    node->stat().b_twy_mass_u.Copy(rroot_->stat().b_ty);
  
    node->stat().b_twb_mass_u.Copy(rroot_->stat().b_tb);
   
  }
  else
    {
      node->stat().b_twy_mass_u .Copy(rroot_->stat().b_ty);
      node->stat().b_twb_mass_u.Copy(rroot_->stat().b_tb);
      SetUpperBounds_(node->left());
      SetUpperBounds_(node->right());
    }
}

template <typename TKernel>

void FastRegression<TKernel>::Init(Matrix &q_matrix, Matrix &r_matrix, 
				   double bandwidth,
				   double tau,
				   index_t leaf_length,
				   Vector &rset_weights){

  //Set up value of tau,qset_,rset_


  qset_.Alias(q_matrix);
  rset_.Alias(r_matrix);
  tau_=tau;
  index_t leaflen=leaf_length;
  rset_weights_.Alias(rset_weights);
   
  /* Construct Query and Reference trees */
  rroot_ = tree::MakeKdTreeMidpoint < Tree >
    (rset_, leaflen, &old_from_new_r_, &new_from_old_r_);

  qroot_=tree::MakeKdTreeMidpoint < Tree >
    (qset_, leaflen, NULL, NULL);   

  //printf("Tree is..\n");
  //qroot_->Print();

  /**Note the init function of the statistics of the node 
   * calculates the
   * value of B^TB bottom up recursively. however we cannot 
   * calculate B^TY
   * in the init function of the node statistic as the function 
   * doesnt have 
   * access to the "Y" vector
   * hence lets fo these calculation in separate function 
   * declared below
   */
  CalculateB_TYRecursively_(rroot_);
  /** Similairy mass_l and mass_u of B^TWB and B^TWY cant be 
   *  calculated in the 
   * init function of the node statistic as these values depend 
   * on the value of the
   * root node's B^TY and B^TB. hence they get calculated in a 
   * function declared 
   * below 
   */

  SetUpperBounds_(qroot_);
 
 
  //Initialize everything else.......

  b_twb_l_estimate_.Init(qset_.n_cols());
  b_twb_u_estimate_.Init(qset_.n_cols());
  b_twb_e_estimate_.Init(qset_.n_cols());
    
  int num_of_dimensions=rset_.n_rows();
  for(index_t i=0;i<qset_.n_cols();i++){

    /* Initialize each of them to the size of (D+1)*(D+1) */
    b_twb_l_estimate_[i].Init(num_of_dimensions+1,num_of_dimensions+1);
    b_twb_u_estimate_[i].Init(num_of_dimensions+1,num_of_dimensions+1);
    b_twb_e_estimate_[i].Init(num_of_dimensions+1,num_of_dimensions+1);
  }
   
  for(index_t i=0;i<qset_.n_cols();i++){

    /* Initialize each of them to all 0's */
    b_twb_l_estimate_[i].SetAll(0);
    b_twb_e_estimate_[i].SetAll(0);
  }
   
  b_twy_l_estimate_.Init(qset_.n_cols());
  b_twy_u_estimate_.Init(qset_.n_cols());
  b_twy_e_estimate_.Init(qset_.n_cols());    
   
  for(index_t i=0;i<qset_.n_cols();i++){

    /* Initialize each of them to the size of (D+1)*1 */
    b_twy_l_estimate_[i].Init(num_of_dimensions+1,1);
    b_twy_u_estimate_[i].Init(num_of_dimensions+1,1);
    b_twy_e_estimate_[i].Init(num_of_dimensions+1,1);
  }
    
  for(index_t i=0;i<qset_.n_cols();i++){

    /* Initialize each of them to all 0's */
    b_twy_l_estimate_[i].SetAll(0);
    b_twy_e_estimate_[i].SetAll(0);

  }
    
    
  //However we still have 2 quantities uninitialized namely b_twb_u_
  //and b_twy_u_. this is because they depend on the values of b_tb
  //and b_ty of the root node of the tree formed by the reference set.
  //As the tree is now complete lets set up these values

  for(index_t i=0;i<qset_.n_cols();i++){
    
    b_twy_u_estimate_[i].CopyValues(rroot_->stat().b_ty);
    b_twb_u_estimate_[i].CopyValues(rroot_->stat().b_tb);
  
  }

  // initialize the kernel
 
    
  kernel_.Init (bandwidth);
  printf("Kernel Initialized...\n");
    
}

#endif
