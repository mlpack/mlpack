#ifndef DTE_H
#define DTE_H
#define LEAFLEN 1
#include "fastlib/fastlib.h"
template<typename TKernel> class DTreeEvaluation{
 public:

  //Forward decalaration of class
  class DTreeStat;
  typedef BinarySpaceTree < DHrectBound <2>, Matrix, DTreeStat> Tree;
  
  class DTreeStat{
    
  public:
    
    // The lower bound on mass of the reference node
    double mass_l;
    
    //The upper bound on the mass of the reference node
    
    double mass_u;
    
    //lower bound offset being passed on by the parent
    double owed_l;
    
    //Upper bound offset being passed on by the parent
    double owed_u;
    
    //Additional lower bound offset being sent to the leaf node by the
    //ancestor nodes. Note this is only for the LEAF NODE
    
    double more_l;
    
    
    //Additional upper bound offset being sent to the leaf node by the
    //ancestor nodes. This is is only for THE LEAF NODES
    
    double more_u;
    
    //The Init functions


    void Init(){
      
      //Need to set up all stat variables
      
      
    }
    
    // This is the leaf node
    void Init(const Matrix &dataset, index_t start, index_t count){
      
      
    }
    
    //This is the internal node
    void Init(const Matrix &dataset, index_t start, index_t count, 
	      const DTreeStat &left_stat, DTreeStat &right_stat){
      
    }

    /*//This is the leaf node
      
    void Init(const Matrix &dataset, index_t start, index_t end){
    
    }

    // This is the leaf node
    void Init(const Matrix &dataset, index_t start, index_t count){
    
    
    }
    
    //This is the internal node
    void Init(const Matrix &dataset, index_t start, index_t count, 
    const DTreeStat &left_stat, const DTreeStat &right_stat){
    
    }*/
  };

  //Private members of the class DTreeEvaluation
 private:
  
  //The query and reference trees. I am using different trees because
  //they are built out of different sets of points
  
  Tree *qroot_;
  Tree *rroot_;
  
  //The query set and the reference sets
  
  Matrix qset_;
  Matrix rset_;
  
  //The Kernel
  
  TKernel kernel_;
  
  //The tolerance
  double tau_;
  
  //Permutation of the Query set
  ArrayList <index_t> old_from_new_q_;
  ArrayList <index_t> new_from_old_q_;

  
  //Permutation of the reference set
  ArrayList <index_t> old_from_new_r_;
  ArrayList <index_t> new_from_old_r_;

  //Vector of estimates. Since these estimates are like density
  //calculations, I shall call these estimates as density estimates
  
  Vector density_estimates_l_;
  Vector density_estimates_u_;
  Vector density_estimates_e_;
  
  //The number of prunes
  
  index_t number_of_prunes_;
  
  //Bandwdth of the kernel being use
  
  double bandwidth_;
  
 private:


  void MergeChildBounds_(DTreeStat &parent_stat,DTreeStat &left_stat,
			 DTreeStat &right_stat){

    parent_stat.mass_u=
      min(parent_stat.mass_u,
	  max(left_stat.mass_u,right_stat.mass_u));

    parent_stat.mass_l=
      max(parent_stat.mass_l,
	  min(left_stat.mass_l,right_stat.mass_l));
  }

  /** determine which of the node to expand first */
  void BestNodePartners_(Tree * nd, Tree * nd1, Tree * nd2, 
			 Tree ** partner1,Tree ** partner2){
    
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
  
  void PreProcessRTree_(Tree *rnode){

    //Nothing to be done here 
  }
  void PreProcessQTree_(Tree *qnode){
    
    qnode->stat().mass_l=0;
   
    //mass_u will be set to the number of reference points

    qnode->stat().owed_l=0;
    qnode->stat().owed_u=0;

    if(qnode->is_leaf()){
      qnode->stat().more_l=0;
      qnode->stat().more_u=0;
      return;
    }

    //Preprocess the left and right children
    else{
      PreProcessQTree_(qnode->left());
      PreProcessQTree_(qnode->right());
    }
  }

  void UpdateBounds_(Tree *node, double dl,double du){

    //printf("dl in update bounds is %f...\n",dl);
    //    printf("du in update bounds is %f...\n",du);

    node->stat().mass_l+=dl;
    node->stat().mass_u+=du;
    
    //If it is a non-leaf node then propogate this change to its
    //children

    if(!node->is_leaf()){

      //Update left children

      node->left()->stat().owed_l+=dl;
      node->right()->stat().owed_l+=dl;

      node->left()->stat().owed_u+=du;
      node->right()->stat().owed_u+=du;
    }
    else{
      //This node is a leaf

      node->stat().more_l+=dl;
      node->stat().more_u+=du;
    }
    return;
  }

  index_t Prunable_(Tree *qnode, Tree *rnode, double *dl,double *du){

    //Get maximum and minimum bounding distance
    double min_dist=qnode->bound().MinDistanceSq(rnode->bound());
    double max_dist=qnode->bound().MaxDistanceSq(rnode->bound());

    //Get the range of kernel values

    double max_kernel_value=kernel_.EvalUnnormOnSq(min_dist);
    double min_kernel_value=kernel_.EvalUnnormOnSq(max_dist);

    //The new lower bound after incorporating the change

    index_t num_points_rnode=rnode->end()-rnode->begin();

    *dl=min_kernel_value*num_points_rnode;
    *du=num_points_rnode*(-1+max_kernel_value);

    double new_mass_l=qnode->stat().mass_l+(*dl);

    double m=0.5*(max_kernel_value-min_kernel_value);
    double total_error=m*num_points_rnode;
    index_t total_num_of_points=rroot_->end()-rroot_->begin();

   
    double allowed_error=
      tau_*new_mass_l*num_points_rnode/total_num_of_points;

   
    if(total_error<allowed_error){

      number_of_prunes_++;
      return 1;
    }
    else{

      *dl=*du=0;
      return 0;
    }
  }

  void SetTheUpperBoundInQueryTree_(Tree *qnode){
    
    //The total number of points in the 
    qnode->stat().mass_u=rroot_->end()-rroot_->begin();

    if(qnode->is_leaf()){

      return;
    }
    else{

      //Set the upper bound of left child
       SetTheUpperBoundInQueryTree_(qnode->left());

       //Set the upper bound of right child
       SetTheUpperBoundInQueryTree_(qnode->right());  
    }
  }

  void FastDTreeEvaluation_(Tree *qnode,Tree *rnode){

    //The first step is to Update Bounds
   
    UpdateBounds_(qnode,qnode->stat().owed_l,qnode->stat().owed_u);

    //Flush owed_l and owed_u

    qnode->stat().owed_l=0;
    qnode->stat().owed_u=0;

    //Check For prunability

    double dl=0;
    double du=0;
    if(Prunable_(qnode,rnode,&dl,&du)){

      //Prune and Return
      UpdateBounds_(qnode,dl,du);
      
      if(!qnode->is_leaf()){
	MergeChildBounds_(qnode->stat(),qnode->left()->stat(),
			  qnode->right()->stat());
      }
      return;
    }

    else{
      
      if(qnode->is_leaf()&&rnode->is_leaf()){
	
	//Both nodes are leaf nodes. So hit the base case
	
	FastDTreeEvaluationBase_(qnode,rnode);
      }

      //Recurse Appropriately otherwise
      else{
	
	//rnode is not a leaf
	if(qnode->is_leaf()&&!rnode->is_leaf()){
	  
	  Tree *rnode_first=NULL;
	  Tree *rnode_second=NULL;

	  BestNodePartners_(qnode, rnode->left (), rnode->right (),
			    &rnode_first, &rnode_second);

	  FastDTreeEvaluation_(qnode,rnode_first);
	  FastDTreeEvaluation_(qnode,rnode_second);
	}
	else{

	  //qnode is not a leaf
	  if(!qnode->is_leaf()&&rnode->is_leaf()){
	    
	    Tree *qnode_first=NULL;
	    Tree *qnode_second=NULL;
	    
	    BestNodePartners_(rnode, qnode->left(),qnode->right(),
			      &qnode_first, 
			      &qnode_second);
	    
	    FastDTreeEvaluation_(qnode_first,rnode);
	    FastDTreeEvaluation_(qnode_second,rnode);
	  }
	 
	  else{
	    //Both are non-leaf nodes

	    Tree *rnode_first=NULL;
	    Tree *rnode_second=NULL;
	    
	    //Best Partner with qnode->left()
	    BestNodePartners_(qnode->left(),rnode->left(),rnode->right(),
			      &rnode_first,&rnode_second);
	    FastDTreeEvaluation_(qnode->left(),rnode_first);
	    FastDTreeEvaluation_(qnode->left(),rnode_second);
	    
	    //Best Partner with qnode->right()
	    BestNodePartners_(qnode->right(),rnode->left(),rnode->right(),
			      &rnode_first,&rnode_second);
	    FastDTreeEvaluation_(qnode->right(),rnode_first);
	    FastDTreeEvaluation_(qnode->right(),rnode_second);
	  }
	  //Update the bounds of the parent node using bounbds of the
	  //children node

	  MergeChildBounds_(qnode->stat(),qnode->left()->stat(),
			    qnode->right()->stat());
	}
      }
    }
  }

 public:

 //The constructor......
  DTreeEvaluation(){

  }


  ~DTreeEvaluation(){

    //delete both the trees

    delete(qroot_);
    delete(rroot_);
  }

  ///Getters and setters

  void GetEstimatesInitialized(index_t p, Vector &row_p){

    //depermute them before sending

    Vector temp;
    temp.Init(qset_.n_cols());

    for(index_t q=0;q<qset_.n_cols();q++){

      temp[q]=density_estimates_e_[new_from_old_q_[q]];
    }
    
    //printf("The estimates are..\n");
    //temp.PrintDebug();
    row_p.CopyValues(temp);

    //    printf("Number of Prunes:%d. ...\n",number_of_prunes_);


  }  

  void PostProcess_(Tree *qnode){
    
    UpdateBounds_(qnode,qnode->stat().owed_l,qnode->stat().owed_u);

    if(qnode->is_leaf()){
      for(index_t i=qnode->begin();i<qnode->end();i++){
	
	density_estimates_e_[i]=
	  (density_estimates_l_[i]+qnode->stat().more_l
	   +density_estimates_u_[i]+qnode->stat().more_u)/2;
      }
    }
    else{
      PostProcess_(qnode->left());
      PostProcess_(qnode->right());   
    }
  }
  
  void FastDTreeEvaluationBase_(Tree *qnode,Tree *rnode){
    
    //This module has to be written
    
    index_t num_points_rnode=rnode->end()-rnode->begin();
    qnode->stat().more_u-=num_points_rnode;
    for(index_t q=qnode->begin();q<qnode->end();q++){
      
      double *point1=qset_.GetColumnPtr(q);
      double total_kernel_contrib=0;
      
      //Sum up contribution due to all reference points
      for(index_t r=rnode->begin();r<rnode->end();r++){
	
	double *point2=rset_.GetColumnPtr(r);
	
	double sqd_distance=
	  la::DistanceSqEuclidean(rset_.n_rows(),point1,point2);
	
	double kernel_value=kernel_.EvalUnnormOnSq(sqd_distance);
	total_kernel_contrib+=kernel_value;
      }

      density_estimates_l_[q]+=total_kernel_contrib;
      density_estimates_u_[q]+=total_kernel_contrib;
    }

    //Tighten the bounds
    double min_l=DBL_MAX;
    double max_u=DBL_MIN;
      for(index_t i=qnode->begin();i<qnode->end();i++){

	if(min_l>density_estimates_l_[i]){

	  min_l=density_estimates_l_[i];
	}
	if(max_u<density_estimates_u_[i]){
	  
	  max_u=density_estimates_u_[i];
	}
	qnode->stat().mass_l=min_l+qnode->stat().more_l;
	qnode->stat().mass_u=max_u+qnode->stat().more_u;
      }
  }
  void Compute(){

    //Preprocess Query tree
    PreProcessQTree_(qroot_);
    
    //Preprocess the reference tree
    PreProcessRTree_(rroot_);

    //Set the upper bound at each level of the query tree

    SetTheUpperBoundInQueryTree_(qroot_);

    FastDTreeEvaluation_(qroot_,rroot_);
    
    PostProcess_(qroot_);
    
    //Note my equations do not require me to multiply with a
    //normalization consant and hence I shall not perform the
    //normalization operation
    
  }
  
  void Init(Matrix &qset, Matrix &rset,double bandwidth,double tau){
    
    //Copy the datasets
    
    qset_.Copy(qset);
    rset_.Copy(rset);

    //Copy the bandwdith
    bandwidth_=bandwidth;
    
    tau_=tau;
    number_of_prunes_=0;
    
    //Set up the trees
    
    rroot_=tree::MakeKdTreeMidpoint < Tree> (rset_,LEAFLEN,
					     &old_from_new_r_,
					     &new_from_old_r_);

    qroot_=tree::MakeKdTreeMidpoint <Tree> (qset_, LEAFLEN, &old_from_new_q_, 
					    &new_from_old_q_);


    //printf("Built both query and reference trees...\n");

    kernel_.Init(bandwidth_);

    density_estimates_l_.Init(qset_.n_cols());
    density_estimates_u_.Init(qset_.n_cols());
    density_estimates_e_.Init(qset_.n_cols());

    //Set up density estimates to all 0

    density_estimates_l_.SetZero();
    density_estimates_u_.SetAll(rset_.n_cols());
   
    density_estimates_e_.SetZero();
    
  }
}; ///Definition of class DTreeEvaluation ends here.........
#endif
