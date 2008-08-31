#ifndef FAST_LLDE_h
#define FAST_LLDE_H
#define LEAFLEN 5000
#define EPSILON pow(10,-10)
#include "fastlib/fastlib_int.h"

class FastLLDEMulti{
  

 public:

  //Forward declaration of reference tree stat and query tree stat
  
  class RefTreeStat;
  class QueryTreeStat;
  
  //Reference and Query tree
  typedef BinarySpaceTree < DHrectBound <2>, Matrix, RefTreeStat> RefTree;
  typedef BinarySpaceTree <DHrectBound <2>, Matrix, QueryTreeStat> QueryTree;
  
  //Our query tree uses this class

  class QueryTreeStat{
    
    //THIS WILL BE EMPTY
  public:
    
    void Init(){
      
      //Need to set up all stat variables
      
      
    }
    
    // This is the leaf node of the query tree
    void Init(const Matrix &dataset, index_t start, index_t count){
      
      
    }
    
    //This is the internal node of the query tree
    void Init(const Matrix &dataset, index_t start, index_t count, 
	      const QueryTreeStat &left_stat, QueryTreeStat &right_stat){
      
    }
    
  }; //END OF QueryTreeSTAT
  
  //Our reference tree uses this class
  
  class RefTreeStat{

  public:

    // Sum of polynomial expansions summed over all reference points
    // in that node
    Vector sum_of_poly_exp;

    //Sum of poly exp scaled by r_{i,d}^2 in each direction. This will
    //be a matrix
    Matrix sum_of_poly_exp_scaled_by_rsqd;

    //Sum of poly exp scaled by r_{i,d} in each direction. This will
    //be a matrix

    Matrix sum_of_poly_exp_scaled_by_r;

    void Init(){
      
      //Need to set up all stat variables
    }


     
    // This is the leaf node of the reference tree
    void Init(const Matrix &ref_dataset, index_t start, index_t count){

      ////NEED TO BE FILLED////////
    }
    
    //This is the internal node of the reference tree
    
    void Init(const Matrix &ref_dataset, index_t start, index_t count, 
	      const RefTreeStat &left_stat, RefTreeStat &right_stat){
      

      //////NEED TO BE FILLED/////////
      
  }
};//END OF REFTREESTAT
  
 private:

  //A matrix of true_densities

  //  Matrix true_densities_;

  //Vector of densities 

  Vector densities_;

  //Bandwidth used by kernel for calculations

  double bandwidth_;
  
  //Query set
  
  Matrix qset_;
  
  //Reference set
  
  Matrix rset_;

  index_t num_of_prunes_;

 
  
  //The density can be seen as the product of the non-exponential term
  //and the L1 norm of a D-dimensional vector raised to the
  //exponential power. Lets store each of them

  Vector non_exponential_term_; //This will be a N-dimensional vector 
			       //N=number of query points

  Matrix exponential_term_; //This will be a D-dimensional vector for
			   //each query point. Hence we shall store it
			   //as an DxN matrix
  
  //The epanechnikov kernel
  EpanKernel kernel_;
  
  //Number of dimensions
  
  index_t num_of_dimensions_;
  
  //The query tree
  QueryTree *qroot_;
  
  //The reference tree
  RefTree *rroot_;
  
  //The permutations of the reference dataset
  
  ArrayList<index_t> old_from_new_r_;
  ArrayList<index_t> new_from_old_r_;
  
  //The permutation of the query dataset
  
  ArrayList <index_t> old_from_new_q_;
  ArrayList <index_t> new_from_old_q_;
 

  void DoInclusionPruning_(QueryTree *qnode, RefTree *rnode){


    RefTreeStat &rstat=rnode->stat();
    ///////NEEDS TO BE FILLED///////////

    //printf("Will do inclusive pruning...\n");

    //For all the query points in the query node lets calculate the
    //non_exponential_term

    index_t num_of_poly_terms=(index_t)pow(3,num_of_dimensions_);

    //For the non_exponential term I need the sum_of_poly_exp

    char *ternary_str;
    ternary_str=(char*)malloc(num_of_dimensions_);

    for(index_t counter=num_of_poly_terms-1;counter>=0;counter--){

      GetTernaryEquivalent_(counter,ternary_str); //get the ternary
						  //equiuvalent of the
						  //counter


      for(index_t q=qnode->begin();q<qnode->end();q++){ //for each and
							//every query
							//point

	double coeff=1.0;
	index_t length_ternary_str=strlen(ternary_str);
	for(index_t i=strlen(ternary_str)-1;i>=0;i--){
       
	  //Get the ith cooridnate
	  double coord=qset_.get(num_of_dimensions_-length_ternary_str+i,q);
	  coeff*=pow(coord,(ternary_str[i]-48));
	}

	//coeff is the term for the dot product with the reference, 
	//from the query side
	non_exponential_term_[q]+=coeff*
	  rstat.sum_of_poly_exp[num_of_poly_terms-1-counter];

	//To get the exponential term consider each and every direction

	for(index_t d=0;d<num_of_dimensions_;d++){ //Along each direction

	  double coord=qset_.get(d,q); //this is q's coordinate along
				       //the dth dimension
	  double initial_value=
	    exponential_term_.get(d,q);

	  double val_to_add=0;

	  double val1=rstat.sum_of_poly_exp_scaled_by_rsqd.
	    get(num_of_poly_terms-1-counter,d);

	  val_to_add+=val1*coeff; //Added \sum_{i=1}^ poly r_{i,d}^2

	  val_to_add+=
	    rstat.sum_of_poly_exp[num_of_poly_terms-1-counter]*
	    coord*coord*coeff;

	  val_to_add+=-2*coord*
	    rstat.sum_of_poly_exp_scaled_by_r.
	    get(num_of_poly_terms-1-counter,d)*coeff;

	  exponential_term_.set(d,q,initial_value+val_to_add);
	}
      }
    }
    //Lets print the non_exponential_term now
    //printf("non_exponential_term becomes...\n");
    //non_exponential_term_.PrintDebug();
    //printf(" exponential term is ...\n");
    //exponential_term_.PrintDebug();
  }

  void BaseComputations_(QueryTree *qnode, RefTree *rnode){
    
    
    //In this case we need to perform exhaustive calculations by
    //calculating the kernel value between every pair of points
    
    double *diff; //A scratch variable to store vector diff
    diff=(double*)malloc(num_of_dimensions_*sizeof(double));
    
    for(index_t q=qnode->begin();q<qnode->end();q++){

      double *q_col=qset_.GetColumnPtr(q);

      //For each point we need to calculate the exponential part and
      //the non_exponential part. Lets begin with exponential part.

      double total_kernel_value=0;

      for(index_t r=rnode->begin();r<rnode->end();r++){

	double *r_col=rset_.GetColumnPtr(r);
	
	//First GetVector Difference between the query and the
	//reference points

	la::SubOverwrite (num_of_dimensions_,q_col,r_col,diff);
	
	//calculate kernel value using the multiplicative epan kernel

	double kernel_contrib=1.0;
       

	//calculate the contribution of the reference point to the
	//query point. Note that it is a multiplicative kernel, hence
	//we need to itereate along each direction

	for(index_t dim=0;dim<num_of_dimensions_;dim++){

	  double val=kernel_.EvalUnnormOnSq(diff[dim]*diff[dim]);
	  kernel_contrib*=val;
	}

	//add the kernel contribution of the reference point to the
	//total kernel value

	//printf("Kernel contrib of ref_point %d on query=%d is %f\n",
	//     q,r,kernel_contrib);

	total_kernel_value+=kernel_contrib;

	//We can now utilize these estimates to calculate the
	//contribution of this reference point to the exponential part

	for(index_t dim=0;dim<num_of_dimensions_;dim++){ //go along
							 //each
							 //direction

	  double initial_value=exponential_term_.get(dim,q);

	  //I am adding the contribution of the reference point to the
	  //exponential term along each dimension

	  double total_value=initial_value+
	    kernel_contrib*(r_col[dim]-q_col[dim])*(r_col[dim]-q_col[dim]);

	  exponential_term_.set(dim,q,total_value);
	}
      }

      //BY now we have the unnomalized kernel contribution. Since we
      //maintain for each query point an exponential and a
      //non_exponential part hence add this total kernel value to the
      //query points exponential part

      non_exponential_term_[q]+=total_kernel_value;

    }

    //printf("NonExponential term is ...\n");
    //non_exponential_term_.PrintDebug();

    //printf("The exponential term is ....\n");
    //exponential_term_.PrintDebug();
  }


  //This function finds out the max separation between the ref node
  //and query node in a particuar direction

  double GetMinSeparation_(DRange &qnode_range, DRange &rnode_range){

    //Check to see if the intervals are distinct, in which case finds ou the min separation
    if(qnode_range.hi<rnode_range.lo||qnode_range.lo>rnode_range.hi){

      double min_min_dist=fabs(qnode_range.lo-rnode_range.lo);
      double min_max_dist=fabs(qnode_range.lo-rnode_range.hi);
      double max_min_dist=fabs(qnode_range.hi-rnode_range.lo);
      double max_max_dist=fabs(qnode_range.hi-rnode_range.hi);
      return min(min(min_min_dist,min_max_dist),
		 min(max_min_dist,max_max_dist));
      
    }
    else{
      //In case of overlapping intervals send a conservative estimate
      //of the minimum separation
      return 0;
    }
  }

  double GetMaxSeparation_(DRange &qnode_range, DRange &rnode_range){

    double min_min_dist=fabs(qnode_range.lo-rnode_range.lo);
    double min_max_dist=fabs(qnode_range.lo-rnode_range.hi);
    double max_min_dist=fabs(qnode_range.hi-rnode_range.lo);
    double max_max_dist=fabs(qnode_range.hi-rnode_range.hi);
    return max(max(min_min_dist,min_max_dist),
	       max(max_min_dist,max_max_dist));
  }
  void FastMultiDimensionalLocalLikelihood_(QueryTree *qnode, 
					    RefTree *rnode){
    
    //First check if the distance between the reference and the query
    //node along each dimension is within the bandwidth. If not we can
    //prune

    // printf("The points in the query node are start=%d and end=%d\n",
    //qnode->begin(),qnode->end());

    //printf("The points in the ref node are start=%d and end=%d\n",
    //rnode->begin(),rnode->end());
    DRange qnode_range;
    DRange rnode_range;
    for(index_t i=0;i<num_of_dimensions_;i++){
      
      qnode_range=qnode->bound().get(i);
      rnode_range=rnode->bound().get(i);
      
      //get the minimumm separation between these bounding boxes along
      //each direction
      double min_dist=GetMinSeparation_(qnode_range,rnode_range);
      //printf("Along dir %d min_dist=%f\n",i,min_dist);
      if(min_dist>bandwidth_-EPSILON){
	
	//prune 
	
	//printf("Will do exclusion pruning...\n");
	num_of_prunes_++;
	return;
      }
    }

    index_t flag=1; //A flag to indicate if the entire query node is
		    //within a bandwidth of the reference node

    //Check if the max distance along each direction is within the
    //bandwidth

    for(index_t i=0;i<num_of_dimensions_;i++){

      qnode_range=qnode->bound().get(i);
      rnode_range=rnode->bound().get(i);
      
      //get the maximumm separation between these bounding boxes
      double max_dist=GetMaxSeparation_(qnode_range,rnode_range);

      //printf("Along dir %d max_dist=%f\n",i,max_dist);

      if(max_dist>bandwidth_-EPSILON){
	
	flag=0;
	//printf("max_distanace is %f and dimension=%d\n",max_dist,i);
	break;
      }
      else{
	
	//dont do anything
      }
    }


    //Now if the flag is 0 then we need to recurse, else we can do
    //exclusive pruning
    
    if(flag==1){
      
      DoInclusionPruning_(qnode,rnode);
      return;
    }
    
    else{
      
      //This means that the entire query box is not within the
      //bandwidth of the kernel. Hence lets continue recursing
      
      //printf("Continue recursing....\n");
      
      
      if(!qnode->is_leaf() && !rnode->is_leaf()){
	
	//4-way recursion	  
	FastMultiDimensionalLocalLikelihood_(qnode->left(),rnode->left());
	FastMultiDimensionalLocalLikelihood_(qnode->left(),rnode->right());
	
	FastMultiDimensionalLocalLikelihood_(qnode->right(),rnode->left());
	FastMultiDimensionalLocalLikelihood_(qnode->right(),rnode->right());
      }
      else{
	if(!qnode->is_leaf()&& rnode->is_leaf()){
	  
	  FastMultiDimensionalLocalLikelihood_(qnode->left(),rnode);
	  FastMultiDimensionalLocalLikelihood_(qnode->right(),rnode);
	}
	else{
	  
	  if(qnode->is_leaf()&& !rnode->is_leaf()){
	    
	    FastMultiDimensionalLocalLikelihood_(qnode,rnode->left());
	    FastMultiDimensionalLocalLikelihood_(qnode,rnode->right());
	  }
	  else{
	    //This means both the nodes are leaf nodes
	    
	    //printf("Both are leaf nodes...hence do base case\n");
	    BaseComputations_(qnode,rnode);
	  }
	}
      }
    }
  }
  
  //A small helper function written in order to eaily handle the
  //monstrous local likelihood expression
  void get_the_norm_of_exponential_term_(Vector &exponential_bias){
     
    for(index_t q=0;q<qset_.n_cols();q++){
      
      double *q_ptr=exponential_term_.GetColumnPtr(q);
      double distance=la::LengthEuclidean(num_of_dimensions_,q_ptr);
      
      //divide this expression with the non_exponetial_value. However
      //for numerical stability make sure that the
      //non_exponential_term is not zero
      
      if(fabs(non_exponential_term_[q])>EPSILON){
	
	double temp=
	  -1*distance*distance/(2*pow(non_exponential_term_[q]*bandwidth_,2));
	
	exponential_bias[q]=pow(math::E,temp);
      }
      else{
	
	exponential_bias[q]=0;
      }
    }
  }

  //This function combines the exponential and non-exponential terms
  //to give the local likelihood density at a point

    void PostProcess_(){

    Vector exponential_bias;
    exponential_bias.Init(qset_.n_cols());


    //For numerical statbility calculate the norm of exponential term
    //only if the non_exponential_term is non-zero

    get_the_norm_of_exponential_term_(exponential_bias);

    double norm_const=kernel_.CalcNormConstant(1);

    norm_const=pow(norm_const,num_of_dimensions_);

    printf("Normalization constant is %f\n",norm_const*rset_.n_cols());

    for(index_t q=0;q<qset_.n_cols();q++){ //for each query point

      densities_[q]=non_exponential_term_[q]*exponential_bias[q];

      //Now normalize the density estimates
      densities_[q]/=(norm_const*rset_.n_cols());
    }

    //printf("Denisities are...\n");
    //densities_.PrintDebug();

    printf("NUmber of Prunes are %d\n",num_of_prunes_);
  }

   


  index_t IsPowerOf3(index_t counter){
    
    index_t ceil_val=(index_t)ceil((log(counter))/log(3.0));
    index_t floor_val=(index_t)floor((log(counter))/log(3.0));
    if(ceil_val==floor_val){
      return 1;
    }
    else return 0;
  }

  void GetTernaryEquivalent_(index_t counter, char *ternary_str){
    //printf("Counter is %d\n",counter);
    
    if(counter<3){
      ternary_str[0]=48+counter;
      ternary_str[1]=0;
      //printf("the ternary str is %s\n",ternary_str);
      return;
    }
    
    if(counter==3){
      ternary_str[0]=49;
      ternary_str[1]=48;
      //printf("THe ternary str is %s\n",ternary_str);
      return;
    }
    index_t length;
    if(!IsPowerOf3(counter)){
       length=(index_t)ceil((log(counter))/log(3.0));
    }
    else{
      length=(index_t)ceil((log(counter))/log(3.0))+1;
    }
    ternary_str[length]=0;
    length--;
    while(counter>=3){
      ternary_str[length]=48+(counter%3);
      counter/=3;
      length--;
    }
    
    ternary_str[length]=48+counter;
  }
  
  
  
  void FillStatistics_(index_t total_terms_in_poly_exp,
		       index_t num_of_dimensions,index_t start,
		       index_t end,RefTreeStat &rstat){
    
    
    //For each point i need a polynomial expansion. Lets get them first
    Matrix poly_exp;
    poly_exp.Init(total_terms_in_poly_exp,end-start);
    
    //Lets maintain a ternary counter. This will help us know
    //which term of the polynomail we are considering. The counter
    //begins at all 2's.
    
   index_t counter;
   counter=total_terms_in_poly_exp-1;
   //reset the counter 
   counter=total_terms_in_poly_exp-1;
   char *ternary_str;
   ternary_str=(char*)malloc(100*sizeof(char));
   while(counter>=0){ //for each counter value

     //In order to know which term in the expansion we are using we
     //need to know the ternary( base-3) equivalent of the counter.
     
     GetTernaryEquivalent_(counter,ternary_str);
     index_t length_ternary_str=strlen(ternary_str);

     // printf("The counter is %d\n",counter);
     //printf("ternary string is %s\n",ternary_str);

     for(index_t ref_num=start;ref_num<end;ref_num++){ //For each point

      
       double coeff=1;
	  
       //Get the coefficient
	  
       //Start from the end of the string
     
     
       for(index_t i=strlen(ternary_str)-1;i>=0;i--){
       
	 //Get the ith cooridnate
	 double coord=rset_.get(num_of_dimensions-length_ternary_str+i,
				ref_num);
	   
	 if(ternary_str[i]==50){
	      
	   coeff*=-1/(bandwidth_*bandwidth_);
	 }
	 else if(ternary_str[i]==49){
	      
	   coeff*=2*coord/(bandwidth_*bandwidth_);
	 }
	 else{
	      
	   coeff*=(1-pow(coord/bandwidth_,2));
	 }
       }
	  
       for(index_t i=strlen(ternary_str);i<num_of_dimensions;i++){
	    
	 index_t temp=num_of_dimensions-i-1;
	 double coord=rset_.get(temp,ref_num);
      
	 coeff*=(1-pow(coord/bandwidth_,2));
       } 

       //We now have the coeff of this particular polynomial expansion   

       poly_exp.set(total_terms_in_poly_exp - counter-1,ref_num-start,coeff);
     }
     counter--;
   }

   
   //Having got the polynomial expansions of all the reference
   //points in the node get the required statistics of the node by
   //using these expansions

   //Lets use the polynomial expansions to form the statistics

   //Scratch variables for calculations  

   Vector temp1;
   temp1.Init(total_terms_in_poly_exp);
   Vector temp2;
   temp2.Init(total_terms_in_poly_exp);
   temp2.Init(total_terms_in_poly_exp);

   for(index_t col=0;col<poly_exp.n_cols();col++){


     //Lets first get the sum_of_poly_exp by adding all columns
     //We can do this by extracting the columns and adding them up
     

     //extract the column first
     double *column=poly_exp.GetColumnPtr(col);
     Vector vec;
     vec.Alias(column,total_terms_in_poly_exp);
     
     //Add this column to the sum_of_poly_exp
     la::AddTo(vec,&rstat.sum_of_poly_exp);


     //Now lets get the sum_of_poly_exp_scaled_by_rsqd. This has to be
     //done along every dimension. So lets consider dimensions
     //one-by-one

   
     for(index_t d=0;d<num_of_dimensions;d++){
       
       double coordinate=rset_.get(d,start+col); //dth coordinate of the point
       double sqd_coordinate=coordinate*coordinate; // The square of
						    // the dth
						    // coordinate
      
       
       la::ScaleOverwrite(sqd_coordinate,vec,&temp1); //get the scaled polynomial
                                                       //expansion along that
                                                       //direction/dimension


       //printf("sqd coordinate is %f\n",sqd_coordinate);
       //printf("temp1 is ..\n");
       //temp1.PrintDebug();
       la::ScaleOverwrite(coordinate,vec,&temp2); //get the scaled polynomial
                                                       //expansion along that
                                                       //direction/dimension


       //Add this scaled version,to the exisitng matrix
       
       //first get the column from the matrix
       //sum_of_poly_exp_scaled_by_rsqd along this dimension

       double *vec1=rstat.sum_of_poly_exp_scaled_by_rsqd.GetColumnPtr(d);
       //Add temp1 to vec1
       la::AddTo(total_terms_in_poly_exp,temp1.ptr(),vec1);


       double *vec2=rstat.sum_of_poly_exp_scaled_by_r.GetColumnPtr(d);
       //Add temp2 to vec2
       la::AddTo(total_terms_in_poly_exp,temp2.ptr(),vec2);

       //Note since we are getting a pointer to the matrix column
       //hence we dont need to write back to the matrix

     }
   }
   //   printf("The sum of vectors from start=%d to end=%d\n",start,end);

   /* printf("Sum of poly exp is ..\n");
   rstat.sum_of_poly_exp.PrintDebug();

   printf("Sum of scaled by r poly exp is ...\n");
   rstat.sum_of_poly_exp_scaled_by_r.PrintDebug();


   printf("Sum of scaled by rsqd poly exp is ...\n");
   rsat.sum_of_poly_exp_scaled_by_rsqd.PrintDebug(); */
  }

 void PreProcess_(RefTree *rnode){
      
   index_t start=rnode->begin();
   index_t end=rnode->end();
   index_t num_of_dimensions=rset_.n_rows();
      
      
   //The statistics of the reference tree
   RefTreeStat &rstat=rnode->stat();
      
   //This will be the total number of terms in the polynomial
   //expansion of the reference point. This is just 3^D
      
   index_t total_terms_in_poly_exp;
      
   total_terms_in_poly_exp=(index_t)pow(3,num_of_dimensions);
      
   // First lets initialize all the quantities and then fill it up
      
   rstat.sum_of_poly_exp.Init(total_terms_in_poly_exp); //Vector
      
   rstat.sum_of_poly_exp_scaled_by_rsqd.
     Init(total_terms_in_poly_exp,num_of_dimensions); //matrix(3^DXD)
      
   rstat.sum_of_poly_exp_scaled_by_r.
     Init(total_terms_in_poly_exp,num_of_dimensions);
      
   //Initialize all these quantities to 0
      
   rstat.sum_of_poly_exp.SetZero();
   rstat.sum_of_poly_exp_scaled_by_rsqd.SetZero();
   rstat.sum_of_poly_exp_scaled_by_r.SetZero();
      
      
   if(rnode->is_leaf()){
	
     //Having initialized lets caclulate all these quantities. To do
     //that I need the polynomial expansion of each point in the
     //reference node.
     FillStatistics_(total_terms_in_poly_exp,num_of_dimensions,start,end,rstat);
   }
   else{
     //PreProcess the left and righ child and then stich it up
	
     PreProcess_(rnode->left());
     PreProcess_(rnode->right());
	
     ////NOW STICH IT UP

     //STICH UP the vector sum_of_poly_exp first
     la::AddOverwrite(rnode->left()->stat().sum_of_poly_exp,
		      rnode->right()->stat().sum_of_poly_exp,
		      &rstat.sum_of_poly_exp);

     //Stich up the matrix sum_of_poly_exp_scaled_by_rsqd

     la::AddOverwrite(rnode->left()->stat().sum_of_poly_exp_scaled_by_rsqd,
		      rnode->right()->stat().sum_of_poly_exp_scaled_by_rsqd,
		      &rstat.sum_of_poly_exp_scaled_by_rsqd);

     //Stich up the matrix sum_of_poly_exp_scaled_by_r
     
     la::AddOverwrite(rnode->left()->stat().sum_of_poly_exp_scaled_by_r,
		      rnode->right()->stat().sum_of_poly_exp_scaled_by_r,
		      &rstat.sum_of_poly_exp_scaled_by_r);


     /*printf("Sum of poly exp is ..\n");
     rstat.sum_of_poly_exp.PrintDebug();

     printf("Sum of scaled by r poly exp is ...\n");
     rstat.sum_of_poly_exp_scaled_by_r.PrintDebug();
   
   
     printf("Sum of scaled by rsqd poly exp is ...\n");
     rstat.sum_of_poly_exp_scaled_by_rsqd.PrintDebug();*/

   }
 }

 void PrintToFile_(){

   FILE *fp;
   fp=fopen("fast_local_likelihood_multi_results.txt","w+");

   fprintf(fp,"normalized local likelihood estimatesd...\n");
   for(index_t i=0;i<densities_.length();i++){
     
     fprintf(fp,"density[%d]=%f\n",
	     i,densities_[new_from_old_q_[i]]);
   }
   fclose(fp);
 }

 void GetVectorDifference_(double *vec1, double *vec2, index_t length, 
			   Vector &diff){
   for(index_t l=0;l<length;l++){
     diff[l]=vec1[l]-vec2[l];
   }

 }

 void PrintToFileNaive_(){

   FILE *fp;
   fp=fopen("naive_results_multi.txt","w+");

   Vector temp;
   temp.Init(qset_.n_cols());
   for(index_t i=0;i<qset_.n_cols();i++){
     
     temp[old_from_new_q_[i]]=densities_[i];
   }
   
   for(index_t i=0;i<densities_.length();i++){
      
      fprintf(fp,"naive_density[%d]=%f\n",i,temp[i]);
    }
   fclose(fp);
 }


 
 void NaiveMultiDimenisonalLocalLikelihood_(){
   
   printf("will do naive calculations...\n");
   exponential_term_.SetZero();
   non_exponential_term_.SetZero();
   densities_.SetZero();

   //exponential_term_.Init(qset_.n_rows(),qset_.n_cols());
   
   //Initialize the non-expoential_term. The non-exponential term is
   //a scalar for each point
   // non_exponential_term_.Init(qset_.n_cols());
   
   double *diff; //A scratch variable to store vector diff
   diff=(double*)malloc(num_of_dimensions_*sizeof(double));
   
   for(index_t q=0;q<qset_.n_cols();q++){
     
     double *q_col=qset_.GetColumnPtr(q);
     
     //For each point we need to calculate the exponential part and
     //the non_exponential part. Lets begin with exponential part.
     
     double total_kernel_value=0;
     
     for(index_t r=0;r<rset_.n_cols();r++){
       
       double *r_col=rset_.GetColumnPtr(r);
       
       //First GetVector Difference between the query and the
       //reference points
       
       la::SubOverwrite (num_of_dimensions_,q_col,r_col,diff);
       
       //calculate kernel value using the multiplicative epan kernel
       
       double kernel_contrib=1.0;
       
       
       //calculate the contribution of the reference point to the
       //query point. Note that it is a multiplicative kernel, hence
       //we need to itereate along each direction
       
       for(index_t dim=0;dim<num_of_dimensions_;dim++){
	 
	 double val=kernel_.EvalUnnormOnSq(diff[dim]*diff[dim]);
	 kernel_contrib*=val;
       }
       
       //add the kernel contribution of the reference point to the
       //total kernel value
          
       total_kernel_value+=kernel_contrib;
       
       //We can now utilize these estimates to calculate the
       //contribution of this reference point to the exponential part
       
       for(index_t dim=0;dim<num_of_dimensions_;dim++){ //go along
	 //each
	 //direction
	 
	 double initial_value=exponential_term_.get(dim,q);
	 
	 //I am adding the contribution of the reference point to the
	 //exponential term along each dimension
	 
	 double total_value=initial_value+
	   kernel_contrib*(r_col[dim]-q_col[dim])*(r_col[dim]-q_col[dim]);
	 
	 exponential_term_.set(dim,q,total_value);
       }
     }
     
     //BY now we have the unnomalized kernel contribution. Since we
     //maintain for each query point an exponential and a
     //non_exponential part hence add this total kernel value to the
     //query points exponential part
     
     non_exponential_term_[q]+=total_kernel_value; 
   }

   //having got the exponential and non_exponential term lets get the
   //density

   //postprocess
   PostProcess_();
   
 }

    
 ///////////////////////GETTERS///////////////
 public:
    
 void get_permuted_density_estimates(Vector &density_estimates){

   //Note the density estimates we have are permuted. The user shall
   //depermute it by using the permutation arrays
   density_estimates.Alias(densities_);
 }

 void get_depermuted_density_estimates(Vector &density_estimates){

   //This will return the depermuted density estimates. To do that we
   //first need to de-permute the densities to the original order

   Vector temp;
   temp.Init(qset_.n_cols());

   for(index_t q=0;q<qset_.n_cols();q++){
     
     temp[q]=densities_[new_from_old_q_[q]];
   }
   densities_.CopyValues(temp);
   density_estimates.CopyValues(densities_);
 }


 void get_old_from_new_q(ArrayList<index_t> &old_from_new_q){
   
   for(index_t q=0;q<qset_.n_cols();q++){
     
     old_from_new_q[q]=old_from_new_q_[q];
   }
   
 }
  
 void  Destruct(){

   qset_.Destruct();
   rset_.Destruct();

   old_from_new_q_.Destruct();
   new_from_old_q_.Destruct();

   old_from_new_r_.Destruct();
   new_from_old_r_.Destruct();

   densities_.Destruct();
   exponential_term_.Destruct();
   non_exponential_term_.Destruct();
   delete(qroot_);
   delete(rroot_);

   //Lets initialize them to very small memory inorder to avoid
   //segfault

   qset_.Init(1,1);
   rset_.Init(1,1);
   densities_.Init(1);
   exponential_term_.Init(1,1);
   non_exponential_term_.Init(1);
   old_from_new_q_.Init(1);
   new_from_old_q_.Init(1);

   old_from_new_r_.Init(1);
   new_from_old_r_.Init(1);

 }
  
 void Compute(){
   
   printf("Preprocess the tree..\n");
   PreProcess_(rroot_);
  
   fx_timer_start(NULL,"compute_fast");
   FastMultiDimensionalLocalLikelihood_(qroot_,rroot_); 
   //PrintExpAndNonExpTerms_();
  
   printf("will postprocess now...\n");
   PostProcess_();
   fx_timer_stop(NULL,"compute_fast");
   printf("will print to file..\n");
   PrintToFile_();
    
   //fx_timer_start(NULL,"naive");
   //NaiveMultiDimenisonalLocalLikelihood_();
   //fx_timer_stop(NULL,"naive");    
   //printf("Will print naive calc to file..\n");
   //PrintToFileNaive_();

   
 }
  
 void Init(Matrix &query, Matrix &references, double bandwidth){
    

   printf("Doing fast local likelihood...\n");
   //Copy the query and reference sets  
   qset_.Alias(query);
   rset_.Alias(references);
   printf("Aliased the dataset...\n");
    
   //Initialize the kernel with the bandwidth
   kernel_.Init(bandwidth);
    
   //Initialize the vector of densities
   densities_.Init(qset_.n_cols());
    
   num_of_dimensions_=rset_.n_rows();

   printf("Other elem quants set up...\n");

   //BUILD THE QUERY TREE

   fx_timer_start(NULL,"create_trees");
   qroot_=tree::MakeKdTreeMidpoint <QueryTree> 
     (qset_,LEAFLEN,&old_from_new_q_,&new_from_old_q_);
   printf("query tree made..\n");
    
   //BUILD THE REFERENCE TREE
   rroot_=tree::MakeKdTreeMidpoint <RefTree> 
     (rset_,LEAFLEN,&old_from_new_r_,
      &new_from_old_r_);
   fx_timer_stop(NULL,"create_trees");


   printf("ref tree made...\n");
   //Initialize the matrix exponential_term. Exponential term is a
   //d-dimensional vector for each reference point. Hence we shall
   //store it as a column major matrix

   exponential_term_.Init(qset_.n_rows(),qset_.n_cols());

   //Initialize the non-expoential_term. The non-exponential term is
   //a scalar for each point
   non_exponential_term_.Init(qset_.n_cols());

   exponential_term_.SetZero();
   non_exponential_term_.SetZero();

   //The origianl multi-dimensional local likelihood takes in a
   //vector of bandwidths, one for each dimension. Lets take in just
   //a single scalar which is the bandwidth in all directions. This
   //is fine to do because even if the data is kind of more wide in
   //one dimension and less wider in another we can pre-whiten the
   //data by condensing it to a unitt cube or making it standard
   //normal (fukunaga's pre-whitening of the data)

   bandwidth_=bandwidth;
   kernel_.Init(bandwidth);



   /*printf("THe reference tree is.....\n");
   rroot_->Print();


   printf("The query tree is ...\n");
   qroot_->Print();*/

   num_of_prunes_=0;

   printf("initializes....\n");
   
 }
};//END OF CLASS FASTLLDE

#endif
