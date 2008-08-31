#ifndef FAST_LLDE_h
#define FAST_LLDE_H
#define LEAFLEN 5000
#define EPSILON pow(10,-10)
#include "fastlib/fastlib_int.h"

class FastLLDE{
  

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

    //This is simply \sum_{i=1}^N X_{i,k}    
    Vector sum_of_ref_coordinates;

    //This is simply \sum_{i=1}^N X_{i,k} (x_i.x_i)
    Vector sum_of_scaled_vector_lengths;

    //This is simply \sum_{i=1}^N x_{i,k} X_i
    Matrix sum_of_scaled_ref_points;

    //This is simply \sum_{i=1}^N (X_i.X_i)
    double sum_of_sqd_lengths;

    //This is just \sum X_i
    Vector sum_of_ref_points;

    void Init(){
      
      //Need to set up all stat variables
    }
    
    // This is the leaf node of the reference tree
    void Init(const Matrix &ref_dataset, index_t start, index_t count){

      //printf("Came to the leaf node.......\n");

      //printf("Lets initialize the variables in the leaf node....\n");

      index_t num_of_dimensions;
      index_t end=start+count;
      num_of_dimensions=ref_dataset.n_rows();
      
      // First lets initialize all the quantities and then fill it up

      //Initialize all the vectors first..............
      sum_of_ref_coordinates.Init(num_of_dimensions);
      sum_of_scaled_vector_lengths.Init(num_of_dimensions);
      sum_of_scaled_ref_points.Init(num_of_dimensions,num_of_dimensions);
      sum_of_ref_points.Init(num_of_dimensions);

      //Initialize all these quantities to 0

      sum_of_ref_coordinates.SetZero();
      sum_of_scaled_vector_lengths.SetZero();
      sum_of_scaled_ref_points.SetZero();
      sum_of_ref_points.SetZero();
    

      //Initialize the double quantity to 0
      sum_of_sqd_lengths=0;


      //Having initialized lets caclulate all these quantities
      //For matters of efficiency lets access the data matrix column wise
      Vector col_vec;
      col_vec.Init(num_of_dimensions);

      for(index_t col=start;col<end;col++){
	
	//first get a handle to the column
	
	double *ptr=(double*)ref_dataset.GetColumnPtr(col);

	col_vec.CopyValues(ptr);

	//Sum of ref points
	
	la::AddTo(col_vec,&sum_of_ref_points);
	
	//The sqd length of this column vector

	double length=la::LengthEuclidean(col_vec);
	double sqd_length=length*length;
	
	la::AddTo(col_vec,&sum_of_ref_coordinates);
	
	//Scale the col_vec by its sqdlength
	
	la::Scale(sqd_length,&col_vec);
	
	la::AddTo(col_vec,&sum_of_scaled_vector_lengths);


	
	//Sum of sqd_lengths. This is a scalar quantity
	sum_of_sqd_lengths+=sqd_length;
      }
      
      //In the above calculations we havent calculated the matrix
      //sum_of_scaled_reference_points. We do that below.Note this has
      //been arranged column-wise
      
      Vector temp;
      temp.Init(num_of_dimensions);
      temp.SetZero();
      for(index_t k=0;k<num_of_dimensions;k++){
	
	Vector total_sum_vector;
	total_sum_vector.Init(num_of_dimensions);
	total_sum_vector.SetZero();
	
	//Get the column ptr and scale it by the kth coordinate

	for(index_t col1=start;col1<end;col1++){
	  
	  double *col_ptr=(double*)ref_dataset.GetColumnPtr(col1);
	  temp.CopyValues(col_ptr);
	  la::Scale(ref_dataset.get(k,col1),&temp);
	  la::AddTo(temp,&total_sum_vector);
	}

	for(index_t row=0;row<num_of_dimensions;row++){
	  
	  sum_of_scaled_ref_points.set(row,k,total_sum_vector[row]);	  
	}
      }
     
    }
    
    //This is the internal node of the reference tree
    
    void Init(const Matrix &ref_dataset, index_t start, index_t count, 
	      const RefTreeStat &left_stat, RefTreeStat &right_stat){
      

      // printf("Came to the internal node..........\n");
      
      index_t num_of_dimensions=ref_dataset.n_rows();
      sum_of_ref_coordinates.Init(num_of_dimensions);
      sum_of_scaled_vector_lengths.Init(num_of_dimensions);
      sum_of_scaled_ref_points.Init(num_of_dimensions,num_of_dimensions);
      sum_of_ref_points.Init(num_of_dimensions);
      

      //For the internal node we shall make use of the calculations
      //already performed in the leaf node


      //First set up the SUM_OF_REF_COORDINATES
      la::AddOverwrite(left_stat.sum_of_ref_coordinates,
		       right_stat.sum_of_ref_coordinates,
		       &(this->sum_of_ref_coordinates));

      //Now set up the SUM_OF_SCALED_VECTOR_LENGTHS

      la::AddOverwrite(left_stat.sum_of_scaled_vector_lengths,
	      right_stat.sum_of_scaled_vector_lengths,
	      &(this->sum_of_scaled_vector_lengths));

      //SET UP SUM_OF_SCALED_REFERENCE_POINTS. REMEMBER THIS IS A MATRIX
      la::AddOverwrite(left_stat.sum_of_scaled_ref_points,
	      right_stat.sum_of_scaled_ref_points,
	      &(this->sum_of_scaled_ref_points));

      //SET UP SUM_OF_SQD_LENGTHS

      this->sum_of_sqd_lengths=
	left_stat.sum_of_sqd_lengths+
	right_stat.sum_of_sqd_lengths;

      //SET UP SUM_OF_REFERENCE_POINTS

      la::AddOverwrite(left_stat.sum_of_ref_points,
		       right_stat.sum_of_ref_points,
		       &sum_of_ref_points);
	
      
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
  
  //The density can be seen as the product of the non-exponential term
  //and the L1 norm of a D-dimensional vector raised to the
  //exponential power. Lets store each of them

  Vector non_exponential_term; //This will be a N-dimensional vector 
			       //N=number of query points

  Matrix exponential_term; //This will be a D-dimensional vector for
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


  void BaseComputations_(QueryTree *qnode, RefTree *rnode){
  
    //Do the base computations

    for(index_t q=qnode->begin();q<qnode->end();q++){

      double *q_ptr=qset_.GetColumnPtr(q);

      for(index_t r=rnode->begin();r<rnode->end();r++){

	//Lets first calculate the non_exponential term
	double *r_ptr=rset_.GetColumnPtr(r);
	double sqd_distance=
	  la::DistanceSqEuclidean(qset_.n_rows(),q_ptr,r_ptr);

	double kernel_contrib=kernel_.EvalUnnormOnSq(sqd_distance);
	non_exponential_term[q]+=kernel_contrib;

	//Now lets get the exponential_term in each direction
	for(index_t k=0;k<num_of_dimensions_;k++){

	  double exponential_term_initial=exponential_term.get(k,q);

	  double exponential_term_new=exponential_term_initial+
	    (kernel_contrib*pow((rset_.get(k,r)-qset_.get(k,q)),2));

	  exponential_term.set(k,q,exponential_term_new);
	}
      }
    }
  } // END OF BASE COMPUTATIONS..................


  void CalculateKernelContributions_(QueryTree *qnode, RefTree *rnode){

    RefTreeStat *stat=&(rnode->stat());
    index_t num_points_rnode=rnode->end()-rnode->begin();
    double bandwidth_sqd=bandwidth_*bandwidth_;
    for(index_t q=qnode->begin();q<qnode->end();q++){

      double *q_ptr=qset_.GetColumnPtr(q);
      double sqd_length_q_vec=pow(la::LengthEuclidean(qset_.n_rows(),q_ptr),2);

      non_exponential_term[q]+=
	num_points_rnode*(1-sqd_length_q_vec/bandwidth_sqd);

      double dot_product_with_sum_of_ref_points=
	la::Dot(num_of_dimensions_,stat->sum_of_ref_points.ptr(),q_ptr);

      non_exponential_term[q]-=
	(stat->sum_of_sqd_lengths-2*dot_product_with_sum_of_ref_points)/
	bandwidth_sqd;

      //With this the non_exponential_term has been updated. 

      //Lets now update the exponential_term. we do this along each direction

      for(index_t k=0;k<num_of_dimensions_;k++){ //Along each direction
       
	double k_coordinate_of_q_point=qset_.get(k,q);
	double initial_value=exponential_term.get(k,q);

	initial_value+=stat->sum_of_ref_coordinates[k] - 
	  num_points_rnode*k_coordinate_of_q_point;

	initial_value-=((stat->sum_of_ref_coordinates[k]-
			 num_points_rnode*k_coordinate_of_q_point)*
			sqd_length_q_vec)/(bandwidth_sqd);

	initial_value-=(stat->sum_of_scaled_vector_lengths[k]-
			k_coordinate_of_q_point*stat->sum_of_sqd_lengths)/
	  (bandwidth_sqd);

	double *vec=stat->sum_of_scaled_ref_points.GetColumnPtr(k);
	Vector vec1;
	vec1.Alias(vec,num_of_dimensions_);
	Vector dummy;

	//This contains x_k \sum X_i
	la::ScaleInit(k_coordinate_of_q_point,stat->sum_of_ref_points,&dummy); 


	Vector diff;
	la::SubInit(dummy,vec1,&diff);

	double dot_product_with_query_point;
	Vector q_vec;
	q_vec.Alias(q_ptr,num_of_dimensions_);
	dot_product_with_query_point=la::Dot(q_vec,diff);
	initial_value+=2*dot_product_with_query_point/(bandwidth_sqd);

	//Insert this final value back into exponential_term
	exponential_term.set(k,q,initial_value);
      }
    }
  }

  void FastLocalLikelihood_(QueryTree *qnode, RefTree *rnode){
    
    //First check if the minimum distance between the ref node and
    //query node is greater than the bandwidth. If it is so then we
    //can prune


    double sqd_min_dist=
      qnode->bound().MinDistanceSq(rnode->bound());

    if(sqd_min_dist > bandwidth_* bandwidth_){

      //We can prune
     
      return;
    }
    else{
      
      double sqd_max_distance= 
	qnode->bound().MaxDistanceSq(rnode->bound());
      
      if(sqd_max_distance < bandwidth_ * bandwidth_){
	//This means the entire query node falls in the bandwidth range
	CalculateKernelContributions_(qnode,rnode);
      }

      else{

	//This means that the entire query box is not within the
	//bandwidth of the kernel. Hence lets continue recursing
	

	if(!qnode->is_leaf() && !rnode->is_leaf()){

	  //4-way recursion	  
	  FastLocalLikelihood_(qnode->left(),rnode->left());
	  FastLocalLikelihood_(qnode->left(),rnode->right());
	  
	  FastLocalLikelihood_(qnode->right(),rnode->left());
	  FastLocalLikelihood_(qnode->right(),rnode->right());
	}
	else{
	  if(!qnode->is_leaf()&& rnode->is_leaf()){

	    FastLocalLikelihood_(qnode->left(),rnode);
	    FastLocalLikelihood_(qnode->right(),rnode);
	  }
	  else{

	    if(qnode->is_leaf()&& !rnode->is_leaf()){

	      FastLocalLikelihood_(qnode,rnode->left());
	      FastLocalLikelihood_(qnode,rnode->right());
	    }
	    else{
	      //This means both the nodes are leaf nodes
	      
	      BaseComputations_(qnode,rnode);

	    }
	  }
	}
      }
    }
  }


  //A small helper function written in order to eaily handle the
  //monstrous local likelihood expression
  void get_the_norm_of_exponential_term_(Vector &exponential_bias){

    for(index_t q=0;q<qset_.n_cols();q++){

      double *q_ptr=exponential_term.GetColumnPtr(q);
      double distance=la::LengthEuclidean(num_of_dimensions_,q_ptr);
     
      //divide this expression with the non_exponetial_value. However
      //for numerical stability make sure that the
      //non_exponential_term is not zero

      if(fabs(non_exponential_term[q])>EPSILON){

	double temp=
	  -1*distance*distance/(2*pow(non_exponential_term[q]*bandwidth_,2));

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

    for(index_t q=0;q<qset_.n_cols();q++){ //for each query point

      densities_[q]=non_exponential_term[q]*exponential_bias[q];

      //Now normalize the density estimates
      densities_[q]/=(kernel_.CalcNormConstant(num_of_dimensions_)*
		      rset_.n_cols());
    }
  }

  void NaiveLocalLikelihood_(){

    Vector exponential_term;
    exponential_term.Init(num_of_dimensions_);
    exponential_term.SetZero();


    for(index_t q=0;q<qset_.n_cols();q++){
      
      double *q_ptr=qset_.GetColumnPtr(q);
      exponential_term.SetZero();
      
      double non_exponential_term=0;
      for(index_t r=0;r<rset_.n_cols();r++){
	
	//Lets first calculate the non_exponential term
	double *r_ptr=rset_.GetColumnPtr(r);
	double sqd_distance=la::DistanceSqEuclidean
	  (qset_.n_rows(),q_ptr,r_ptr);
	
	double kernel_contrib=kernel_.EvalUnnormOnSq(sqd_distance);
	non_exponential_term+=kernel_contrib;
	
	//Now lets get the exponential_term in each direction
	for(index_t k=0;k<num_of_dimensions_;k++){
	  
	  exponential_term[k]+=
	    (kernel_contrib*(pow(rset_.get(k,r)-qset_.get(k,q),2)));
	}
      }
   
      double distance=la::LengthEuclidean(exponential_term);
      
      //divide this expression with the non_exponetial_value;
      double temp=
	-1*distance*distance/(2*pow(non_exponential_term*bandwidth_,2));
      
      double exponential_bias=pow(math::E,temp);
     
      double density=non_exponential_term*exponential_bias;
      //Now normalize the density estimates
      density/=
	(kernel_.CalcNormConstant(num_of_dimensions_)*rset_.n_cols());
    }
  }


  void PrintToAFile_(){
    

    /*Vector temp;
    temp.Init(qset_.n_cols());
    
    for(index_t q=0;q<qset_.n_cols();q++){
      
      temp[old_from_new_q_[q]]=densities_[q];      
      }*/

    FILE *fp;
    fp=fopen("local_likelihod_densities.txt","w+");
    fprintf(fp,"Printing results...\n");
    for(index_t q=0;q<qset_.n_cols();q++){      

      fprintf(fp,"density[%d]=%f, non_exp_term[%d]=%f,exp_term[%d]=%f\n",
	      q,densities_[q],q,non_exponential_term[q],
	      q,exponential_term.get(0,q));
    }
  }
///////////////////////GETTERS///////////////
 public:
  
  void get_density_estimates(Vector &density_estimates){
    
     density_estimates.Alias(densities_);

    ////////////////////NEED TO BE FILLED////////////////////
  }
  
  
  void Compute(){

    fx_timer_start(NULL,"compute_fast");
    FastLocalLikelihood_(qroot_,rroot_); 
    PostProcess_();
    fx_timer_stop(NULL,"compute_fast");
    
    //fx_timer_start(NULL,"naive");
    //NaiveLocalLikelihood_();
    //fx_timer_stop(NULL,"naive");    

    PrintToAFile_();
  }
  
  void Init(Matrix &query, Matrix &references, double bandwidth){
    
    //Copy the query and reference sets  
    qset_.Alias(query);
    rset_.Alias(references);
    
    //Initialize the kernel with the bandwidth
    kernel_.Init(bandwidth);
    
    //Initialize the vector of densities
    densities_.Init(qset_.n_cols());
    
    num_of_dimensions_=rset_.n_rows();

    //BUILD THE QUERY TREE

    fx_timer_start(NULL,"create_trees");
    qroot_=tree::MakeKdTreeMidpoint <QueryTree> (qset_,LEAFLEN,&old_from_new_q_,
						 &new_from_old_q_);
    //BUILD THE REFERENCE TREE
    rroot_=tree::MakeKdTreeMidpoint <RefTree> (rset_,LEAFLEN,&old_from_new_r_,
					       &new_from_old_r_);
    fx_timer_stop(NULL,"create_trees");


    //Initialize the matrix exponential_term
    exponential_term.Init(qset_.n_rows(),qset_.n_cols());

    //Initialize the expoential_term
    non_exponential_term.Init(qset_.n_cols());

    exponential_term.SetZero();
    non_exponential_term.SetZero();

    bandwidth_=bandwidth;
 
  }
};//END OF CLASS FASTLLDE

#endif
