
//Lets do it step-by-step
#include "fastlib/fastlib_int.h"
#ifndef ALLKNN_H
#define ALLKNN_H
#define LEAF_SIZE 3
#define MAX_DOUBLE 32768.0
//Class definition for AllNNNaive..............................................

class AllNNNaive{
  
 private:
  Matrix q_matrix_;
  Matrix r_matrix_;


 public:

  //forward declaration of class
  class NaiveResults;
  
  class NaiveResults{

  public:
    ArrayList <index_t> index_of_neighbour;
    ArrayList <double> distance_sqd;
    void Init()
      {

      }

  

  }; //naive results ends here

  NaiveResults results;

  //getters and setters
  Matrix& get_query_set(){

    return q_matrix_;
  }

  Matrix& get_reference_set(){

      return r_matrix_;
  }

  ArrayList<double>& get_results(){

    return results.distance_sqd;
  }

  //Interesting functions.......
 public:
  void Init()
    {
      char *qfile=(char*)malloc(40);
      char *rfile=(char*)malloc(40);
      strcpy(qfile,fx_param_str_req(NULL,"qfile"));
      strcpy(rfile,fx_param_str_req(NULL,"rfile"));
      
      data::Load(qfile,&q_matrix_);
      data::Load(rfile,&r_matrix_); 

      //initialize index_of_neighbour, distance_sqd
      results.Init();
      results.index_of_neighbour.Init(get_query_set().n_cols());
      results.distance_sqd.Init(get_query_set().n_cols());
      printf("number of columns of the query set are %d..\n",get_query_set().n_cols());
      
    }
  
  
  void ComputeAllNNNaive()
    {
      //for each column vector of q_matrix find the nearest neighbour in r_matrix
      double min_dist;
      double dist;  
      
      for(index_t cols_q=0; cols_q < q_matrix_.n_cols(); cols_q++)
	{
	  min_dist=MAX_DOUBLE;
	  dist=0.0;
	  for(index_t cols_r=0; cols_r<r_matrix_.n_cols(); cols_r++)
	    {
	      dist=la::DistanceSqEuclidean(get_query_set().n_rows(),get_query_set().GetColumnPtr(cols_q),get_reference_set().GetColumnPtr(cols_r));
	      
	      //distance!=0 guarantees that i am not comparing the same points to find the nearest neighbour
	      if(dist<min_dist && dist!=0.0)
		{
		  results.index_of_neighbour[cols_q]=cols_r;
		  results.distance_sqd[cols_q]=dist;
		  min_dist=dist;
		}
	    }
	  printf("min dist is %f\n",min_dist);
	}
    }
  
  void PrintResults()
    {
      FILE *fp;
      fp=fopen("naive_nearest_neighbour.txt","w+");
      printf("The nearest neighbours are as follows\n");
      // printf("The number of results are\n");
      //printf("%d\n",results.size());
      for(index_t i=0;i< q_matrix_.n_cols();i++){
	  
	fprintf(fp,"Point:%d\tDistance:%f\n", results.index_of_neighbour[i],results.distance_sqd[i]); 
      }
    }
};    //End of class definition of AllNNNaive........................................



class AllNNSingleTree{

 public:

 //forward declaration sof class
  class SingleTreeResults;
  
  class SingleTreeResults{
    
  public:
    ArrayList <index_t> index_of_neighbour;
    ArrayList <double> distance_sqd;
    
    void Init(){  
      
      //Init of SingleTreeResults

    }
  };  //definition of class SingleTreeResults ends..................................

  SingleTreeResults results;

  //forward declaration of SingleTreeStat

  class SingleTreeStat;
  typedef BinarySpaceTree < DHrectBound < 2 >, Matrix, SingleTreeStat > Tree;
  
  class SingleTreeStat {

  public:
    
    void Init(){
      
    }
    
    void Init(const Matrix& dataset, index_t start, index_t count) {
      
      Init();
    }
    
    void Init(const Matrix& dataset, index_t start, index_t count,const SingleTreeStat& left_stat, const SingleTreeStat& right_stat) {
      
      Init();
    }
    
  };  //definition fo SingleTreeStat ends here.....


 private:
  double distance_;
  index_t index_;
  Matrix q_matrix_;
  Matrix r_matrix_;
  Tree *rroot_;

  

//Definition of AllNNSingleTree begins........................................


 public:

  ArrayList<double>& get_results(){

    return results.distance_sqd;
  }

  //getters and setters
  Matrix& get_query_set(){
    
    return q_matrix_;
  }
  
  Matrix& get_reference_set(){
    
    return r_matrix_;
  }
  

  //Interesting functions....


  void Init(){

    char *qfile=(char*)malloc(40);
    char *rfile=(char*)malloc(40);
    strcpy(qfile,fx_param_str_req(NULL,"qfile"));
    strcpy(rfile,fx_param_str_req(NULL,"rfile"));
    
    data::Load(qfile,&q_matrix_);
    data::Load(rfile,&r_matrix_); 

    //Initialize results

    results.Init();
    results.index_of_neighbour.Init(get_query_set().n_cols());
    results.distance_sqd.Init(get_query_set().n_cols());

    //Build the tree out of the reference set
    rroot_ = tree::MakeKdTreeMidpoint < Tree > (r_matrix_, LEAF_SIZE);

  }

  bool check_if_equal(double *arr1, double *arr2,index_t size)
    {
      index_t i=0;
      for(i=0;i<size;i++)
	{
	  if(arr1[i]!=arr2[i])
	    return false;
	}
      return true;
    }

  //This is the function which will perform the actual single tree algorithm and spit the results 
  
  void ComputeAllNNSingleTree(){
      
      //For each point find out the nearest neighbour
     
      for(index_t q=0;q<q_matrix_.n_cols();q++){

	  double distance=MAX_DOUBLE;
	  index_t index=-1;
	  FindNearestNeighbour(rroot_,q_matrix_.GetColumnPtr(q),r_matrix_,distance,index);
	  results.index_of_neighbour[q]=index;
	  results.distance_sqd[q]=distance;
	 
	} 
    }

  void FindNearestNeighbour(Tree *rnode,double *point, Matrix r_matrix, double &potential_distance, index_t &potential_index){


      //check if it is the leaf 
      if(rnode->is_leaf())
	{
	  //find out the minimum distance by querying all points. Remember we need to find neighbours within the potential_distance estimate we have
	  
	  for(index_t i=rnode->begin();i<rnode->end();i++)
	    {
	      double temp_dist=la::DistanceSqEuclidean(r_matrix_.n_rows(),r_matrix_.GetColumnPtr(i),point);

	      //check_if_equal function is being called to avoid comparison between the same points
	      if(check_if_equal(r_matrix_.GetColumnPtr(i),point,r_matrix.n_rows())==false && temp_dist < potential_distance) {
	      
		potential_distance=temp_dist;
		potential_index=i;
		}
	    }
	}
      
      else
	{
	  //This is not a leaf
	  //so find the nearer node and the farther node
	  
	  double min_distance_to_left_child=rnode->left()->bound().MinDistanceSq(point);
	  double min_distance_to_right_child=rnode->right()->bound().MinDistanceSq(point);

	  
	  double nearest_bb_distance=min_distance_to_left_child<min_distance_to_right_child?min_distance_to_left_child:min_distance_to_right_child;

	  //continue searching only if the nearest bounding box is closer than the potential nn we have as of now
	  if(potential_distance > nearest_bb_distance){

	      Tree *farther;
	      if(min_distance_to_left_child < min_distance_to_right_child){
		
		//recursively explore the left half
		farther=rnode->right();
		FindNearestNeighbour(rnode->left(),point,r_matrix,potential_distance,potential_index);
	      }
	      
	      else{
		
		//recursively explore the right half
		
		farther=rnode->left();
		FindNearestNeighbour(rnode->right(),point,r_matrix,potential_distance,potential_index);	
	      }
	      
	      //check if the potential distance u have is more than the distance to the farther bb
	      
	      if(potential_distance>farther->bound().MinDistanceSq(point)){
		
		FindNearestNeighbour(farther,point,r_matrix,potential_distance,potential_index);	    
	      }
	  }
	  
	}
  }

  void PrintResults(){

    FILE *fp;
    fp=fopen("allnnsingletree_nearest_neighbour.txt","w+");
    printf("The nearest neighbours are as follows\n");
    // printf("The number of results are\n");
    //printf("%d\n",results.size());
    for(index_t i=0;i< q_matrix_.n_cols();i++){
      
      fprintf(fp,"Point:%d\tDistance:%f\n", results.index_of_neighbour[i],results.distance_sqd[i]); 
    }
  }
  
};   //Definition of AllNNSingleTree ends.........................................




class AllKNNSingleTree
{

 public:

 //forward declaration sof class
  class AllKNSingleTreeResults;
  
  class AllKNNSingleTreeResults{

   
    
  public:
    ArrayList <index_t> index_of_neighbour;
    ArrayList <double> distance_sqd;
    
    void Init(){  
      
      //Init of AllKNNSingleTreeResults

    }
  };  //definition of class AllKNNSingleTreeResults ends..................................

  AllKNNSingleTreeResults *results;

  //forward declaration of AllKNNSingleTreeStat

  class AllKNNSingleTreeStat;
  typedef BinarySpaceTree < DHrectBound < 2 >, Matrix, AllKNNSingleTreeStat > Tree;
  
  class AllKNNSingleTreeStat {

  public:
    
    void Init(){
      
    }
    
    void Init(const Matrix& dataset, index_t start, index_t count) {
      
      Init();
    }
    
    void Init(const Matrix& dataset, index_t start, index_t count,const AllKNNSingleTreeStat& left_stat, const AllKNNSingleTreeStat& right_stat) {
      
      Init();
    }
    
  };  //definition for AllKNNSingleTreeStat ends here.........

private:

  Matrix q_matrix_;
  Matrix r_matrix_;
  Tree *rroot_;
  index_t k_;

 public:

  //Destructor
  ~AllKNNSingleTree()
    {
      /*for(int i=0;i<q_matrix_.n_cols();i++)
	{
	  delete(results[i].distance_sqd);
	  delete(results[i].index_of_neighbour);
	  }*/
      // delete(results);
      delete(rroot_);
    }

  void Init(){

    char *qfile=(char*)malloc(40);
    char *rfile=(char*)malloc(40);

    strcpy(qfile,fx_param_str_req(NULL,"qfile"));
    strcpy(rfile,fx_param_str_req(NULL,"rfile"));
    
    data::Load(qfile,&q_matrix_);
    data::Load(rfile,&r_matrix_); 
    delete(qfile);
    delete(rfile);
    k_=fx_param_int_req(NULL,"k");
    printf("k is %d\n",k_);
    
    //Initialize results

    results=new AllKNNSingleTreeResults[q_matrix_.n_cols()];

    /* for(int i=0;i<q_matrix_.n_cols();i++){
      results[i].index_of_neighbour.Init(k_);
      results[i].distance_sqd.Init(k_);
      }*/

    //Build the tree out of the reference set
    rroot_ = tree::MakeKdTreeMidpoint < Tree > (r_matrix_, LEAF_SIZE);
  }
  
      
  //interesting functions.............................................................

  ArrayList<double> &get_results(){

    return results->distance_sqd;

  }

  //checks if point1 is the same as point2

  bool check_if_equal(double *point1,double *point2, int len){
    
    index_t i=0;
    for(i=0;i<len;i++)
      {
	if(point1[i]!=point2[i])
	  return false;
      }
    return true;
  }

  //This is the function which will perform the actual single tree algorithm and spit the results 
  
  void ComputeAllKNNSingleTree()
    {
      
      //temp_result holds the result of k-nearest computation of a single point. This will be used again and again in iterations

      AllKNNSingleTreeResults temp_result;

      temp_result.Init();
      temp_result.index_of_neighbour.Init(k_);
      temp_result.distance_sqd.Init(k_);

      for(index_t i=0;i<k_;i++){
	temp_result.index_of_neighbour[i]=-1;
	temp_result.distance_sqd[i]=MAX_DOUBLE;
      }

       for(int i=0;i<q_matrix_.n_cols();i++)
	 {   
	   index_t length=0;
	   //length is the number of neighbours found till now. Initialized to 0
	   FindKNearestNeighbours(rroot_,q_matrix_.GetColumnPtr(i),&r_matrix_,temp_result,length,k_);
	   results[i].index_of_neighbour.Copy(temp_result.index_of_neighbour);
	   results[i].distance_sqd.Copy(temp_result.distance_sqd);	
   
	   printf("Need to flush all values..\n");
	   for(index_t l=0;l<k_;l++){
	     temp_result.index_of_neighbour[l]=-1;
	     temp_result.distance_sqd[l]=MAX_DOUBLE;
	   }
	 }
     }

   void FindKNearestNeighbours(Tree *rnode, double *point,Matrix *r_matrix,AllKNNSingleTreeResults &result,int &length,int k_){

     //Base case is that the node is a leaf 
       
          if(rnode->is_leaf()){

	   int position;
	  
	   double dist;
	   int end;

	   //length=0 => no neighbours have been found and hence end=-1 in such a  case
	   
	   end=length-1;  //end points to the position where the array ends. That is it is the index of the last element 
	    

	   for(int i=rnode->begin();i<rnode->end();i++){
	     printf("came to base case..\n");

	       //one very important check is to see that query point is not the same point as any other point being compared

	       if(!check_if_equal(point,r_matrix->GetColumnPtr(i),r_matrix->n_rows()))
		 { 
		   dist=la::DistanceSqEuclidean(r_matrix->n_rows(),point,r_matrix->GetColumnPtr(i)); 
		   printf("distance is %f\n",dist);
		  //find where the new element should be pushed

		   int start=0;
		   position=find_index(result,dist,start,end); 
		   printf("found index...\n");
                   //push into array returns 1 if an element is pushable. an element is pushable if it is a possible knn
		   printf("will push into array...\n");
		   length+=push_into_array(result,position,dist,length,i,k_);
		   end=length-1; 
		   
		 }
	       
	     }
	 }

       else
	 {
	   //this is not a root node. Hence find the distance to the bounding boxes
	   
	   double min_distance_to_left_child=rnode->left()->bound().MinDistanceSq(point);
	   double min_distance_to_right_child=rnode->right()->bound().MinDistanceSq(point);
	   
           double min_dist_to_bb=min_distance_to_left_child>min_distance_to_right_child?min_distance_to_right_child:min_distance_to_left_child;

	   Tree *farther;

	   //This condition considers further recursion only if all k nearest neighbours have not been found or if distance of the nearest bb is less than the distance of the kth nearest neighbour

	   printf("first check..\n");
	   if(length<k_ ||min_dist_to_bb < result.distance_sqd[k_-1]){

	     if(min_distance_to_left_child < min_distance_to_right_child){
	       printf("left child..\n");
	       
	       //Recursively explore the left child
	       farther=rnode->right();
	       FindKNearestNeighbours(rnode->left(),point,r_matrix,result,length,k_);
	     }

	   else{
	     printf("right child..\n");
	     //Recursively explore the right child
	     farther=rnode->left();
	     FindKNearestNeighbours(rnode->right(),point,r_matrix,result,length,k_);	       
	     }
	   
	   //If number of neighbours found are less than k then go ahead and explore the other half too
	   
	     if(length<k_){
	       printf("will explore the right child because all k nn have not been found...\n");
	       
	       //recursively explore the farther child		 
	       FindKNearestNeighbours(farther,point,r_matrix,result,length,k_);
	     }

	     else
	       {
		 //This means i have k nearest neighbours
		 //check the other half only if the kth nn distance is greater than the distance of the point form the farther bounding box
		 printf("will print distance..\n");
		 printf("distance is %f\n",result.distance_sqd[k_-1]);
		 if(result.distance_sqd[k_-1]> farther->bound().MinDistanceSq(point)){
		     FindKNearestNeighbours(farther,point,r_matrix,result,length,k_);
		   }
	       }
	   }
	 }
   }
	
   /* This function finds where the element whose distance is dist from the query point should be pushed in the array results*/

   int find_index(AllKNNSingleTreeResults &result,double dist,int start,int end){
     printf("Came to find index..\n");

 
     //this means that there are no elements in the array
     if(start>end){
       printf("will return 0..\n");
       return 0;
     }

  //this means there is exactly 1 element in the array
  if(start==end)
    {
     
      if(dist>result.distance_sqd[start])
	{
	  /* the element should be added to the back of the array*/
	  printf("will return %d\n",end+1);
	  return end+1; 
	}
    
      else {
	printf("return %d\n",start);
	return start;
      }
    }

  //find where the element will be in the sorted array. This is just the binary search

  if(dist==result.distance_sqd[(start+end)/2]){
    printf("Will return %d \n",(start+end)/2);  
    return (start+end)/2;
  }

  else
    {
     
      if(dist<result.distance_sqd[(start+end)/2]) {
	  //go left

	  return find_index(result,dist,start,(start+end)/2);
	}

      else{

	//go right
	return find_index(result,dist,(start+end)/2+1,end);
      }
    }
   }
   /* This will push the element into result. It will return 1 if there is an increase in the length of the array else it will return 0 */

 int push_into_array(AllKNNSingleTreeResults &result,int position, double dist,int length, int index,int k_)
   {
     printf("came to push into array..\n");
     if(position==length) //that means add the element to the end of list
       {
	 if(length==k_){
	   printf("returning w/o adding..\n");	
	   return 0;
	   }

	 else{
	     
	     //add it to the end of the array and return 1
	     result.distance_sqd[length]=dist;
	     result.index_of_neighbour[length]=index;
	     return 1;
	   }
       }
     
     
     AllKNNSingleTreeResults temp;
     temp.index_of_neighbour.Init(k_);
     temp.distance_sqd.Init(k_);

     //the element will be added in the middle of the array
     
     
     for(int j=0;j<position;j++){
       temp.distance_sqd[j]=result.distance_sqd[j];
       temp.index_of_neighbour[j]=result.index_of_neighbour[j];
     }
     
   
     temp.distance_sqd[position]=dist;
     temp.index_of_neighbour[position]=index; 
     
     if(length==k_){
       printf("I kicked the last element..\n");

	 for(int t=length-1;t>position;t--){

	   temp.distance_sqd[t]=result.distance_sqd[t-1];
	   temp.index_of_neighbour[t]=result.index_of_neighbour[t-1];

	 }
	 //copy temp back to result
	 for(int j=0;j<length;j++){
	   printf("copying distance %f\n",temp.distance_sqd[j]);
	   result.distance_sqd[j]=temp.distance_sqd[j];
	   result.index_of_neighbour[j]=temp.index_of_neighbour[j];
	 }

	 return 0;
       }

     else{
       //this is because an element hase been added
	 for(int t=length;t>position;t--){

	   temp.distance_sqd[t]=result.distance_sqd[t-1];
	   temp.index_of_neighbour[t]=result.index_of_neighbour[t-1];
	 }
	 
	 
	 for(int j=0;j<length+1;j++){

	   result.distance_sqd[j]=temp.distance_sqd[j];
	   result.index_of_neighbour[j]=temp.index_of_neighbour[j];
	 }
	 return 1;
       }
   }

 void PrintResults(){
   
   FILE *fp;
   fp=fopen("allknnsingletree_nearest_neighbour.txt","w+");
   printf("The nearest neighbours are as follows\n");
   // printf("The number of results are\n");
   //printf("%d\n",results.size());
   for(index_t i=0;i< q_matrix_.n_cols();i++){
     for(index_t l=0;l<k_;l++){

       fprintf(fp,"Point:%d\tDistance:%f\n", i,results[i].distance_sqd[l]); 
     }
   }
 }

};       //Definition of AllKNNSingleTree ends................................................................................



//Definition of AllKNNDualTreeResults begins................................................................................


class AllKNNDualTree{

 public:

 //forward declaration sof class
  class AllKNDualTreeResults;
  
  class AllKNNDualTreeResults{

  public:
    ArrayList <index_t> index_of_neighbour;
    ArrayList <double> distance_sqd;
    
    void Init(){  
      
      //Init of AllKNNDualTreeResults

    }
  };  //definition of class AllKNNDualTreeResults ends..................................

  AllKNNDualTreeResults *results;

  //forward declaration of AllKNNDualTreeStat

  class AllKNNDualTreeStat;
  typedef BinarySpaceTree < DHrectBound < 2 >, Matrix, AllKNNDualTreeStat > Tree;
  
  class AllKNNDualTreeStat {


  public:
  
    /** This gives the maximum distance within which one will find all the knn for the node*/
    double distance_max;
    
    
    void Init(){

      distance_max=MAX_DOUBLE;
      
    }
    
    void Init(const Matrix& dataset, index_t start, index_t count) {
      
      Init();
    }
    
    void Init(const Matrix& dataset, index_t start, index_t count,const AllKNNDualTreeStat& left_stat, const AllKNNDualTreeStat& right_stat) {
      
      Init();
    }
    
  };  //definition for AllKNNDualTreeStat ends here.........

private:

  Matrix q_matrix_;
  Matrix r_matrix_;
  Tree *rroot_;
  Tree *qroot_;
  index_t k_;
  ArrayList<index_t> old_from_new;

 public:

  //Destructor
  ~AllKNNDualTree()
    {
      /*for(int i=0;i<q_matrix_.n_cols();i++)
	{
	  delete(results[i].distance_sqd);
	  delete(results[i].index_of_neighbour);
	  }*/
      // delete(results);
      delete(rroot_);
    }

  void Init(){

    char *qfile=(char*)malloc(40);
    char *rfile=(char*)malloc(40);

    strcpy(qfile,fx_param_str_req(NULL,"qfile"));
    strcpy(rfile,fx_param_str_req(NULL,"rfile"));
    
    data::Load(qfile,&q_matrix_);
    data::Load(rfile,&r_matrix_); 
    delete(qfile);
    delete(rfile);
    k_=fx_param_int_req(NULL,"k");
    printf("k is %d\n",k_);
    
    //Initialize results

    results=new AllKNNDualTreeResults[q_matrix_.n_cols()];
    //Initialize to 0 size
     for(int i=0;i<q_matrix_.n_cols();i++){
      results[i].index_of_neighbour.Init();
      results[i].distance_sqd.Init();
     }
     //old_from_new.Init(q_matrix_.n_cols());

    //Build the tree out of the reference set
    rroot_ = tree::MakeKdTreeMidpoint < Tree > (r_matrix_, LEAF_SIZE,NULL,NULL);

    qroot_ = tree::MakeKdTreeMidpoint < Tree > (q_matrix_, LEAF_SIZE,&old_from_new,NULL);
  }
  
      
  //interesting functions.............................................................

  //checks if point1 is the same as point2
  bool check_if_equal(double *point1,double *point2, int len){
    
    index_t i=0;
    for(i=0;i<len;i++)
      {
	if(point1[i]!=point2[i])
	  return false;
      }
    return true;
  }


  /* ArrayList<double>& get_results(){
    
    return results.distance_sqd;
    } */ 

  //This is the function which will perform the actual dual tree algorithm and spit the results 

  void ComputeAllKNNDualTree() 
    {
       
      FindKNearestNeighboursDualTree(qroot_, rroot_);

      //Note that the order of points in the query and reference set have changed due to tree formation. hence map the values properly
      TransformResults();

    }
  

  void TransformResults(){

    AllKNNDualTreeResults *temp;
    

    temp=new AllKNNDualTreeResults[q_matrix_.n_cols()];

    //Initialize to 0 size
    for(int i=0;i<q_matrix_.n_cols();i++){

      temp[i].index_of_neighbour.Init(k_);
      temp[i].distance_sqd.Init(k_);
     }

    //fill up the temporary variable first
    for(index_t i=0;i< q_matrix_.n_cols();i++){
      for(index_t l=0;l<k_;l++){

	temp[old_from_new[i]].distance_sqd[l]=results[i].distance_sqd[l];
	temp[old_from_new[i]].index_of_neighbour[l]=results[i].index_of_neighbour[l];
     }
   }
    //copy them back to results variable

    for(index_t i=0;i< q_matrix_.n_cols();i++){
      for(index_t l=0;l<k_;l++){
	results[i].distance_sqd[l]=temp[i].distance_sqd[l];
	results[i].index_of_neighbour[l]=results[old_from_new[i]].index_of_neighbour[l];
      }
    }

  }

void  FindKNearestNeighboursDualTree(Tree *q_node,Tree *r_node) 
 {

   //if distance between the two boxes is larger than the max _distance then return
   double distance_between_boxes=q_node->bound().MinDistanceSq (r_node->bound()); 

   //Base Case
   if(q_node->is_leaf()&& r_node->is_leaf())
     {
       int start,end;

       //check if pruneable
       if(q_node->stat().distance_max< distance_between_boxes)
	 {
	   
	   return;
	 }
       else
	 {
	   //not purneable. therefore carry out exhaustive point-to-point computations
	   
	   
	   double distance;
	   int position;
	  
	   
	   for(int i=q_node->begin();i<q_node->end();i++){ //for each query point in the node
	       
	      for(int j=r_node->begin();j<r_node->end();j++){ //for each reference point in the reference node

		  if(!check_if_equal(q_matrix_.GetColumnPtr(i),r_matrix_.GetColumnPtr(j),q_matrix_.n_rows())) //to make sure that we are not comparing the same set of points
		    {
		      
		      distance=la::DistanceSqEuclidean (r_matrix_.n_rows(),q_matrix_.GetColumnPtr(i),r_matrix_.GetColumnPtr(j));
		      //we would like to find index of this point into str. this will enter into str[i]. This function takes in as argument an array list of single tree results           

		      start=0;
		      /* points to the index of the last element*/
		      end=results[i].distance_sqd.size()-1;
		      
		      
		      int length=results[i].distance_sqd.size(); //this calculates the old length		     
	             
		      //Find where this element should be inserted
		      position=find_index(results[i],distance,start,end);
		      if(length<k_){
			 
			  //increase the length of the results[i]. This creates an additional pocket to hold an extra element
			  results[i].distance_sqd.AddBack(1); 
			  results[i].index_of_neighbour.AddBack(1);
			}
		      //Note the length is still the old length, the one that has been claulated in the step above. So it is 1 less than the actual length
		      push_into_array(results[i],position,distance,length,j);
		     
		    }
		
		}

	      //see if all knn have been found
	      if(results[i].distance_sqd.size()<k_){
		
		q_node->stat().distance_max=MAX_DOUBLE;
	      }
	      else {
		
		//all k nn have been found
		q_node->stat().distance_max=q_node->stat().distance_max > results[i].distance_sqd[k_-1]?q_node->stat().distance_max:results[i].distance_sqd[k_-1];
		
	      }
	   }
	 }
     }
   
   //not base case. 
   else
    {
      //Check if one can Prune
      if(q_node->stat().distance_max < distance_between_boxes)
	{
	  //then there is no need to go further and hence we can return
	  return; 
	}

      //NOT PRUNEABLE
      //both are not leafs
      if(!q_node->is_leaf() && !r_node->is_leaf()){
	 
	  FindKNearestNeighboursDualTree(q_node->left(),r_node->left());
	  FindKNearestNeighboursDualTree(q_node->left(),r_node->right());

	  double max_dist_q_left= q_node->left()->stat().distance_max;

	 
	  FindKNearestNeighboursDualTree(q_node->right(),r_node->left());
	  FindKNearestNeighboursDualTree(q_node->right(),r_node->right());

	  double max_dist_q_right= q_node->right()->stat().distance_max;
	 
	  double max_dist_q=max_dist_q_left>max_dist_q_right?max_dist_q_left:max_dist_q_right;
	  q_node->stat().distance_max=max_dist_q;
      }
      
      else{
	
	//q_tree is leaf and r_tree is not
	if(q_node->is_leaf()&&!r_node->is_leaf()){
	  
	  FindKNearestNeighboursDualTree(q_node,r_node->left());
	  FindKNearestNeighboursDualTree(q_node,r_node->right());
	}
	
	else
	  {
	    
	    //q_tree is not a leaf and r_tree is
	    
	    if(!q_node->is_leaf()&&r_node->is_leaf())
	      {
		
		FindKNearestNeighboursDualTree(q_node->left(),r_node);
		double max_dist_q_left=q_node->left()->stat().distance_max;
		
		FindKNearestNeighboursDualTree(q_node->right(),r_node);
		double max_dist_q_right=q_node->right()->stat().distance_max;
		
		double max_dist_q= max_dist_q_left> max_dist_q_right? max_dist_q_left: max_dist_q_right;
		q_node->stat().distance_max=max_dist_q;
	      }
	  }
      }
    }
 }

 int find_index(AllKNNDualTreeResults &result,double dist,int start,int end){
     printf("Came to find index..\n");

 
     //this means that there are no elements in the array
     if(start>end){
       printf("will return 0..\n");
       return 0;
     }

  //this means there is exactly 1 element in the array
  if(start==end)
    {
     
      if(dist>result.distance_sqd[start])
	{
	  /* the element should be added to the back of the array*/
	  printf("will return %d\n",end+1);
	  return end+1; 
	}
    
      else {
	printf("return %d\n",start);
	return start;
      }
    }

  //find where the element will be in the sorted array. This is just the binary search

  if(dist==result.distance_sqd[(start+end)/2]){
    printf("Will return %d \n",(start+end)/2);  
    return (start+end)/2;
  }

  else
    {
     
      if(dist<result.distance_sqd[(start+end)/2]) {
	  //go left

	  return find_index(result,dist,start,(start+end)/2);
	}

      else{

	//go right
	return find_index(result,dist,(start+end)/2+1,end);
      }
    }
   }
   /* This will push the element into result. It will return 1 if there is an increase in the length of the array else it will return 0 */

 int push_into_array(AllKNNDualTreeResults &result,int position, double dist,int length, int index)
   {
     printf("came to push into array..\n");
     if(position==length) //that means add the element to the end of list
       {
	 if(length==k_){
	   printf("returning w/o adding..\n");	
	   return 0;
	   }

	 else{
	     
	     //add it to the end of the array and return 1
	     result.distance_sqd[length]=dist;
	     result.index_of_neighbour[length]=index;
	     return 1;
	   }
       }
     
     
     AllKNNDualTreeResults temp;
     temp.index_of_neighbour.Init(k_);
     temp.distance_sqd.Init(k_);

     //the element will be added in the middle of the array
     
     
     for(int j=0;j<position;j++){
       temp.distance_sqd[j]=result.distance_sqd[j];
       temp.index_of_neighbour[j]=result.index_of_neighbour[j];
     }
     
   
     temp.distance_sqd[position]=dist;
     temp.index_of_neighbour[position]=index; 
     
     if(length==k_){
       printf("I kicked the last element..\n");

	 for(int t=length-1;t>position;t--){

	   temp.distance_sqd[t]=result.distance_sqd[t-1];
	   temp.index_of_neighbour[t]=result.index_of_neighbour[t-1];

	 }
	 //copy temp back to result
	 for(int j=0;j<length;j++){
	   printf("copying distance %f\n",temp.distance_sqd[j]);
	   result.distance_sqd[j]=temp.distance_sqd[j];
	   result.index_of_neighbour[j]=temp.index_of_neighbour[j];
	 }
	 printf("about to return...\n");
	 return 0;
       }

     else{

       printf("an element has been added..\n");
       //this is because an element hase been added
	 for(int t=length;t>position;t--){

	   temp.distance_sqd[t]=result.distance_sqd[t-1];
	   temp.index_of_neighbour[t]=result.index_of_neighbour[t-1];
	 }
	 
	 
	 for(int j=0;j<length+1;j++){
	   printf("copying  distance =%f\n",temp.distance_sqd[j]);

	   result.distance_sqd[j]=temp.distance_sqd[j];
	   result.index_of_neighbour[j]=temp.index_of_neighbour[j];
	 }
	 return 1;
       }
   }

 void PrintResults(){
   
   FILE *fp;
   fp=fopen("allknndualtree_nearest_neighbour.txt","w+");
   printf("The nearest neighbours are as follows\n");
  
   for(index_t i=0;i< q_matrix_.n_cols();i++){
     for(index_t l=0;l<k_;l++){
       
       fprintf(fp,"Point:%d\tDistance:%f\n", i,results[i].distance_sqd[l]); 
     }
   }
 }
};

#endif
