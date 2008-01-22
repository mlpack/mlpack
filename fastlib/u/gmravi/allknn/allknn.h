/** @file allknn.h
 *  
 * In this file we declare 5 classes, which do nn or knn search by using the     
 * naive search method or by using dual trees 
 *
 * @see allknn_main.cc
 */

#include "fastlib/fastlib_int.h"
#include <values.h>
#ifndef ALLKNN_H
#define ALLKNN_H
#define LEAF_SIZE 3
//Class definition for AllNNNaive.......................

/** This class supports functions to find out 1 single 
 *  nearest neighbour for each query point
 */

class AllNNNaive{
  
 private:
  Matrix q_matrix_;
  Matrix r_matrix_;

  /** We shall use a class called NaiveResults which has 2
   *  elements. An array List of indexes of the closest single
   *  neighbour for a given query point and an 
   *  arraylist of the distance of the nearest neighbour from
   *  the query point considered
   */

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

  // Constructor
  AllNNNaive(){

  }

  //Destructor 

 ~AllNNNaive(){

  }


  /** This function will simply load the query files 
    *  and the reference files specified by the user
    *  into matrices
    */
  void Init(Matrix &q_matrix, Matrix &r_matrix)
    {
      //First Copy the dataset
      q_matrix_.Alias(q_matrix);
      r_matrix_.Alias(r_matrix);

      //An object naive of the class NaiveResults was created
      //and initalized 
      //allocate memory for index_of_neighbour, distance_sqd. Note since 
      //we are finding exactly 1 nn for each query point, 
      //we shall allocate memory 
      //equal to the number of query points. 
      //This is what is being done in the code below

      results.Init();
      results.index_of_neighbour.Init(get_query_set().n_cols());
      results.distance_sqd.Init(get_query_set().n_cols());
    }
  
  /** This does the naive computation of comparing the 
    *  query point to all the reference points
    */
  
  void ComputeAllNNNaive()
    {
      //for each column vector of q_matrix find the 
      //nearest neighbour in r_matrix
      double min_dist;
      double dist;  
      
      for(index_t cols_q=0; cols_q < q_matrix_.n_cols(); cols_q++)
	{
	  min_dist=DBL_MAX;
	  dist=0.0;
	  for(index_t cols_r=0; cols_r<r_matrix_.n_cols(); cols_r++)
	    {
	      dist=la::DistanceSqEuclidean(get_query_set().n_rows(),
					   get_query_set().GetColumnPtr(cols_q),
					   get_reference_set().GetColumnPtr(cols_r));
	      
	      //distance!=0 guarantees that i am not 
	      //comparing the same points to find the nearest neighbour
	      if(dist<min_dist && dist!=0.0)
		{
		  results.index_of_neighbour[cols_q]=cols_r;
		  results.distance_sqd[cols_q]=dist;
		  min_dist=dist;
		}
	    }
	  //printf("min dist is %f\n",min_dist);
	}
    }
  
  /** This is a simple print module and it prints the 
   *  nearest neighbour to each point onto a file 
   */

  void PrintResults()
    {
      FILE *fp;
      fp=fopen("naive_nearest_neighbour.txt","w+");
      for(index_t i=0;i< q_matrix_.n_cols();i++){
	  
	fprintf(fp,"Point:%d\tDistance:%f\n", i,results.distance_sqd[i]); 
      }
    }
};    //End of class definition of AllNNNaive........................................

/** This class has functions to compute the 1- nearest neighbour for each query point
 *  It uses a single tree built out of the reference dataset and for each query point
 *  finds out the nearest neihbour in the reference set by traversing the reference tree
 */


class AllNNSingleTree{

 public:

 //forward declaration sof class
  class SingleTreeResults;
  

  /** This class will hold the index of the nearest neighbour reference point 
   *  and the distance of the nearest neighbour to the query point
  */

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

  /** The reference kd-tree built can store statistics if required
   * however in this case there is no nee dfor us to store any statistic
   * hence we declare an empty class 
   */

  class SingleTreeStat;
  typedef BinarySpaceTree < DHrectBound < 2 >, Matrix, SingleTreeStat > Tree;
  
  class SingleTreeStat {

  public:
    
    void Init(){
      
    }
    
    void Init(const Matrix& dataset, index_t start, index_t count) {
      
      Init();
    }
    
    void Init(const Matrix& dataset, index_t start, index_t count,
	      const SingleTreeStat& left_stat, const SingleTreeStat& right_stat) {
      
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
  

  //Constructor
  AllNNSingleTree(){

  }

  //Destructor
  ~AllNNSingleTree(){

  }

  //Interesting functions....

  /** This function simply loads the reference dataset and the query dataset into matrices 
   *  and makes a tree form the reference dataset 
   */

  void Init(Matrix &q_matrix, Matrix &r_matrix){
    
    //First Copy the dataset
    q_matrix_.Alias(q_matrix);
    r_matrix_.Alias(r_matrix);
    
    
    //Initialize results
    
    results.Init();
    results.index_of_neighbour.Init(get_query_set().n_cols());
    results.distance_sqd.Init(get_query_set().n_cols());

    //Build the tree out of the reference set
    rroot_ = tree::MakeKdTreeMidpoint < Tree > (r_matrix_, LEAF_SIZE);

  }

  /** This takes two arrays and the length of the arrays and checks if they are equal. 
   *  I couldnt find a library function that does this job. hence had to write one
   */
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

	  double distance=DBL_MAX;
	  index_t index=-1;

	  //This function traverses the tree and finds out the nearest neighbour for each point 

	  FindNearestNeighbour(rroot_,q_matrix_.GetColumnPtr(q),r_matrix_,distance,index);
	  results.index_of_neighbour[q]=index;
	  results.distance_sqd[q]=distance;
	 
	} 
    }
  

  /** This is the function which finds out the nearest neighbour 
    * to the given query point 
    * it does so by traversing the tree depth first 
    * and testing if their is  a potential 
    * neighbour at a distance which is lesser than the 
    * present estimate for the nearest neighbour
    */

  /*rnode-- reftree node
   *point- the query point whose nearest neighbour we need to find 
   *r_matrix - the reference set as a matrix
   *potential_distance: The distance of the nearest neighbour estimate we have as of now
   *potential_index: the index in the reference set of the potential neighbour
  */ 
  void FindNearestNeighbour(Tree *rnode,double *point, Matrix r_matrix, 
			    double &potential_distance, index_t &potential_index){
    

    //check if it is the leaf 
    if(rnode->is_leaf())
      {
	//find out the minimum distance by querying all points. 
	//Remember we need to find neighbours within the potential_distance estimate we have
	
	for(index_t i=rnode->begin();i<rnode->end();i++)
	  {
	    double temp_dist=la::DistanceSqEuclidean(r_matrix_.n_rows(),r_matrix_.GetColumnPtr(i),point);
	    
	    //check_if_equal function is being called to 
	    //avoid comparison between the same points
	    //This function will not be requied if we are sure 
	    //that the query and reference sets 
	    //have no common points

	    if(check_if_equal(r_matrix_.GetColumnPtr(i),point,r_matrix.n_rows())==false && temp_dist < 
	       potential_distance) {
	      
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
	
	
	double nearest_bb_distance=min_distance_to_left_child <
	  min_distance_to_right_child?
	  min_distance_to_left_child:
	min_distance_to_right_child;
	
	//continue searching only if the nearest bounding box is 
	//closer than the potential nn we have as of now
	if(potential_distance > nearest_bb_distance){
	  
	  Tree *farther;
	  if(min_distance_to_left_child < min_distance_to_right_child){
	    
	    //recursively explore the left half
	    farther=rnode->right();
	    FindNearestNeighbour(rnode->left(),
				 point,r_matrix,potential_distance,potential_index);
	  }
	  
	  else{
	    
	    //recursively explore the right half
	    
	    farther=rnode->left();
	    FindNearestNeighbour(rnode->right(),
				 point,r_matrix,potential_distance,potential_index);	
	  }
	  
	  //check if the potential distance u 
	  //have is more than the distance to the farther bb
	  
	  if(potential_distance>farther->bound().MinDistanceSq(point)){
	    
	    FindNearestNeighbour(farther,point,r_matrix,potential_distance,potential_index);	    
	  }
	}
	
      }
  }


  /** A module to print the results onto a file
   */

  void PrintResults(){

    FILE *fp;
    fp=fopen("allnnsingletree_nearest_neighbour.txt","w+");
    for(index_t i=0;i< q_matrix_.n_cols();i++){
      
      fprintf(fp,"Point:%d\tDistance:%f\n", i,results.distance_sqd[i]); 
    }
  }
  
};   //Definition of AllNNSingleTree ends.........................


/** This class has functions to find the k-nearest neighbours
 * by recursing the reference tree
 */

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
  };  //definition of class AllKNNSingleTreeResults ends...............

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
    
    void Init(const Matrix& dataset, index_t start, index_t count,
	      const AllKNNSingleTreeStat& left_stat, 
	      const AllKNNSingleTreeStat& right_stat) {
      
      Init();
    }
    
  };  //definition for AllKNNSingleTreeStat ends here.........

private:

  Matrix q_matrix_;
  Matrix r_matrix_;
  Tree *rroot_;
  index_t k_;

 public:

  //Constructor
 AllKNNSingleTree()
    {
     
    }

  //Destructor
  ~AllKNNSingleTree()
    {
      delete(rroot_);
    }

  void Init(Matrix &q_matrix, Matrix &r_matrix, index_t k){

    //Copy datasets

    q_matrix_.Alias(q_matrix);
    r_matrix_.Alias(r_matrix);
    k_=k;
    
    //Initialize results

    results=new AllKNNSingleTreeResults[q_matrix_.n_cols()];

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

  //This is the function which will find out 
  //the k- nearest neighbour for all the query point
  
  void ComputeAllKNNSingleTree()
    {
      
      //temp_result holds the result of k-nearest computation of a single point. 
      //This will be used again and again in iterations

      AllKNNSingleTreeResults temp_result;

      temp_result.Init();
      temp_result.index_of_neighbour.Init(k_);
      temp_result.distance_sqd.Init(k_);


      //Initialize temp_result with values that imply that the 
      //nearest neighbours are infinitely far off

      for(index_t i=0;i<k_;i++){
	temp_result.index_of_neighbour[i]=-1;
	temp_result.distance_sqd[i]=DBL_MAX;
      }

       for(int i=0;i<q_matrix_.n_cols();i++)
	 {   
	   index_t length=0;

	   //length is the number of neighbours found till now. Initialized to 0
	   FindKNearestNeighbours(rroot_,q_matrix_.GetColumnPtr(i),
				  &r_matrix_,temp_result,length,k_);

	   results[i].index_of_neighbour.Copy(temp_result.index_of_neighbour);
	   results[i].distance_sqd.Copy(temp_result.distance_sqd);	
   
	   //Flush all these values for further usage in the iteration
	   for(index_t l=0;l<k_;l++){
	     temp_result.index_of_neighbour[l]=-1;
	     temp_result.distance_sqd[l]=DBL_MAX;
	   }
	 }
     }

  /** The FindKNearestNeighbours function finds out the 
    * k-nearest neighbours for each query point
    * the length paramter tells us how many of the k-nearest 
    * neighbours have been found. Hence
    * this param is set to 0 
  */


  void FindKNearestNeighbours(Tree *rnode, double *point,Matrix *r_matrix,
			      AllKNNSingleTreeResults &result,int &length,int k_){
    
    //Base case is that the node is a leaf 
       
          if(rnode->is_leaf()){

	   int position;
	  
	   double dist;
	   int end;

	   //length=0 => no neighbours have been found and hence end=-1 in such a  case
	   
	   /*end points to the position where the arrays in the class 
	    * AllKNNSingleTreeResults ends. 
	    * That is it is the index of the last element 
	   */

	   end=length-1; 
	   for(int i=rnode->begin();i<rnode->end();i++){
	     
	     //one very important check is to see that 
	     //query point is not the same point as any other point being compared
	     
	     if(!check_if_equal(point,r_matrix->GetColumnPtr(i),r_matrix->n_rows()))
	       { 
		 dist=la::DistanceSqEuclidean(r_matrix->n_rows(),point,
					      r_matrix->GetColumnPtr(i)); 
	       
		 //find where the new element should be pushed
		 
		 int start=0;

		 //find_index function findws what would be the position of the 
		 // potential k-nearest neighbour which is at a distance dist as 
		 //calculated above
 
		 position=find_index(result,dist,start,end); 
		 

		 //push into array returns 1 if an element is pushable. 
		 //an element is pushable if it is a possible knn
	      
		 length+=push_into_array(result,position,dist,length,i,k_);
		 end=length-1; 
		 
	       }
	     
	   }
	  }
	  
	  else
	    {
	      //this is not a root node. Hence find 
	      //the distance to the bounding boxes
	      
	      double min_distance_to_left_child=rnode->left()->bound().MinDistanceSq(point);
	      double min_distance_to_right_child=rnode->right()->bound().MinDistanceSq(point);
	      
	      double min_dist_to_bb=min_distance_to_left_child >
		min_distance_to_right_child ?
		min_distance_to_right_child :
		min_distance_to_left_child;

	   Tree *farther;

	   //This condition considers further recursion only if all k nearest neighbours 
	   //have not been found or if distance of the nearest bb 
	   //is less than the distance of the kth nearest neighbour

	  
	   if(length<k_ ||min_dist_to_bb < result.distance_sqd[k_-1]){

	     if(min_distance_to_left_child < min_distance_to_right_child){

	       //Recursively explore the left child
	       farther=rnode->right();
	       FindKNearestNeighbours(rnode->left(),point,r_matrix,result,length,k_);
	     }

	   else{

	     //Recursively explore the right child
	     farther=rnode->left();
	     FindKNearestNeighbours(rnode->right(),point,r_matrix,result,length,k_);	       
	     }
	   
	     //If number of neighbours found are less than 
	     //k then go ahead and explore the other half too
	   
	     if(length<k_){
	       
	       //recursively explore the farther child		 
	       FindKNearestNeighbours(farther,point,r_matrix,result,length,k_);
	     }

	     else
	       {
		 //This means i have k nearest neighbours
		 //check the other half only if the kth nn distance is greater 
		 //than the distance of the point form the farther bounding box

		 if(result.distance_sqd[k_-1]> farther->bound().MinDistanceSq(point)){
		   FindKNearestNeighbours(farther,point,r_matrix,result,length,k_);
		 }
	       }
	   }
	    }
  }
  
  /* This function finds where the element whose distance 
   * is dist from the query point should be pushed in the array results
   */

   int find_index(AllKNNSingleTreeResults &result,
		  double dist,int start,int end){
 
     //this means that there are no elements in the array
     if(start>end){

       return 0;
     }

  //this means there is exactly 1 element in the array
  if(start==end)
    {
     
      if(dist>result.distance_sqd[start])
	{
	  /* the element should be added to the back of the array*/
	 
	  return end+1; 
	}
    
      else {
   
	return start;
      }
    }

  // find where the element will be in the sorted array. 
  // This is just the binary search

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

   /** This will push the element into result at the 
    *  appropriate place, so that the distances 
    *  are in ascending order 
    *  It will return 1 if there is an increase in the 
    *  length of the array else it will return 0 
    */

   int push_into_array(AllKNNSingleTreeResults &result,int position, 
		       double dist,int length, int index,int k_)
   {
    
     if(position==length) //that means add the element to the end of list
       {
	 if(length==k_){
	 
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
       
	 for(int t=length-1;t>position;t--){

	   temp.distance_sqd[t]=result.distance_sqd[t-1];
	   temp.index_of_neighbour[t]=result.index_of_neighbour[t-1];

	 }
	 //copy temp back to result
	 for(int j=0;j<length;j++){
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
   for(index_t i=0;i< q_matrix_.n_cols();i++){
     for(index_t l=0;l<k_;l++){
       
       fprintf(fp,"Point:%d\tDistance:%f\n", i,results[i].distance_sqd[l]); 
     }
   }
 }

};       //Definition of AllKNNSingleTree ends............


//Definition of AllKNNDualTree begins.........................

/** So this class find sout the k-nearest neighbours for all the query points by 
 *building up 2 -trees one out of the reference set and another f the query set 
 */

class AllKNNDualTree{
  
 public:

 //forward declaration sof class
  class AllKNDualTreeResults;
  
  class AllKNNDualTreeResults{
    
  public:
    ArrayList <index_t> index_of_neighbour;
    ArrayList <double> distance_sqd;
    
    void Init(){  
      
      

    }
  };  //definition of class AllKNNDualTreeResults ends..................

  AllKNNDualTreeResults *results;

  //forward declaration of AllKNNDualTreeStat

  class AllKNNDualTreeStat;
  typedef BinarySpaceTree < DHrectBound < 2 >, Matrix, AllKNNDualTreeStat > Tree;
  
  class AllKNNDualTreeStat {


  public:
  
    /** This gives the maximum distance within 
     * which one will find all the knn for the node
     */

    double distance_max;
    
    
    void Init(){

      distance_max=DBL_MAX;
      
    }
    
    void Init(const Matrix& dataset, index_t start, index_t count) {
      
      Init();
    }
    
    void Init(const Matrix& dataset, index_t start, index_t count,
	      const AllKNNDualTreeStat& left_stat, 
	      const AllKNNDualTreeStat& right_stat) {
      
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
      delete(rroot_);
    }
  //Constructor

  AllKNNDualTree()
    {
     
    }

  void Init(Matrix &q_matrix, Matrix &r_matrix,index_t k){

    q_matrix_.Alias(q_matrix);
    r_matrix_.Alias(r_matrix);
    k_=k;    
    //Initialize results

    results=new AllKNNDualTreeResults[q_matrix_.n_cols()];

    for(int i=0;i<q_matrix_.n_cols();i++){
      results[i].index_of_neighbour.Init();
      results[i].distance_sqd.Init();
     }
   
    //Build the tree out of the reference set
    rroot_ = tree::MakeKdTreeMidpoint < Tree > (r_matrix_, LEAF_SIZE,NULL,NULL);

    qroot_ = tree::MakeKdTreeMidpoint < Tree > (q_matrix_, LEAF_SIZE,&old_from_new,NULL);
  }
  
      
  //interesting functions....................

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


  //This is the function which will perform 
  //the actual dual tree algorithm and spit the results 

  void ComputeAllKNNDualTree() 
    {
       
      FindKNearestNeighboursDualTree(qroot_, rroot_);

      //Note that the order of points in the query and 
      //reference set have changed due to tree formation.
      // hence map the values properly

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
	  
	   
	   for(int i=q_node->begin();i<q_node->end();i++){
	     
	     //for each query point in the node
	     
	     for(int j=r_node->begin();j<r_node->end();j++){ 

		//for each reference point in the reference node

	       if(!check_if_equal(q_matrix_.GetColumnPtr(i),r_matrix_.GetColumnPtr(j),
				  q_matrix_.n_rows())) 
		    {
		      
		      distance=la::DistanceSqEuclidean (r_matrix_.n_rows(),
							q_matrix_.GetColumnPtr(i),
							r_matrix_.GetColumnPtr(j));

		      //we would like to find index of this point into str. 
		      //this will enter into str[i]. This function takes in as 
		      //argument an array list of single tree results           

		      start=0;
		      /* points to the index of the last element*/

		      end=results[i].distance_sqd.size()-1;

		      //this calculates the old length		     		      
		      int length=results[i].distance_sqd.size();
	             
		      //Find where this element should be inserted
		      position=find_index(results[i],distance,start,end);
		      if(length<k_){
			 
			//increase the length of the results[i]. 
			//This creates an additional pocket to hold an extra element

			  results[i].distance_sqd.AddBack(1); 
			  results[i].index_of_neighbour.AddBack(1);
			}

		      //Note the length is still the old length, 
		      //the one that has been claulated in the step above. 
		      //So it is 1 less than the actual length

		      push_into_array(results[i],position,distance,length,j);
		     
		    }
		
		}

	      //see if all knn have been found
	      if(results[i].distance_sqd.size()<k_){
		
		q_node->stat().distance_max=DBL_MAX;
	      }
	      else {
		
		//all k nn have been found
		q_node->stat().distance_max=q_node->stat().distance_max > 
		  results[i].distance_sqd[k_-1]?
		  q_node->stat().distance_max:
		  results[i].distance_sqd[k_-1];
		
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
	 
	  double max_dist_q=max_dist_q_left > 
	    max_dist_q_right?
	    max_dist_q_left:
	  max_dist_q_right;

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
		
		double max_dist_q= max_dist_q_left> 
		  max_dist_q_right? 
		  max_dist_q_left: 
		max_dist_q_right;
		q_node->stat().distance_max=max_dist_q;
	      }
	  }
      }
    }
 }

 int find_index(AllKNNDualTreeResults &result,double dist,int start,int end){
  
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

  //find where the element will be in the sorted array. 
  //This is just the binary search

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
 /** This will push the element into result. 
  * It will return 1 if there is an increase in the length 
  * of the array else it will return 0 
  */

 int push_into_array(AllKNNDualTreeResults &result,int position, 
		     double dist,int length, int index)
   {
 
     if(position==length) 
       {
	 //add the element to the end of list

	 if(length==k_){
	   
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
       
       for(int t=length-1;t>position;t--){
	 
	 temp.distance_sqd[t]=result.distance_sqd[t-1];
	 temp.index_of_neighbour[t]=result.index_of_neighbour[t-1];
	 
       }
       //copy temp back to result
       for(int j=0;j<length;j++){
	
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
   fp=fopen("allknndualtree_nearest_neighbour.txt","w+");
  
   for(index_t i=0;i< q_matrix_.n_cols();i++){
     for(index_t l=0;l<k_;l++){
       
       fprintf(fp,"Point:%d\tDistance:%f\n", i,results[i].distance_sqd[l]); 
     }
   }
 }
};

#endif
