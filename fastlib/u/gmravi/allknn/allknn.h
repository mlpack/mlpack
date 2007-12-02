// This is the new header file that is being formed as a result of code improvement and code displacement

//Lets do it step-by-step
#include "fastlib/fastlib_int.h"
#ifndef ALLKNN_H
#define ALLKNN_H
#define LEAF_SIZE 3
#define MAX_DOUBLE 32768.0
//Class definition for AllNNNaive..............................................

class AllNNNaive
{
  
 public:
  ArrayList <index_t> results;
  AllNNNaive(){};


  //Interesting functions.......
  void Compute(Matrix *q_matrix, Matrix *r_matrix)
    {
      //for each column vector of q_matrix find the nearest neighbour in r_matrix
      
      // ArrayList <double> arr1;
      //ArrayList <double> arr2;
      index_t cols_q;
      index_t cols_r;
      double min_dist;
      double dist;
      //arr1.Init(q_matrix->n_rows()+1);
      //arr2.Init(q_matrix->n_rows()+1);
      
      results.Init(q_matrix->n_cols());
      
      //printf("cols_q is %d\n",q_matrix->n_cols());
      //printf("cols_r is %d\n",r_matrix->n_cols());
      //printf("entered nearest neighbour computation\n");
      
      for(cols_q=0; cols_q < q_matrix->n_cols(); cols_q++)
	{
	  min_dist=32768.0;
	  dist=0.0;
	  for(cols_r=0; cols_r < r_matrix->n_cols(); cols_r++)
	    {
	      using namespace la;
	      dist=la::DistanceSqEuclidean(q_matrix->n_rows(),q_matrix->GetColumnPtr(cols_q),q_matrix->GetColumnPtr(cols_r));
	      //printf("The distance is %f\n",dist);
	      if(dist<min_dist&&dist!=0.0)
		{
		  results[cols_q]=cols_r;
		  min_dist=dist;
		}
	    }
	  
	}
    }
  void PrintResults(Matrix *q_matrix, Matrix *r_matrix)
    {
      FILE *fp;
      fp=fopen("naive_nearest_neighbour.txt","w");
      printf("The nearest neighbours are as follows\n");
      // printf("The number of results are\n");
      //printf("%d\n",results.size());
      for(index_t i=0;i< results.size();i++)
	{
	  fprintf(fp,"Point:%d\tDistance:%f\n", i,la::DistanceSqEuclidean(r_matrix->n_rows(),r_matrix->GetColumnPtr(results[i]),q_matrix->GetColumnPtr(i)));
	  //fprintf(fp,"\n");
	  
	}
    }
};

//End of class definition of AllNNNaive........................................


//Definition of class Stat.....................................................
 class Stat 
{
  //This is not used in case of single tree operations. used only in case of dual tree operations
  
 private:
  double distance_max; //This gives the max distance within which all neighbours should be found for all points
 public:
  void set_maximum_distance(double distance)
    {
      this->distance_max=distance;
    }
  double get_maximum_distance()
    {
      return distance_max;
    }
  void Init() {set_maximum_distance(32768.0);}
  
  void Init(const Matrix& dataset, index_t start, index_t count) 
    {
      Init();
    }
  
  void Init(const Matrix& dataset, index_t start, index_t count,
	    const Stat& left_stat, const Stat& right_stat) 
    {
      Init();
    }
};

//End of definition of class Stat..............................................




//Definition of class SingleTreeResults begins................................

class SingleTreeResults
{
 private: 
  double distance_;
  index_t index_;
  
 public:

  //setter function
  void set_result(double distance, index_t index)
    {
      //Here index is the index in r_matrix which is the potential nn for a given query point
      this->distance_=distance;
      this->index_=index;
    }

  //getter function

  double get_distance()
    {
      return distance_;
    }

  index_t get_index()
    {
      return index_;
    }

 
  SingleTreeResults()
    {
      this->distance_=MAX_DOUBLE;
      this->index_=-1;
    }
};

//definition of class SingleTreeResults ends...................................



typedef BinarySpaceTree<DHrectBound<2>, Matrix, Stat> Tree;


//Definition of AllNNSingleTree begins........................................

class AllNNSingleTree
{
 public:
  ArrayList <SingleTreeResults> str;
 
  AllNNSingleTree()
    {
      //Initialize the results list
      str.Init();
    }
  

  //Interesting functions....

  //This is a friend function and can be invoked by other classes too. This takes in two points and sees if they are the same or not
  friend bool check_if_equal(double *arr1, double *arr2,index_t size)
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
  
  void ComputeAllNNSingleTree(Matrix *q_matrix, Matrix *r_matrix)
    {

      // expand the size of results
      str.GrowTo(q_matrix->n_cols()); 
      
      // Build the kdtree form the reference matrix.
      using namespace tree;
      Tree *root=tree::MakeKdTreeMidpoint<Tree>(*r_matrix,LEAF_SIZE,NULL,NULL);
      
      
      double max_dist_sq=MAX_DOUBLE;  
      
      //For each point find out the nearest neighbour
     
      for(index_t i=0;i<q_matrix->n_cols();i++)
	{
	  SingleTreeResults *nearest_nn=new SingleTreeResults();
	  str[i]=*FindNearestNeighbour(root,q_matrix->GetColumnPtr(i),r_matrix,nearest_nn);
	  printf("Distance to the nearest neighbour is %f\n",str[i].get_distance());
	  nearest_nn->set_result(MAX_DOUBLE,-1);  //This function flushes the values stored in nearest_nn, for the next iteration
	} 
    }

  SingleTreeResults *FindNearestNeighbour(Tree *root,double *point, Matrix *r_matrix, SingleTreeResults *nearest_nn)
    {
      //check if it is the leaf 
      if(root->is_leaf())
	{
	  //find out the minimum distance by querying all points

	  double min_dist_for_this_node=MAX_DOUBLE;
	  SingleTreeResults *potential_nn;
	  potential_nn=new SingleTreeResults();
	  potential_nn->set_result(32768.0,-1);
	  
	  for(index_t i=root->begin();i<root->end();i++)
	    {
	      //print(r_matrix->GetColumnPtr(i),r_matrix->n_rows());
	      //print(point,r_matrix->n_rows());
	      double temp_dist=la::DistanceSqEuclidean(r_matrix->n_rows(),r_matrix->GetColumnPtr(i),point);

	     
	      if(check_if_equal(r_matrix->GetColumnPtr(i),point,r_matrix->n_rows())==false && temp_dist < min_dist_for_this_node) //check_if_equal function is being called to avoid comparison between the same points
		{
		 
		  potential_nn->set_result(temp_dist,i);
		  min_dist_for_this_node=temp_dist;
		}
	    }
	  return potential_nn;
	}
      
      else
	{
	  //This is not a leaf
	  //so find the nearer node and the farther node
	  //WE CAN FIND OUT THE MINIMUM DISTANCE OF THE POINT TO THE BOUNDING BOX AND EXPLORE THE BOUNDING BOX WHICH IS THE CLOSEST
	  
	  float min_distance_to_left_child=root->left()->bound().MinDistanceSq(point);
	  float min_distance_to_right_child=root->right()->bound().MinDistanceSq(point);

	 
	  Tree *farther;
	  SingleTreeResults *potential_nn;
	  if(min_distance_to_left_child < min_distance_to_right_child)
	    {
	      //recursively explore the left half
	      farther=root->right();
	      potential_nn=FindNearestNeighbour(root->left(),point,r_matrix,nearest_nn);
	    }
	  else
	    {
	      //recursively explore the right half
	      
	      farther=root->left();
	      potential_nn=FindNearestNeighbour(root->right(),point,r_matrix,nearest_nn);	
	    }
	  
	  if(potential_nn->get_distance()< nearest_nn->get_distance())
	    {
	      nearest_nn->set_result(potential_nn->get_distance(),potential_nn->get_index());
	    }
	  
	  if(potential_nn->get_distance()>farther->bound().MinDistanceSq(point))
	    {
	      SingleTreeResults *potential_nn=FindNearestNeighbour(farther,point,r_matrix,nearest_nn);
	      if(potential_nn->get_distance()<nearest_nn->get_distance()) 
		{
		  nearest_nn->set_result(potential_nn->get_distance(),potential_nn->get_index());
		  return nearest_nn;
		}
	    }
	  else
	    {
	      //the farther node need not be explored . Simply return nearest_nn
	      return nearest_nn;
	    }
	  return nearest_nn;
	}
    }

  };


//Definition of AllNNSingleTree ends.........................................


//Definition of AllKNNSingleTreeResults begins....................................
class AllKNNSingleTreeResults
{
 public:
  ArrayList<ArrayList<SingleTreeResults> > astr; // u see it is basically a 2-D array of SingleTreeResults

  AllKNNSingleTreeResults()
    {
     
    }
};

//Definition of AllKNNSignleTreeResults ends here.................................................

//definition of AllKnnSingleTree begines....................................................
class AllKNNSingleTree
{
 public:

  friend bool same_points(double *point1,double *point2, int len);//already defined 
  AllKNNSingleTreeResults results_matrix;
  
  AllKNNSingleTree() 
    {
      //Initialize  results_matrix. This will later be reinitialized to the number of columns in the function ComputeAllKNNSingleTree
      // results_matrix.astr.Init();
     
    }


  //This is the function which will perform the actual single tree algorithm and spit the results 
  
   void ComputeAllKNNSingleTree(Matrix *q_matrix,Matrix *r_matrix,int k)
     {
       //Initialize results_matrix
       
       results_matrix.astr.Init(r_matrix->n_cols());
       for(int i=0;i<r_matrix->n_cols();i++)
	 results_matrix.astr[i].Init(k);
       
       //printf("results_matrix initialized\n");
       // Build the kdtree form the reference matrix.
       using namespace tree;
       
       Tree *root=tree::MakeKdTreeMidpoint<Tree>(*r_matrix,3,NULL,NULL); //here the leaf size is set to 3
        printf("Tree built\n");
       
       //Now for each query point find the k nearest neighbours
       
      
       
       for(int i=0;i<r_matrix->n_cols();i++)
	 {   
	   ArrayList<SingleTreeResults> str;
	   str.Init(k);
	   int length=0;
	   
	   //Length=k if there are more than k number of reference points.By clever programming the length parameter which is now being sent explicitly, can be totally avoided
	   length=FindKNearestNeighbours(root,q_matrix->GetColumnPtr(i),r_matrix,str,length,k);
	   
	   printf("The nearest neighbour distances are\n");
	   
	   for(int l=0;l<length;l++)
	     printf("%f\n",str[l].get_distance());
	   
	   
	   for(int j=0;j<length;j++)
	     {
	       //printf("Will add distance=%f\n",str[j].get_distance());
	       results_matrix.astr[i][j].set_result(str[j].get_distance(),str[j].get_index());
	     }
	   
	   
	   
	   printf("The origianl point is\n");
	   
	   for(int j=0;j<r_matrix->n_rows();j++)
	     printf("%f\n",q_matrix->GetColumnPtr(i)[j]);
	   
	 }
     }
   
   //A 1-D matrix of the k nearest neighbours is returned

   int FindKNearestNeighbours(Tree *root, double *point,Matrix *r_matrix,ArrayList<SingleTreeResults> &str,int length,int k)
     {
       //Base case is that the node is a leaf 
       
          if(root->is_leaf())
	 {
	   // printf("Will explore leaf\n");
	   int position;
	   //check for number of elements
	   double dist;
	   int end;
	   if(length==0)
	     { 
	       end=-1;  //end points to the position where the array ends. That is it is the index of the last element 
	     }
	   else
	     {
	       end=length-1;
	     }

	   for(int i=root->begin();i<root->end();i++)
	     {
	       //one very important check is to see that query point is not the same point as any other point beingcompared
	       if(!check_if_equal(point,r_matrix->GetColumnPtr(i),r_matrix->n_rows()))
		 { 
		   dist=la::DistanceSqEuclidean(r_matrix->n_rows(),point,r_matrix->GetColumnPtr(i)); 
		  //find where the new element should be pushed
		   int start=0;
		   position=find_index(str,dist,start,end); 
		  
		   length+=push_into_array(str,position,dist,length,i,k);
		   end=length-1; 
		  
		   //this is just like a test. One can safely remove it
		   if(length>k)
		     {
		       printf("SOMEWHERE ERROR HAS OCCURED\n");
		       exit(0);
		     }
		   
		 }
	       //printf("The value of length to be returned from this leaf node is %d\n",length);
	     }
	   return length;
	 }

       else
	 {
	   //this is not a root node. Hence find the distance to the bounding boxes
	   
	   double min_distance_to_left_child=root->left()->bound().MinDistanceSq(point);
	   double min_distance_to_right_child=root->right()->bound().MinDistanceSq(point);
	   
	   Tree *farther;
	   
	   if(min_distance_to_left_child < min_distance_to_right_child)
	     {
	       //Recursively explore the left child
	       farther=root->right();
	       length=FindKNearestNeighbours(root->left(),point,r_matrix,str,length,k);
	     }
	   else
	     {
	       //Recursively explore the right child
	       farther=root->left();
	       length=FindKNearestNeighbours(root->right(),point,r_matrix,str,length,k);
	     
	       
	     }
	   
	   //If number of neighbours found are less than k then go ahead and explore the other half too
	   
	   if(length<k)
	     {
	       //recursively explore the farther child
	      
	       length=FindKNearestNeighbours(farther,point,r_matrix,str,length,k);
	      
	     }
	   else
	     {
	       //check the other half only if the kth nn distance is greater than the distance of the point form the farther bounding box
	      
	       if(str[k-1].get_distance()> farther->bound().MinDistanceSq(point))
		 {
		   length=FindKNearestNeighbours(farther,point,r_matrix,str,length,k);
		 }
	     }
	   return length;
	 }
     }

   //The function sto follow are friend functions. they are also used by the dual tree algorithms

friend int find_index(ArrayList<SingleTreeResults> str,double dist,int start,int end)
{
 
  //this means that there are no elements in the array
  if(start>end)
    return 0;

  //this means there is exactly 1 element in the array
  if(start==end)
    {
     
      if(dist>str[start].get_distance())
	{
	  // printf("will return %d\n",end+1);
	  return end+1; //return end if the element to be added is at the end of the list
	}
      else return start;
    }

  //find where the element will be in the sorted array. This is just the binary search

  if(dist==str[start+end/2].get_distance()) 
    return start+end/2;

  else
    {
     
      if(dist<str[(end+start)/2].get_distance())
	{
	  //go left
	  return find_index(str,dist,start,(start+end)/2);
	}
      else
	{
	      //go right
	  return find_index(str,dist,(start+end)/2+1,end);
	}
    }
}

friend int push_into_array(ArrayList<SingleTreeResults> &str,int position, double dist,int length, int index,int k)
{
  // printf("Length is %d\n",length);
  //printf("Index to be inserted is %d\n",index);
  //printf("Distance to be inserted is %f\n",dist);
  //printf("The position is %d\n",position);
  if(position==length) //that means add the element to the end of list
    {
      if(length==k)
	{
	  //cannot push into array
	  return 0;
	}
      else
	{

	  //add it to the end of the array
	  str[length].set_result(dist,index);
	  return 1;
	}
    }

  ArrayList <SingleTreeResults> temp;
  temp.Init(k);
   
  //the element will be added in the middle of the array
 

  for(int j=0;j<position;j++)
    temp[j].set_result(str[j].get_distance(),str[j].get_index());

  
  temp[position].set_result(dist,index);
  

  if(length==k)
    {
      for(int t=length-1;t>position;t--)
	temp[t].set_result(str[t-1].get_distance(),str[t-1].get_index());

      for(int j=0;j<length;j++)
	str[j].set_result(temp[j].get_distance(),temp[j].get_index());
      // delete(temp);
	return 0;
    }
  else
    {
     
    
      for(int t=length;t>position;t-=1)
	{
	  temp[t].set_result(str[t-1].get_distance(),str[t-1].get_index());
	}
      
    
      for(int j=0;j<length+1;j++)
	{
	  str[j].set_result(temp[j].get_distance(),temp[j].get_index());
	}
      return 1;
    }
}
};

//Definition of AllKNNSingleTree ends................................................................................



//Definition of AllKNNDualTreeResults begins................................................................................

class AllKNNDualTreeResults //This is just the same as AllKNNSingleTreeResults
{
 public:
   ArrayList<ArrayList<SingleTreeResults> > astr; 
};

class AllKNNDualTree
{
 public:
  AllKNNDualTreeResults results_matrix; 
  AllKNNDualTree() 
    {
    
    }
  
  //Friend Functions............................

  bool check_if_equal(double *arr1, double *arr2, int length);

  

  //This is the function which will perform the actual dual tree algorithm and spit the results 

  void ComputeAllKNNDualTree(Matrix *q_matrix,Matrix *r_matrix,int k) 
    {
        
      // Build the kdtree form the reference matrix and the query matrix
      using namespace tree;
      
      Tree *r_tree=tree::MakeKdTreeMidpoint<Tree>(*r_matrix,3,NULL,NULL); //here the leaf size is set to 3
      printf("ref tree built\n"); 
      
      Tree *q_tree=tree::MakeKdTreeMidpoint<Tree>(*r_matrix,3,NULL,NULL); //here the leaf size is set to 3
      printf("query tree built\n");
      
     
      //Inialize the 2-D array

      results_matrix.astr.Init(r_matrix->n_cols()); //Initialize the array list to initially contain the number of elements passed on as the parameter
     
      for(int i=0;i<r_matrix->n_cols();i++) //initalize each column of the arraylist
	results_matrix.astr[i].Init();

      FindKNearestNeighboursDualTree(q_tree,r_tree,q_matrix,r_matrix,k);
     
      /* for(int l=0;l<r_matrix->n_cols();l++)
	{
	  for(int t=0;t<k;t++)
	    {
	      printf("distance is %f ",results_matrix.astr[l][t].get_distance());
	    }
	  printf("\n");
	  }*/
      //****************************************************************
    }
  


double  FindKNearestNeighboursDualTree(Tree *q_tree,Tree *r_tree,Matrix *q_matrix,Matrix *r_matrix,int k) //A 1-D matrix of the k nearest neighbours is returned
 {

 //if distance between the two boxes is larger than the max _distance then return
  double distance_between_boxes=q_tree->bound().MinDistanceSq (r_tree->bound()); //base case
  if(q_tree->is_leaf()&& r_tree->is_leaf())
    {
      int start,end;

      //check if pruneable
      if(q_tree->stat().get_maximum_distance() < distance_between_boxes)
	{
	  //then there is no need to go further and hence we can return
	  return MAX_DOUBLE;
	}
      else
	{
	  //not purneable. therefore carry out exhaustive point-to-point computations

	  double max_dist=0.0;
	  double distance;
	  int position;
	  int count=0;

	  for(int i=q_tree->begin();i<q_tree->end();i++)
	    {
	     
	      count=0;
	      for(int j=r_tree->begin();j<r_tree->end();j++)
		{
		  if(!check_if_equal(q_matrix->GetColumnPtr(i),r_matrix->GetColumnPtr(j),q_matrix->n_rows())) //to make sure that we are not comparing the same set of points
		    {
		      
		      distance=la::DistanceSqEuclidean (r_matrix->n_rows(),q_matrix->GetColumnPtr(i),r_matrix->GetColumnPtr(j));
		      //we would like to find index of this point into str. this will enter into str[i]. This function takes in as argument an array list of single tree results            
		     

		      if(results_matrix.astr[i].size()==0)  
			{
			  //initialize start and end
			 
			  start=0;
			  end=-1;
			}

		      int length=results_matrix.astr[i].size(); //this calculates the old length		     
		      
		      position=find_index(results_matrix.astr[i],distance,start,results_matrix.astr[i].size()-1);
		      if(length<k)
			{
			 
			  //increase the length of the arraylist, only if the number of elements as of now are lesser than k
			  results_matrix.astr[i].AddBack(1); 
			 
			}
		      //Note the length is still the old length, the one that has been claulated in the step above. So it is 1 less than the actual length
		      push_into_array(results_matrix.astr[i],position,distance,length,j,k);
		      count++;
		     
		    }
		
		}

	      if(results_matrix.astr[i].size()<k)
		
		{
		  max_dist=MAX_DOUBLE;
		}
	      else 
		{
		  //all k nn have been found
		  max_dist=max_dist > results_matrix.astr[i][k-1].get_distance()?max_dist:results_matrix.astr[i][k-1].get_distance(); 
	
		}

	      for(int z=0;z<results_matrix.astr[i].size();z++)
		{
		  printf("distance=%f and index=%d\n",results_matrix.astr[i][z].get_distance(),results_matrix.astr[i][z].get_index());
		}
	 

	      return max_dist;
	    }
	}
    }
  
  else
    {
      //Not base case. Check if one can Prune
      if(q_tree->stat().get_maximum_distance() < distance_between_boxes)
	{
	  //then there is no need to go further and henraghavce we can return
	  return MAX_DOUBLE;
	}

      //NOT PRUNEABLE
      //both are not leafs
      if(!q_tree->is_leaf() && !r_tree->is_leaf())
	{
	 
	  double max_dist_q_left_r_left=FindKNearestNeighboursDualTree(q_tree->left(),r_tree->left(),q_matrix,r_matrix,k);
	  
	 
	  
	  double max_dist_q_left_r_right=FindKNearestNeighboursDualTree(q_tree->left(),r_tree->right(),q_matrix,r_matrix,k);

	  double max_dist_q_left= max_dist_q_left_r_left< max_dist_q_left_r_right? max_dist_q_left_r_left: max_dist_q_left_r_right;//take minimum

	 
	  double max_dist_q_right_r_left=FindKNearestNeighboursDualTree(q_tree->right(),r_tree->left(),q_matrix,r_matrix,k);
	 
	  double max_dist_q_right_r_right=FindKNearestNeighboursDualTree(q_tree->right(),r_tree->right(),q_matrix,r_matrix,k);

	  double max_dist_q_right= max_dist_q_right_r_left< max_dist_q_right_r_right? max_dist_q_right_r_left: max_dist_q_right_r_right;
	 
	  double max_dist_q=max_dist_q_left>max_dist_q_right?max_dist_q_left:max_dist_q_right;
	  q_tree->stat().set_maximum_distance(max_dist_q);
	  return q_tree->stat().get_maximum_distance();
	}

      else
	{
	 
	  //q_tree is leaf and r_tree is not
	  if(q_tree->is_leaf()&&!r_tree->is_leaf())
	    {
	 
	      double max_dist_q_r_left=FindKNearestNeighboursDualTree(q_tree,r_tree->left(),q_matrix,r_matrix,k);
	      double max_dist_q_r_right=FindKNearestNeighboursDualTree(q_tree,r_tree->right(),q_matrix,r_matrix,k);
	      double max_dist_q = max_dist_q_r_left > max_dist_q_r_right?max_dist_q_r_right:max_dist_q_r_left; //take minimum
	      q_tree->stat().set_maximum_distance(max_dist_q);
	      return q_tree->stat().get_maximum_distance();
	    }

	  else
	    {
	      
	      //q_tree is not a leaf and r_tree is

	      if(!q_tree->is_leaf()&&r_tree->is_leaf())
		{
		  
		  double max_dist_q_left_r=FindKNearestNeighboursDualTree(q_tree->left(),r_tree,q_matrix,r_matrix,k);
		  double max_dist_q_right_r=FindKNearestNeighboursDualTree(q_tree->right(),r_tree,q_matrix,r_matrix,k);
		  double max_dist_q= max_dist_q_left_r > max_dist_q_right_r? max_dist_q_left_r: max_dist_q_right_r;
		  q_tree->stat().set_maximum_distance(max_dist_q);
		  return max_dist_q;
		}
	    }
	}
    }
 }
};

#endif
