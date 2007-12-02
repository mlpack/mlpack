
#include "allnn.h"

bool check_if_equal(ArrayList <double> arr1, ArrayList <double> arr2)
{
  int i=0;
  for(i=0;i<arr1.size();i++)
    {
      if(arr1[i]!=arr2[i])
	return false;
    }
  return true;
} 
void AllNNNaive::Compute(Matrix *q_matrix, Matrix *r_matrix)
{
  //for each column vector of q_matrix find the nearest neighbour in r_matrix
 
  // ArrayList <double> arr1;
  //ArrayList <double> arr2;
 int cols_q;
 int cols_r;
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

void AllNNNaive::PrintResults(Matrix *q_matrix,Matrix *r_matrix)
{
  FILE *fp;
  fp=fopen("naive_nearest_neighbour.txt","w");
  printf("The nearest neighbours are as follows\n");
  // printf("The number of results are\n");
  //printf("%d\n",results.size());
  for(int i=0;i< results.size();i++)
    {
      fprintf(fp,"Point:%d\tDistance:%f\n", i,la::DistanceSqEuclidean(r_matrix->n_rows(),r_matrix->GetColumnPtr(results[i]),q_matrix->GetColumnPtr(i)));
      //fprintf(fp,"\n");
      
    }
}

void print(double *a,int len)
{
  for(int i=0;i<len;i++)
    printf("%f ",a[i]);
}

SingleTreeResults *AllNNSingleTree::FindNearestNeighbour(Tree *root,double *point, Matrix *r_matrix,SingleTreeResults *nearest_nn)
{
  //check if it is the leaf 
  if(root->is_leaf())
    {
      //find out the minimum distance by querying all points
      double min_dist_for_this_node=32768.0;
      SingleTreeResults *potential_nn;
      potential_nn=new SingleTreeResults();
      potential_nn->set_result(32768.0,-1); 
      //printf("root->begin is %d\n",root->begin());
      //printf("root->end is %d\n",root->end());
      //printf("The distance is being computed between the following points\n");
      for(int i=root->begin();i<root->end();i++)
	{
	  //print(r_matrix->GetColumnPtr(i),r_matrix->n_rows());
	  //print(point,r_matrix->n_rows());
	  double temp_dist=la::DistanceSqEuclidean(r_matrix->n_rows(),r_matrix->GetColumnPtr(i),point);
	  //printf("Temp_dist is %f\n",temp_dist);
	  //printf("min_dist_for_this_node is %f\n",min_dist_for_this_node);
	  if(temp_dist < min_dist_for_this_node && temp_dist!=0) //the condition temp_dist!=0 avoids duplicate points
	    {
	      //  printf("Entered here with temp_dist as %f\n",temp_dist);
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
      //printf("Minimum distance to left bb is %f\n",min_distance_to_left_child);
      //printf("minimum distance to right bb %f\n",min_distance_to_right_child);
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


void AllNNSingleTree::ComputeAllNNSingleTree(Matrix* q_matrix, Matrix  *r_matrix)
{
  // expand the size of results. At the moment it is initalized but size=0
  str_.GrowTo(q_matrix->n_cols()); //the number of query points

  // Build the kdtree form the reference matrix.
  using namespace tree;
  
		  
  Tree *root=tree::MakeKdTreeMidpoint<Tree>(*r_matrix,3,NULL,NULL); //here the leaf size is set to 1

 
  double max_dist_sq=32768.0;  

  //For each point find out the nearest neighbour
  for(int i=0;i<q_matrix->n_cols();i++)
  {
    SingleTreeResults *nearest_nn=new SingleTreeResults();
    str_[i]=*FindNearestNeighbour(root,q_matrix->GetColumnPtr(i),r_matrix,nearest_nn);
    printf("Distance to the nearest neighbour is %f\n",str_[i].get_distance());
    
  }
  //return str_;
}

int find_index(ArrayList<SingleTreeResults> str,double dist,int start,int end)
{
  printf("Start is %d and end is %d\n",start,end);
  printf("Searching for distance %f\n",dist);
  printf("In function find index str is...\n");
  for(int j=start;j<=end;j++)
    {
      printf("distance is %f\n",str[j].get_distance());
    }
  if(start>end)
    return 0;
  if(start==end)
    {
      printf("str[start].get_distance() is %f",str[start].get_distance());
      printf("dist is %f\n",dist);
      if(dist>str[start].get_distance())
	{
	    printf("will return %d\n",end+1);
	  return end+1; //returned if the element to be added is at the end of the list
	}
      else return start;
    }
  //find where the element will be in the sorted array
  if(dist==str[start+end/2].get_distance()) 
    return start+end/2;
  else
    {
      printf("distance of pivot:%f\n",str[(end+start)/2].get_distance());
      if(dist<str[(end+start)/2].get_distance())
	{
	  //go left
	  printf("left\n");
	  return find_index(str,dist,start,(start+end)/2);
	}
      else
	{
	      //go right
	  printf("right\n");
	  printf("New start is %d and new end is %d\n",(start+end)/2+1,end);
	  return find_index(str,dist,(start+end)/2+1,end);
	}
    }
}

int push_into_array(ArrayList<SingleTreeResults> &str,int position, double dist,int length, int index,int k)
{
  printf("Length is %d\n",length);
  printf("Index to be inserted is %d\n",index);
  printf("Distance to be inserted is %f\n",dist);
  printf("The position is %d\n",position);
  if(position==length) //that means add the element to the end of list
    {
      if(length==k)
	{
	  //cannot push into array
	  for(int l=0;l<length;l++)
	printf("distance:%f and index:%d\n",str[l].get_distance(),str[l].get_index());
	  return 0;
	}
      else
	{

	  //add it to the end of the array
	  str[length].set_result(dist,index);
	  for(int l=0;l<length+1;l++)
	    printf("distance:%f and index:%d\n",str[l].get_distance(),str[l].get_index());


 
	  return 1;
	}
    }

  ArrayList <SingleTreeResults> temp;
  temp.Init(k);
   printf("temp created\n");
  //the element will be added in the middle of the array
  printf("The element will be added to the middle of the array\n");

  for(int j=0;j<position;j++)
    temp[j].set_result(str[j].get_distance(),str[j].get_index());

  printf("Distance is %f\n",dist);
  printf("index is %d\n",index);
  temp[position].set_result(dist,index);
  printf("Length is %d and k=%d\n",length,k);
  printf("temp at position is set up\n");

  if(length==k)
    {
      for(int t=length-1;t>position;t--)
	temp[t].set_result(str[t-1].get_distance(),str[t-1].get_index());

      for(int j=0;j<length;j++)
	str[j].set_result(temp[j].get_distance(),temp[j].get_index());
      // delete(temp);
      printf("Before returning i have str as\n");
      for(int l=0;l<length;l++)
	printf("distance:%f and index:%d\n",str[l].get_distance(),str[l].get_index());
     
	return 0;
    }
  else
    {
      printf("since length is not equal to k\n");

    
      for(int t=length;t>position;t-=1)
	{
	  
	  printf("came here\n");
	  printf("Length is %d\n",length);
	  printf("position is %d\n",position);
	  printf("will copy to temp\n");
	  printf("t is %d\n",t);
	  printf("distance is %f\n",str[t-1].get_distance());
	  printf("index is %d\n",str[t-1].get_index());
	  temp[t].set_result(str[t-1].get_distance(),str[t-1].get_index());
	  printf("Set up..\n");
	  printf("t is %d\n",t);
	  printf("wi.ll loop...\n");
	}
      
      printf("adjusrwrewerwerwerwerwe.....");
      for(int j=0;j<length+1;j++)
	{
	  printf("copying element by element...\n");
	  printf("woll copy distance %f\n",temp[j].get_distance());
	  printf("Will copy index =%d\n",temp[j].get_index());
	  printf("j is %d\n",j);
	 
	  str[j].set_result(temp[j].get_distance(),temp[j].get_index());
	  printf("copied....\n");
	}
      printf("copied\n");
      // delete(temp);
      // printf("deleted temp");
      printf("str before leaving is..\n");
      for(int l=0;l<length+1;l++)
	printf("distance:%f and index:%d\n",str[l].get_distance(),str[l].get_index());


      return 1;
    }
}

bool same_points(double *point1,double *point2,int len)
{
  for(int i=0;i<len;i++)
    {
      //printf("point1[i]=%f\n",point1[i]);
      //printf("point2[i] is %f\n",point2[i]);
      if(point1[i]!=point2[i])
	{
	  //printf("returning false\n");
	  return false;
	}
    }
  //printf("Returning true\n");
  return true;
}

int AllKNNSingleTree::FindKNearestNeighbours(Tree *root, double *point,Matrix *r_matrix,ArrayList<SingleTreeResults> &str,int length,int k)
{
  //Base case is that the node is a leaf 

  if(root->is_leaf())
    {
      printf("Will explore leaf\n");
      int position;
      //check for number of elements
      double dist;
      int end;
      if(length==0) 
	end=-1;
      else
	end=length-1;
	  
      for(int i=root->begin();i<root->end();i++)
	{
	  //one very important check is to see that query point is not the same point as any other point beingcompared
	  if(!same_points(point,r_matrix->GetColumnPtr(i),r_matrix->n_rows()))
	    { 
	      dist=la::DistanceSqEuclidean(r_matrix->n_rows(),point,r_matrix->GetColumnPtr(i));
	      printf("Distance is %f\n",dist);
	      //find where the new element should be pushed
	      int start=0;
	      printf("Before going to funcction find index ..\n");
	      for(int l=0;l<length;l++)
		printf("distance is %f\n",str[l].get_distance());
	      position=find_index(str,dist,start,end); 
	      printf("The position where this willbe inserted is \n");
	      printf("%d\n",position);
	      length+=push_into_array(str,position,dist,length,i,k);
	      end=length-1;
	      printf("After pushAFTWER PUSH AFTER PUSH...\n");

	      for(int count=0;count<length;count++)
		printf("distance:%f\n",str[count].get_distance());

	      //printf("length is now %d\n",length);
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
	  //printf("will explore left child\n");
	  farther=root->right();
	  length=FindKNearestNeighbours(root->left(),point,r_matrix,str,length,k);
	  //printf("length:%d\n",length);
	}
      else
	{
	  //Recursively explore the right child
	  // printf("Will explore the right child\n");
	  farther=root->left();
	  length=FindKNearestNeighbours(root->right(),point,r_matrix,str,length,k);
	  // printf("length:%d\n",length);
	  
	}

      //If number of neighbours found are less than k then go ahead and explore the other half too
   
      if(length<k)
	{
	  //recursively explore the farther child
	  //printf("Will explore the other half as the knn bound is still infinity\n");
	  // printf("The root now is %X\n",farther);
	  length=FindKNearestNeighbours(farther,point,r_matrix,str,length,k);
	  //printf("Length is %d\n",length);
	}
      else
	{
	  //check the other half only if the kth nn distance is greater than the distance of the point form the farther bounding box
	  //printf("Length is k=%d\n",k);
	  //printf("will explore the other half if str[k-1].distance=%f\n",str[k-1].get_distance());
	  //printf("distance of the farther bb is %f\n",farther->bound().MinDistanceSq(point));
	  if(str[k-1].get_distance()> farther->bound().MinDistanceSq(point))
	    {
	      //  printf("will explore the other half because of the promise of a closer knn\n");
	      length=FindKNearestNeighbours(farther,point,r_matrix,str,length,k);
	      //printf("length is %d\n",length);
	    }
	}
      return length;
    }
}

 void AllKNNSingleTree::ComputeAllKNNSingleTree(Matrix *q_matrix,Matrix *r_matrix,int k)
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

  ArrayList<SingleTreeResults> str;
  str.Init(k);

  //printf("Will start knn calculations\n"); 
 
  // printf("%d\n",r_matrix->n_cols());

  for(int i=0;i<r_matrix->n_cols();i++)
    {   
      int length=0;
      
      length+=FindKNearestNeighbours(root,q_matrix->GetColumnPtr(i),r_matrix,str,length,k);
 
      printf("The nearest neighbour distances are\n");

      for(int l=0;l<k;l++)
	printf("%f\n",str[l].get_distance());
    

      for(int j=0;j<length;j++)
	{
	  //printf("Will add distance=%f\n",str[j].get_distance());
	  results_matrix.astr[i][j].set_result(str[j].get_distance(),str[j].get_index());
	}

    

       printf("The origianl point is\n");

      for(int j=0;j<r_matrix->n_rows();j++)
       printf("%f\n",q_matrix->GetColumnPtr(i)[j]);

    
      printf("Will clear str now........\n");
      printf("Length is %d\n",length);
     
    }
}

double AllKNNDualTree::FindKNearestNeighboursDualTree(Tree *q_tree,Tree *r_tree,Matrix *q_matrix,Matrix *r_matrix,int k)
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
	  return 32768.0;
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
	      printf("for this point count is %d\n",count);
	      printf("TAKING NEW I=%d............................................................\n",i);
	     
	      count=0;
	      for(int j=r_tree->begin();j<r_tree->end();j++)
		{
		  if(!same_points(q_matrix->GetColumnPtr(i),r_matrix->GetColumnPtr(j),q_matrix->n_rows()))
		    {
		      
		      distance=la::DistanceSqEuclidean (r_matrix->n_rows(),q_matrix->GetColumnPtr(i),r_matrix->GetColumnPtr(j));
		      //we would like to find index of this point into str. this will enter into str[i]. This function takes in as argument an array list of single tree results            
		      printf("distance is %f\n",distance);
		      printf("size is %d\n",results_matrix.astr[i].size());
		      if(results_matrix.astr[i].size()==0)  
			{
			  //initialize start and end
			  printf("Initialized start and end\n");
			  start=0;
			  end=-1;
			}

		      int length=results_matrix.astr[i].size(); //this calculates the old length
		      printf("at the moment length is %d\n",length);
		      
		      position=find_index(results_matrix.astr[i],distance,start,results_matrix.astr[i].size()-1);
		      if(length<k)
			{
			  printf("since the size of array is still lesser than k will increase the size\n");
			  printf("Before expanding length is %d \n",length);
			  results_matrix.astr[i].AddBack(1); //increase the length of the arraylist, only if the number of elements as of now are lesser than k
			  printf("new length after expanding is %d\n",results_matrix.astr[i].size());
			}

		      push_into_array(results_matrix.astr[i],position,distance,length,j,k);
		      printf("pushed into array..\n");
		      count++;
		     
		    }
		
		}

	      if(results_matrix.astr[i].size()<k)
		max_dist=32768.0;
	      else //all k nn have been found
		max_dist=max_dist > results_matrix.astr[i][k-1].get_distance()?max_dist:results_matrix.astr[i][k-1].get_distance(); //this is an update after every point IN THE QUERY NODE
	      
	      for(int z=0;z<results_matrix.astr[i].size();z++)
		printf("distance=%f and index=%d\n",results_matrix.astr[i][z].get_distance(),results_matrix.astr[i][z].get_index());
	    }
	  printf("Will return from this node a value of %f%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n",max_dist);

	  return max_dist;
	}
    }

  else
    {
      //Not base case. PRUNING CASE
      if(q_tree->stat().get_maximum_distance() < distance_between_boxes)
	{
	  //then there is no need to go further and henraghavce we can return
	  return 32768.0;
	}

      //NOT PRUNEABLE
      //both are not leafs
      if(!q_tree->is_leaf() && !r_tree->is_leaf())
	{
	  printf("both are not leafs......\n");
	  double max_dist_q_left_r_left=FindKNearestNeighboursDualTree(q_tree->left(),r_tree->left(),q_matrix,r_matrix,k);
	  
	  /*  if(max_dist < q_tree->stat().get_maximum_distance())
	    {
	      q_tree->stat().set_maximum_distance(max_dist);
	      }*/
	  
	  double max_dist_q_left_r_right=FindKNearestNeighboursDualTree(q_tree->left(),r_tree->right(),q_matrix,r_matrix,k);

	  double max_dist_q_left= max_dist_q_left_r_left< max_dist_q_left_r_right? max_dist_q_left_r_left: max_dist_q_left_r_right;//take minimum

	  /* if(max_dist_q_left< q_tree->stat().get_maximum_distance())
	    {
	      q_tree->stat().set_maximum_distance(max_dist);
	      }*/

	  double max_dist_q_right_r_left=FindKNearestNeighboursDualTree(q_tree->right(),r_tree->left(),q_matrix,r_matrix,k);
	  /*if(max_dist<q_tree->stat().get_maximum_distance())
	    {
	      q_tree->stat().set_maximum_distance(max_dist);
	      }*/

	  double max_dist_q_right_r_right=FindKNearestNeighboursDualTree(q_tree->right(),r_tree->right(),q_matrix,r_matrix,k);

	  double max_dist_q_right= max_dist_q_right_r_left< max_dist_q_right_r_right? max_dist_q_right_r_left: max_dist_q_right_r_right;
	 
	  double max_dist_q=max_dist_q_left>max_dist_q_right?max_dist_q_left:max_dist_q_right;
	  q_tree->stat().set_maximum_distance(max_dist_q);
	  return q_tree->stat().get_maximum_distance();
	}
      else
	{
	  printf("q is leaf and r is not...\n");
	  //q_tree is leaf and r_tree is not
	  if(q_tree->is_leaf()&&!r_tree->is_leaf())
	    {
	 
	      double max_dist_q_r_left=FindKNearestNeighboursDualTree(q_tree,r_tree->left(),q_matrix,r_matrix,k);
	      
	      /*if(max_dist < q_tree->stat().get_maximum_distance())
		{
		  q_tree->stat().set_maximum_distance(max_dist);
		  }*/
	      
	      double max_dist_q_r_right=FindKNearestNeighboursDualTree(q_tree,r_tree->right(),q_matrix,r_matrix,k);
	      
	      /*if(max_dist<q_tree->stat().get_maximum_distance())
		{
		   q_tree->stat().set_maximum_distance(max_dist);
		  }*/
	      
	      double max_dist_q = max_dist_q_r_left > max_dist_q_r_right?max_dist_q_r_right:max_dist_q_r_left; //take minimum
	      q_tree->stat().set_maximum_distance(max_dist_q);
	      return q_tree->stat().get_maximum_distance();
	    }

	  else
	    {
	      printf("q is not a leaf and r is \n");
	      //q_tree is not a leaf and r_tree is

	      if(!q_tree->is_leaf()&&r_tree->is_leaf())
		{
		  
		  double max_dist_q_left_r=FindKNearestNeighboursDualTree(q_tree->left(),r_tree,q_matrix,r_matrix,k);
		  //printf("After getting back to the function the value is %f\n",max_dist);
		  printf("will compare this vlaue with %f\n",q_tree->stat().get_maximum_distance());
		  /* if(max_dist < q_tree->stat().get_maximum_distance())
		    {
		      q_tree->stat().set_maximum_distance(max_dist);
		      }*/
		  
		  double max_dist_q_right_r=FindKNearestNeighboursDualTree(q_tree->right(),r_tree,q_matrix,r_matrix,k);
		  
		  double max_dist_q= max_dist_q_left_r > max_dist_q_right_r? max_dist_q_left_r: max_dist_q_right_r;
		  q_tree->stat().set_maximum_distance(max_dist_q);
		  return max_dist_q;
		}
	    }
	}
    }
}
  
void AllKNNDualTree::ComputeAllKNNDualTree(Matrix *q_matrix,Matrix *r_matrix,int k)
{

 //printf("results_matrix initialized\n");
 // Build the kdtree form the reference matrix and the query matrix
  using namespace tree;
 
 Tree *r_tree=tree::MakeKdTreeMidpoint<Tree>(*r_matrix,3,NULL,NULL); //here the leaf size is set to 3
 printf("ref tree built\n"); 
 
 Tree *q_tree=tree::MakeKdTreeMidpoint<Tree>(*r_matrix,3,NULL,NULL); //here the leaf size is set to 3
 printf("query tree built\n");

 printf("Trees built\n");
 //****************************************************************
 results_matrix.astr.Init(r_matrix->n_cols()); //the arraylist ahs been completely initialized
 printf("size has been set to %d\n",r_matrix->n_cols());
  for(int i=0;i<r_matrix->n_cols();i++)
    results_matrix.astr[i].Init();
  FindKNearestNeighboursDualTree(q_tree,r_tree,q_matrix,r_matrix,k);
  printf("completed the algo. Printing results\n");
  printf("the k nearest neighbours are............\n");
  for(int l=0;l<r_matrix->n_cols();l++)
    {
      for(int t=0;t<k;t++)
       {printf("distance is %f ",results_matrix.astr[l][t].get_distance());}
      printf("\n");
    }
  //****************************************************************
}


int main(int argc, char *argv[])
{ 
  fx_init(argc, argv);
  char *query_file= new char(40);
  char *ref_file= new char(40);
  char *method=new char(40);
  Matrix *q_matrix=new Matrix();
  Matrix *r_matrix=new Matrix();
  //Initialize q_matrix and r_matrix


  //get the command line arguments and load the matrix with the query and reference values

 //Dont forget to make checks if the file is given or not
  //printf("Till here. Will now get the names of the quer_files and ref_files\n");
  //strcpy(query_file,fx_param_str_req(NULL,"query_file"));
  //strcpy(ref_file,fx_param_str_req(NULL,"ref_file"));
  //strcpy(method,fx_param_str_req(NULL,"method"));

  strcpy(query_file,"dummy.ds");
  strcpy(ref_file,"dummy.ds");


  int k=3;
  strcpy(method,"allknnsingletree");
  //Load matrices q_matrix and r_matrix

  data::Load(query_file,q_matrix);
  data::Load(ref_file,r_matrix);
  printf("Method is %s\n",method);
  printf("Done till here.Both matrices are loaded\n");
  printf("query_file is %s\n",query_file);
  printf("ref_file is %s\n",ref_file);

  //if method is allnnnaive. AllNNNaive computations follow
  //printf("String compare with allknnsingletree is %d\n",strcmp(method,"allknnsingletree"));
   if(strcmp(method,"allnnnaive")==0)
  {
    printf("In allnaive");
    AllNNNaive naive;
    naive.Compute(q_matrix,r_matrix);
    naive.PrintResults(q_matrix,r_matrix);
  }
    
  if(strcmp(method,"allnnsingletree")==0)
  {
    //AllNNSingleTree computations

    AllNNSingleTree *ast=new AllNNSingleTree();
 
    ast->ComputeAllNNSingleTree(q_matrix,r_matrix);
    FILE *fp;
    fp=fopen("nearest_neighbour_singletree.txt","w");

  //print the results

    for(int i=0;i<r_matrix->n_cols();i++)
      fprintf(fp,"Point:%d\tDistance:%f\n",i,ast->str_[i].get_distance());
  } 
  if(strcmp(method,"allknnsingletree")==0)
    {
	 printf("in allknnsingletree\n");
	 //AllKNNSingle tree computations

	 //int k=fx_param_int_req(NULL,"k");

	 AllKNNSingleTree *akst= new AllKNNSingleTree();
	 akst->ComputeAllKNNSingleTree(q_matrix,r_matrix,k);
    }

  /*
     if(strcmp(method,"allknndualtree")==0)
       {
	 printf("in allknndualtree\n");
	 //AllKNNSDual tree computations
	 //int k=fx_param_int_req(NULL,"k");
	 AllKNNDualTree *akdt= new AllKNNDualTree();
	 akdt->ComputeAllKNNDualTree(q_matrix,r_matrix,k);
	 }*/
     fx_done();     
}
