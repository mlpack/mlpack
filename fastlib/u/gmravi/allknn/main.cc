
#include "allnn.h"

bool check_if_equal(ArrayList <double> arr1, ArrayList <double> arr2)
{
  int i=0;
  for(i=0;i<arr1.size();i++)
    {
      if(arr1[i]!=arr2[i])
	return 0;
    }
  return 1;
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

int find_index(SingleTreeResults *str,double dist,int start,int end)
{
  printf("Start is %d and end is %d\n",start,end);
  if(start>end)
    return 0;
  if(start==end)
    {
      //printf("str[start].get_distance() is %f",str[start].get_distance());
      //printf("dist is %f\n",dist);
      if(dist>str[start].get_distance())
	{
	  //  printf("will return %d\n",end+1);
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
	  //printf("left\n");
	  return find_index(str,dist,start,(start+end)/2);
	}
      else
	{
	      //go right
	  // printf("right\n");
	  //printf("New start is %d and new end is %d\n",(start+end)/2+1,end);
	  return find_index(str,dist,(start+end)/2+1,end);
	}
    }
}

int push_into_array(SingleTreeResults *str,int position, double dist,int length, int index,int k)
{
  //printf("Length is %d\n",length);
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
  // printf("temp created\n");
  //the element will be added in the middle of the array
  //printf("The element will be added to the middle of the array\n");

  for(int j=0;j<position;j++)
    temp[j].set_result(str[j].get_distance(),str[j].get_index());

  temp[position].set_result(dist,index);
  //printf("Length is %d and k=%d\n",length,k);

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
      for(int t=length;t>position;t--)
	temp[t].set_result(str[t-1].get_distance(),str[t-1].get_index());
      //  printf("adjusted.....");
      for(int j=0;j<length+1;j++)
	str[j].set_result(temp[j].get_distance(),temp[j].get_index());
      printf("copied\n");
      // delete(temp);
      // printf("deleted temp");
      return 1;
    }
}
bool same_points(double *point1,double *point2, int len)
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

int AllKNNSingleTree::FindKNearestNeighbours(Tree *root, double *point,Matrix *r_matrix,SingleTreeResults *str,int length,int k)
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
	      //printf("Distance is %f\n",dist);
	      //find where the new element should be pushed
	      int start=0;
	      position=find_index(str,dist,start,end); 
	      //printf("The position where this willbe inserted is \n");
	      //printf("%d\n",position);
	      length+=push_into_array(str,position,dist,length,i,k);
	      end=length-1;
	      /*printf("After push...\n");

	      for(int count=0;count<length;count++)
		printf("distance:%f\n",str[count].get_distance());*/

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

  SingleTreeResults *str=new SingleTreeResults[k];

  //printf("Will start knn calculations\n"); 
 
  // printf("%d\n",r_matrix->n_cols());

  for(int i=0;i<r_matrix->n_cols();i++)
    {   
      int length=0;
      // printf("came here...\n");
      //printf("root=%X\n",root);
      //printf("the number of points covered are %d\n",i);
      //printf("Total number of points left to be covered are %d\n",r_matrix->n_cols()-i);
      length=FindKNearestNeighbours(root,q_matrix->GetColumnPtr(i),r_matrix,str,length,k);
 
      printf("The nearest neighbour distances are\n");

      for(int l=0;l<k;l++)
	printf("%f\n",str[l].get_distance());
    

      for(int j=0;j<length;j++)
	{
	  //printf("Will add distance=%f\n",str[j].get_distance());
	  results_matrix.astr[i][j].set_result(str[j].get_distance(),str[j].get_index());
	}

      /*  printf("knn for this point have been added...........\n");
    
      for(int j=0;j<length;j++)
	{
          printf("Index:%d distance: %f\n",str[j].get_index(),str[j].get_distance());
	  }*/

       printf("The origianl point is\n");

      for(int j=0;j<r_matrix->n_rows();j++)
       printf("%f\n",q_matrix->GetColumnPtr(i)[j]);

    
      printf("Will clear str now........\n");
      printf("Length is %d\n",length);
      /* for(int l=0;l<length;l++)
	 str[l].set_result(32768.0,-1);*/
    }
  printf("The knn distances are\n");
  printf("The number of points are %d\n",r_matrix->n_cols());
  char fname[40];
  strcpy(fname,"allknn");
  char query_file[40];
  strcat(fname,"_colors50k.ds");
  strcat(fname,query_file);
  printf("The name of query_file is %s\n",query_file);
  FILE *fp=fopen(fname,"w");
  for(int t=0;t<r_matrix->n_cols();t++)
    {
      for(int l=0;l<k;l++)
	fprintf(fp,"%f ",results_matrix.astr[t][l].get_distance());
      fprintf(fp,"\n");
    }
}


void test_module()
{
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

  strcpy(query_file,"colors50k.ds");
  strcpy(ref_file,"colors50k.ds");


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

     //the code that follows is just a test module
     if(strcmp(method,"test"))
       {
	 printf("came here");
	 test_module();
       }
     if(strcmp(method,"allnndual")==0)
       {

       }
     fx_done();     
}
