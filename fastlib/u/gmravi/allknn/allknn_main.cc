#include "allknn.h"
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
  strcpy(method,"allknnsingletree");
  int k=3;
  //Load matrices q_matrix and r_matrix

  data::Load(query_file,q_matrix);
  data::Load(ref_file,r_matrix);
  printf("Method is %s\n",method);
  printf("Done till here.Both matrices are loaded\n");
  printf("query_file is %s\n",query_file);
  printf("ref_file is %s\n",ref_file);

  //if method is allnnnaive. AllNNNaive computations follow
  //printf("String compare with allknnsingletree is %d\n",strcmp(method,"allknnsingletree"));
   if(strcmp(method,"allknnsingletree")==0)
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
      fprintf(fp,"Point:%d\tDistance:%f\n",i,ast->str[i].get_distance());
  } 

  if(strcmp(method,"allknnsingletree")==0)
  {
  printf("in allknnsingletree\n");
  //AllKNNSingle tree computations
  
  int k=fx_param_int_req(NULL,"k");
  int k=3;
  
  AllKNNSingleTree *akst= new AllKNNSingleTree();
  akst->ComputeAllKNNSingleTree(q_matrix,r_matrix,k);
  }
   
  if(strcmp(method,"allknndualtree")==0)
    {
      printf("in allknndualtree\n");
      //AllKNNSDual tree computations
      int k=fx_param_int_req(NULL,"k");
      AllKNNDualTree *akdt= new AllKNNDualTree();
      akdt->ComputeAllKNNDualTree(q_matrix,r_matrix,k);
      }
  fx_done();
}


