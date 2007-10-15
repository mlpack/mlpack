#include "fastlib/fastlib_int.h"
class AllNNNaive
{
  
 public:
  ArrayList <int> results;
  AllNNNaive(){};
  void Compute(Matrix *q_matrix, Matrix *r_matrix);
  void PrintResults(Matrix*,Matrix*);
};

  
 class Stat 
{

  //This is justa dummy class
 public:
  void Init() {}
    
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

class SingleTreeResults
  {
    private: 
      double distance;
      int index;
   
  public:
    //setter function
    void set_result(double distance, int index )
      {
	//Here index is the index in r_matrix which is the potential nn for a given query point
	this->distance=distance;
	this->index=index;
      }
    //getter function
    double get_distance(){return distance;};
    int get_index()
      {
	return index;
      }
    SingleTreeResults()
      {
	this->distance=32768.0;
	this->index=-1;
      }
  };

typedef BinarySpaceTree<DHrectBound<2>, Matrix, Stat> Tree;

class AllNNSingleTree
{
public:
 ArrayList <SingleTreeResults> str_;
 //constructor
 //cosntructor definition
  AllNNSingleTree()
    {
      //Initialize the results list
      str_.Init();
    }

  //This is the function which will perform the actual single tree algorithm and spew the results 

void ComputeAllNNSingleTree(Matrix*, Matrix*);
SingleTreeResults* find_nearest_neighbour(Tree*,double*,Matrix*,SingleTreeResults*);
};

bool check_if_equal(ArrayList <double>, ArrayList <double>);


class AllKNNSingleTreeResults
{
 public:
   SingleTreeResults **astr; 

  AllKNNSingleTreeResults()
    {
      
    }
};

class AllKNNSingleTree
{
 public:
  AllKNNSingleTreeResults results_matrix;
  //constructor
  //cosntructor definition
  AllKNNSingleTree() 
    {
      //Initialize  results_matrix_ This will later be reinitialized to the number of columns in the function ComputeAllKNNSingleTree
      // results_matrix.astr.Init();
      printf("Initialized properly\n");
    }
  
  //This is the function which will perform the actual single tree algorithm and spew the results 
  
  void ComputeAllKNNSingleTree(Matrix*,Matrix*,int);  // A 2-D matrix is returned
  int  FindKNearestNeighbours(Tree*,double*,Matrix*,SingleTreeResults*,int,int); //A 1-D matrix of the k nearest neighbours is returned

};


