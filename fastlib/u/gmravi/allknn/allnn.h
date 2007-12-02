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
SingleTreeResults* FindNearestNeighbour(Tree*,double*,Matrix*,SingleTreeResults*);
};

bool check_if_equal(ArrayList <double>, ArrayList <double>);


class AllKNNSingleTreeResults
{
 public:
   ArrayList<ArrayList<SingleTreeResults> > astr; 

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
  friend bool same_points(double *point1,double *point2, int len);
  //This is the function which will perform the actual single tree algorithm and spew the results 
  
  void ComputeAllKNNSingleTree(Matrix*,Matrix*,int);  // A 2-D matrix is returned
  int  FindKNearestNeighbours(Tree*,double*,Matrix*,ArrayList<SingleTreeResults>&,int,int); //A 1-D matrix of the k nearest neighbours is returned


};

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
  friend bool same_points(double*, double*);
  //This is the function which will perform the actual dual tree algorithm and spew the results 

  void ComputeAllKNNDualTree(Matrix*,Matrix*,int);  // A 2-D matrix is returned
  double  FindKNearestNeighboursDualTree(Tree *q_tree,Tree *r_tree,Matrix*,Matrix*,int k); //A 1-D matrix of the k nearest neighbours is returned
};



