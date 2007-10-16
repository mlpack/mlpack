#ifndef EMST_TREE_H
#define EMST_TREE_H


const int metric = 2;


/********************** EmstTreeNode ***********************************/
class EmstTreeNode {
  
public:
  void Init(Matrix& data, int leaf_size);
  
  EmstTreeNode get_left();
  EmstTreeNode get_right();
  
  int numPoints();
  bool isLeaf();
  
  double minDist(EmstTreeNode& other);
  
private:
  DHrectBound<metric> bounding_box;
  EmstTreeNode left;
  EmstTreeNode right;
  int num_points;
  int num_dimensions;
  Matrix &data;

};

void EmstTreeNode::Init(Matrix& data, int leaf_size) {
  
  num_points = data.n_cols();
  num_dimensions = data.n_rows();
  
  if (num_points < leaf_size) {
  
    left = NULL;
    right = NULL;

    
    
  }
  else {
    
    /*
     1) split the data
     2) build the children
     3) expand the bounding box
     */
    
    Matrix& left_data = data;
    Matrix& right_data = data;
    
    left.Init(left_data, leaf_size);
    right.Init(right_data, leaf_size);
  }
  
  bounding_box.Init(num_dimensions);
  printf("bounding_box.dim() = %d\n", bounding_box.dim());
 
}


int EmstTreeNode::numPoints() {
  return num_points;
}

bool EmstTreeNode::isLeaf() {
  return(this.left==NULL);  
}

double EmstTreeNode::minDist(EmstTreeNode& other) {
  return 0.0;
}


/******************* EmstTree *********************************************/
class EmstTree {
  
public:
  void Init(Matrix data, int leaf_size);
  
  void findNeighbors();
  
  EmstTreeNode* Root();
  int numPoints();
  int numDimensions(); 
  
private:
  int num_points;
  int num_dimensions;
  EmstTreeNode root;
  
};

void EmstTree::Init(Matrix data, int leaf_size) {
  num_points = data.n_cols();
  num_dimensions = data.n_rows();
  //DEBUG_ASSERT_MSG(num_points > num_dimensions, "num_points = %d, num_dimensions = %d\n", num_points, num_dimensions);
  EmstTreeNode new_root;
  new_root.Init(data, leaf_size);
}

void EmstTree::findNeighbors() {
  DEBUG_GOT_HERE(0);
}

EmstTreeNode* EmstTree::Root() {
  return root;
}

int EmstTree::numPoints() { 
  return num_points;
}

int EmstTree::numDimensions() {
  return num_dimensions;
}




#endif
