#include "square_tree.h"
#include "dual_tree_integrals.h"

class SquareTreeTester {

private:

  typedef SquareTree<IntegralTree, IntegralTree, SquareIntegralStat> 
          SqrIntegralTree;
          
  SqrIntegralTree* tree_;
  
  void Setup_() {
  
    tree_ = new SqrIntegralTree();
    
  
  }
  
  void Destruct_() {
  
  }

public:

  void TestAll() {
  
    Setup_();
    
    Destruct_();
  
  } // TestAll()

}; // class SquareTreeTester


int main(int argc, char* argv[]) {

  SquareTreeTester tester;
  tester.TestAll();

  return 0;

} // main()