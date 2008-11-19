#include "cosine_tree.h"
#include "quicsvd.h"

class CosineNodeTest {
  FILE * logfile;
public:
  CosineNodeTest() {
    logfile = fopen("LOG", "w");
  }
  ~CosineNodeTest() {
    fclose(logfile);
  }

  void test_CosineTreeNode() {
    Matrix A;
    data::Load("input.txt", &A);
    CosineNode root(A);
    ot::Print(root, "cosine root", logfile);

    root.Split();

    ot::Print(*root.left_, "left node", logfile);
    ot::Print(*root.right_, "right node", logfile);
  }
  
  void run_tests() {
    test_CosineTreeNode();
  }
};

class QuicSVDTest {
  FILE * logfile;
public:
  QuicSVDTest() {
    logfile = fopen("LOG", "w");
  }
  ~QuicSVDTest() {
    fclose(logfile);
  }

  void test_QuicSVD() {
    Matrix A;
    data::Load("input1.txt", &A);
    QuicSVD svd(A, 0.1);
    Vector s;
    Matrix U, VT;
    svd.ComputeSVD(&s, &U, &VT);
  }

  void run_tests() {
    test_QuicSVD();
  }

};

int main() {
  //CosineNodeTest test;
  //test.run_tests();

  QuicSVDTest test;
  test.run_tests();
}
