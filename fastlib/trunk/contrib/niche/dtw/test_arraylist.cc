#include "fastlib/fastlib.h"

int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  
  ArrayList<Vector> vector_list1;
  vector_list1.Init(0, 2);

  for(int i = 0; i < 1; i++) {
    vector_list1.PushBack(1);
    vector_list1[i].Init(2);
    vector_list1[i].SetZero();
  }
  

  ArrayList<Vector> vector_list2;
  vector_list2.Init(0, 1);

  for(int i = 0; i < 2; i++) {
    Vector vec;
    vec.Init(2);
    vec.SetZero();

    vector_list2.PushBackCopy(vec);
  }

  

  ArrayList<Vector> vector_list3;
  vector_list3.Init(0, 2);

  for(int i = 0; i < 1; i++) {
    vector_list3.PushBackRaw(1);
    vector_list3[i].Init(2);
    vector_list3[i].SetZero();
  }

  

  
  fx_done(fx_root);
}
