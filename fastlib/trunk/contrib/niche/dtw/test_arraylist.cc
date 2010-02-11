#include "fastlib/fastlib.h"

int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  
  ArrayList<Vector> vector_list;

  vector_list.Init(0, 0);

 
  for(int i = 0; i < 2; i++) {
    Vector vector;
    vector.Init(2);
    vector.SetZero();

    vector_list.PushBackCopy(vector);

    //    vector_list.GrowTo(i+1);

    //vector_list[i].Destruct();
    
    //vector_list[i].Init(2);
    //vector_list[i].SetZero();
  }
  
  fx_done(fx_root);
}
