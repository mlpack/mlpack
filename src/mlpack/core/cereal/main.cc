#include "pointer_wrapper.hpp"
#include <fstream>

void ref_to_pointer(double*& p)
{
  if(p == nullptr) {
    p = new double;
  }
  std::cout << "address of  pointer" << p << std::endl;
  
}

int main(int argc, char* argv[])
{

  std::ifstream is("data.xml");
  cereal::XMLInputArchive archive(is);
  
  // std::ofstream is("data.xml");
  // cereal::XMLOutputArchive archive(is);
  
  double data = 40;
  double* d = nullptr;

  //ref_to_pointer(d); 

  archive(cereal::make_pointer(d));

  if ( d == NULL ) {
    std::cout << "d is still NULL" << std::endl;
  } else {

    std::cout << "d value is : " << *d << std::endl;
  }
}
