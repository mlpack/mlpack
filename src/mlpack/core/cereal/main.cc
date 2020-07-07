#include "pointer_wrapper.hpp"
#include <fstream>

int
main(int argc, char* argv[])
{
  std::ifstream is("data.xml");
  cereal::XMLInputArchive archive(is);

  // std::ofstream is("data.xml");
  // cereal::XMLOutputArchive archive(is);

  double data = 40;
  double* z = nullptr;

  // cereal::pointer_wrapper<double> d = cereal::make_pointer(z);
  // archive(d);
  // z = d.release();

  archive(CEREAL_POINTER(z));

  std::cout << "The address of z is: " << z << std::endl;

  if (z == nullptr) {
    std::cout << "z is still NULL" << std::endl;
  } else {
    std::cout << "z value is : " << *z << std::endl;
  }
  delete z;
}
