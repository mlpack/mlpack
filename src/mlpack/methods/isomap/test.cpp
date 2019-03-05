#include <mlpack/prereqs.hpp>
using namespace std;
using namespace arma;
int main()
{
  mat a = {{1,2,3}};
  mat b;
  b = a;
  cout << b;
  return 0;
}