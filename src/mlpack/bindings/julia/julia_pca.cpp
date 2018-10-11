#include "julia_pca.h"

#define BINDING_TYPE BINDING_TYPE_PYX
#include <mlpack/methods/pca/pca_main.cpp>

static void pca_mlpackMain()
{
  mlpackMain();
}

extern "C"
{

void pca()
{
  pca_mlpackMain();
}

void loadSymbols()
{
  // Do nothing.
}

}
