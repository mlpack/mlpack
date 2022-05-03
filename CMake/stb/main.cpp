// The purpose of this file is to include STB's implementation in two separate
// translation units.  One is a.cpp, and one is b.cpp.  This file simply
// includes both of those, so that when we get to the linking phase, we will
// have to link both translation units.
//
// Some versions of STB fail to correctly define some functions as
// static---which will cause a linking failure.  Thus, if this fails to
// compile, then mlpack's use of STB will fail.
#include "a.hpp"
#include "b.hpp"

int main()
{
  A::A();
  B::B();
}
