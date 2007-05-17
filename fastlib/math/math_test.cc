#include "math.h"

#include "base/test.h"

TEST_SUITE_BEGIN(math)

void TestPermutation() {
  ArrayList<index_t> p_rand;
  ArrayList<index_t> p_idnt;
  ArrayList<int> visited;
  int n = 3111;
  
  math::MakeRandomPermutation(n, &p_rand);
  math::MakeIdentityPermutation(n, &p_idnt);
  
  for (int i = 0; i < n; i++) {
    TEST_ASSERT(p_idnt[i] == i);
  }
  
  visited.Init(n);
  
  for (int i = 0; i < n; i++) {
    visited[i] = 0;
  }

  for (int i = 0; i < n; i++) {
    visited[p_rand[i]]++;
  }

  for (int i = 0; i < n; i++) {
    TEST_ASSERT(visited[i] == 1);
  }
}

void TestFactorial() {
  TEST_ASSERT(math::Factorial(0) == 1.0);
  TEST_ASSERT(math::Factorial(1) == 1.0);
  TEST_ASSERT(math::Factorial(2) == 2.0);
  TEST_ASSERT(math::Factorial(3) == 6.0);
  TEST_ASSERT(math::Factorial(4) == 24.0);
  TEST_ASSERT(math::Factorial(5) == 120.0);
  TEST_ASSERT(math::Factorial(6) == 720.0);
}

void TestSphereVol() {
  TEST_ASSERT(fabs(math::SphereVolume(2.0, 1) - 2.0 * 2.0) < 1.0e-7);
  TEST_ASSERT(fabs(math::SphereVolume(2.0, 2) - math::PI * 4.0) < 1.0e-7);
  TEST_ASSERT(fabs(math::SphereVolume(2.0, 3) - 4.0 / 3.0 * math::PI * 8.0) < 1.0e-7);
  TEST_ASSERT(fabs(math::SphereVolume(3.0, 3) - 4.0 / 3.0 * math::PI * 27.0) < 1.0e-7);
}

void TestKernel() {
  GaussianKernel k;
  
  k.Init(2.0);
  
  TEST_ASSERT(k.EvalUnnormOnSq(0) == 1.0);
  TEST_ASSERT(k.EvalUnnorm(0) == 1.0);
  
  TEST_ASSERT(fabs(k.EvalUnnormOnSq(1) - exp(-0.125)) < 1.0e-7);
  
  TEST_ASSERT((k.CalcNormConstant(1) - sqrt(4 * 2 * math::PI)) < 1.0e-7);

  EpanKernel k2;
  k2.Init(2);
  TEST_ASSERT(k2.EvalUnnormOnSq(0) == 1.0);
  TEST_ASSERT(fabs(k2.EvalUnnormOnSq(1) - 0.75) < 1.0e-7);
  TEST_ASSERT(fabs(k2.CalcNormConstant(1) - 4.0 * 2.0 / 3.0) < 1.0e-7);
  // TODO: Test the constant factor
}

void TestMisc() {
  TEST_ASSERT(math::Sqr(3.0) == 9.0);
  TEST_ASSERT(math::Sqr(-3.0) == 9.0);
  TEST_ASSERT(math::Sqr(0.0) == 0.0);
  TEST_ASSERT(math::Sqr(0.0) == 0.0);
  TEST_ASSERT(math::ClampNonNegative(-1.0) == 0.0);
  TEST_ASSERT(math::ClampNonNegative(0.0) == 0.0);
  TEST_ASSERT(math::ClampNonNegative(2.25) == 2.25);
  TEST_ASSERT(math::ClampNonPositive(-1.25) == -1.25);
  TEST_ASSERT(math::ClampNonPositive(0.0) == 0.0);
  TEST_ASSERT(math::ClampNonPositive(2.25) == 0.0);
}

void TestPow() {
  TEST_DOUBLE_EXACT((math::Pow<2, 2>(3.0)), 3.0);
  TEST_DOUBLE_EXACT((math::Pow<1, 1>(3.0)), 3.0);
  TEST_DOUBLE_EXACT((math::Pow<2, 1>(3.0)), 9.0);
  TEST_DOUBLE_EXACT((math::Pow<1, 2>(9.0)), 3.0);
  TEST_DOUBLE_EXACT((math::Pow<3, 3>(9.0)), 9.0);
  TEST_DOUBLE_EXACT((math::Pow<1, 3>(8.0)), 2.0);
  TEST_DOUBLE_EXACT((math::PowAbs<1, 3>(-8.0)), 2.0);
  TEST_DOUBLE_EXACT((math::PowAbs<1, 1>(-8.0)), 8.0);
  TEST_DOUBLE_EXACT((math::PowAbs<2, 1>(-8.0)), 64.0);
}

TEST_SUITE_END(math, TestPermutation, TestFactorial,
    TestSphereVol, TestKernel, TestMisc, TestPow)


