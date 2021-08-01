
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_kernel_traits_test.cpp:

Program Listing for File kernel_traits_test.cpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_kernel_traits_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/kernel_traits_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   
   #include "catch.hpp"
   #include "test_catch_tools.hpp"
   
   using namespace mlpack;
   using namespace mlpack::kernel;
   
   TEST_CASE("IsNormalizedTest", "[KernelTraitsTest]")
   {
     // Reason number ten billion why macros are bad:
     //
     // The Boost unit test framework is built on macros.  When I write
     // REQUIRE(KernelTraits<int>::IsNormalized, false) == what actually
     // happens (in gcc at least) is that the 'false' gets implicitly converted to
     // an int; then, the compiler goes looking for an int IsNormalized variable in
     // KernelTraits.  But this doesn't exist, so we get this error at linker time:
     //
     // kernel_traits_test.cpp:(.text+0xb86): undefined reference to
     // `mlpack::kernel::KernelTraits<mlpack::kernel::LinearKernel>::IsNormalized'
     //
     // and this actually tells us nothing about the error.  When you dig deep
     // enough or get frustrated enough, you end up realizing it's a macro problem
     // and now have to *explicitly* cast to bool.  Yes, really; try it if you
     // don't believe me.
   
     // Test each kernel individually.
     // If the type is not a valid kernel, it should be false (default value).
     REQUIRE((bool) KernelTraits<int>::IsNormalized == false);
   
     // Normalized kernels.
     REQUIRE((bool) KernelTraits<CosineDistance>::IsNormalized == true);
     REQUIRE((bool) KernelTraits<EpanechnikovKernel>::IsNormalized == true);
     REQUIRE((bool) KernelTraits<GaussianKernel>::IsNormalized == true);
     REQUIRE((bool) KernelTraits<LaplacianKernel>::IsNormalized == true);
     REQUIRE((bool) KernelTraits<SphericalKernel>::IsNormalized == true);
     REQUIRE((bool) KernelTraits<TriangularKernel>::IsNormalized == true);
     REQUIRE((bool) KernelTraits<CauchyKernel>::IsNormalized == true);
   
     // Unnormalized kernels.
     REQUIRE((bool) KernelTraits<LinearKernel>::IsNormalized == false);
     REQUIRE((bool) KernelTraits<PolynomialKernel>::IsNormalized == false);
     REQUIRE((bool) KernelTraits<PSpectrumStringKernel>::IsNormalized == false);
   }
