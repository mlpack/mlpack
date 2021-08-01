
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_tests_clean_memory.cpp:

Program Listing for File clean_memory.cpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_tests_clean_memory.cpp>` (``/home/aakash/mlpack/src/mlpack/bindings/tests/clean_memory.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include "clean_memory.hpp"
   
   #include <mlpack/core.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace tests {
   
   void CleanMemory()
   {
     // If we are holding any pointers, then we "own" them.  But we may hold the
     // same pointer twice, so we have to be careful to not delete it multiple
     // times.
     std::unordered_map<void*, util::ParamData*> memoryAddresses;
     auto it = IO::Parameters().begin();
     while (it != IO::Parameters().end())
     {
       util::ParamData& data = it->second;
   
       void* result;
       IO::GetSingleton().functionMap[data.tname]["GetAllocatedMemory"](data,
           NULL, (void*) &result);
       if (result != NULL && memoryAddresses.count(result) == 0)
         memoryAddresses[result] = &data;
   
       ++it;
     }
   
     // Now we have all the unique addresses that need to be deleted.
     std::unordered_map<void*, util::ParamData*>::const_iterator it2;
     it2 = memoryAddresses.begin();
     while (it2 != memoryAddresses.end())
     {
       util::ParamData& data = *(it2->second);
   
       IO::GetSingleton().functionMap[data.tname]["DeleteAllocatedMemory"](data,
           NULL, NULL);
   
       ++it2;
     }
   }
   
   } // namespace tests
   } // namespace bindings
   } // namespace mlpack
