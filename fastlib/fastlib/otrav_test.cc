#include "base/test.h"
#include "base/otrav.h"
#include "la/matrix.h"
#include "data/dataset.h"

TEST_SUITE_BEGIN(otrav)

void TestDatasetPrint() {
  Dataset d;
  
  d.InitFromFile("fake.arff");
  OTPrint(d, stderr);
}

void TestDatasetLayout() {
  Dataset d;
  
  d.InitFromFile("fake.arff");
  OTPrint(d, stderr);
  
  size_t size = OTPointerFrozenSize(d);
  
  printf("SIZE IS %d\n", int(size));
  
  char dump[size];
  
  OTPointerFreeze(d, dump);
  
  Dataset *d2;
  d2 = OTPointerThaw<Dataset>(dump);
  OTPointerRefreeze<Dataset>(d2);
  d2 = OTPointerThaw<Dataset>(dump);
  d2->WriteArff("test_dataset_layout.arff");
}

TEST_SUITE_END(otrav, TestDatasetPrint, TestDatasetLayout)
