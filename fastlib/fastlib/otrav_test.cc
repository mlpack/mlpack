#include "base/base.h"
#include "base/test.h"
#include "la/matrix.h"
#include "data/dataset.h"

TEST_SUITE_BEGIN(otrav)

void TestDatasetPrint() {
  Dataset d;
  
  d.InitFromFile("fake.arff");
  ot::Print(d, stderr);
}

void TestDatasetLayout() {
  Dataset d;
  
  d.InitFromFile("fake.arff");
  ot::Print(d, stderr);
  
  size_t size = ot::PointerFrozenSize(d);
  
  printf("SIZE IS %d\n", int(size));
  
  char dump[size];
  
  ot::PointerFreeze(d, dump);
  
  Dataset *d2;
  d2 = ot::PointerThaw<Dataset>(dump);
  ot::Print(*d2);
  char copy[size];
  mem::BitCopyBytes(copy, reinterpret_cast<char *>(d2), size);
  ot::PointerRefreeze<Dataset>(d2, copy);
  
  Dataset *d3;
  d3 = ot::PointerThaw<Dataset>(copy);
  d3->WriteArff("test_dataset_layout.arff");
}

void TestCopy() {
  Dataset *d = new Dataset;
  
  d->InitFromFile("fake.arff");

  Dataset d2(*d);

  delete d;

  d2.WriteArff("test_dataset_copy.arff");
}

TEST_SUITE_END(otrav, TestDatasetPrint, TestDatasetLayout, TestCopy)
