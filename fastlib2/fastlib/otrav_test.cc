#include "fastlib/base/base.h"
#include "fastlib/base/test.h"
#include "fastlib/la/matrix.h"
#include "fastlib/data/dataset.h"

TEST_SUITE_BEGIN(otrav)

void TestDatasetPrint() {
  Dataset d;
  
  d.InitFromFile("fake.arff");
  ot::Print(d, "d", stderr);
}

void TestDatasetLayout() {
  Dataset d;
  
  d.InitFromFile("fake.arff");
  ot::Print(d, "d", stderr);
  
  size_t size = ot::FrozenSize(d);
  
  printf("SIZE IS %d\n", int(size));
  
  char dump[size];
  
  ot::Freeze(dump, d);
  
  Dataset *d2;
  d2 = ot::SemiThaw<Dataset>(dump);
  ot::Print(*d2, "d2");
  char copy[size];
  mem::BitCopyBytes(copy, reinterpret_cast<char *>(d2), size);
  ot::SemiFreeze<Dataset>(copy, d2);
  
  Dataset *d3;
  d3 = ot::SemiThaw<Dataset>(copy);
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
