/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
#include "base/base.h"
#include "base/test.h"
#include "la/matrix.h"
#include "data/dataset.h"

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
