/*
 * =====================================================================================
 *
 *       Filename:  memory_manager_unit.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/11/2007 11:57:10 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include "memory_manager.h"

int main(int argc, char *argv[]) {
  
	MemoryMager<false>::allocator_ = new MemoryManager<false>();

}
