#ifndef MMGR_H
#define MMGR_H

using namespace std;

void *my_malloc(size_t size) {
  void *ptr;
  if (size <= 0) size=1; /* for AIX compatibility */
  ptr=(void *)malloc(size);
  if (!ptr) { 
    cerr << "Out of memory! Program exists!" << endl; 
    exit (1); 
  }
  return ptr;
}

#endif
