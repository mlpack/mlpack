namespace la {

  inline int DistanceSqEuclidean(
      index_t length, const int *va, const int *vb) {
    int s = 0;
    do {
      int d = *va++ - *vb++;
      s += d * d;
    } while (--length);
    return s;
  }

  inline int DistanceSqEuclidean(
      const GenVector<int>& x, const GenVector<int>& y) {
    DEBUG_SAME_SIZE(x.length(), y.length());
    return DistanceSqEuclidean(x.length(), x.ptr(), y.ptr());
  }

  inline void AddTo(const GenVector<int> &x, GenVector<int> *y) {
    DEBUG_SAME_SIZE(x.length(), y->length());
    printf("Never call this function for GenVector<int>!\n Exiting...\n");
    exit(1);
  }

  inline void Scale(index_t length, int alpha, int *x) {
    const int *end = x + length;
    while (x != end) {
      *x++ *= alpha;
    }
  }
  
  inline void Scale(int alpha, GenVector<int> *x) {
    Scale(x->length(), alpha, x->ptr());
  }
  
};
