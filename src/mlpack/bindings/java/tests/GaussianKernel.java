package org.mlpack;

import org.bytedeco.javacpp.*;

public class GaussianKernel {
  private final Pointer pointer;

  GaussianKernel(Pointer pointer) {
    this.pointer = pointer;
  }

  Pointer getPointer() {
    return pointer;
  }
}
