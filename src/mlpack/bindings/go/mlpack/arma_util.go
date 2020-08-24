package mlpack

/*
#cgo CFLAGS: -I. -I/capi -g -Wall -Wno-unused-variable 
#cgo LDFLAGS: -L. -lmlpack_go_util
#include <stdlib.h>
#include <stdio.h>
#include <capi/io_util.h>
#include <capi/arma_util.h>
*/
import "C"

import (
  "runtime"
  "unsafe"

  "gonum.org/v1/gonum/mat"
)

type mlpackArma struct {

  mem unsafe.Pointer
}

// A Tuple containing `float64` data (data) along with a boolean array
// (Categoricals) indicating which dimensions are categorical (represented by
// `true`) and which are numeric (represented by `false`).  The number of
// elements in the boolean array should be the same as the dimensionality of
// the data matrix.  It is expected that each row of the matrix corresponds to a 
// single data point when calling mlpack bindings.
type matrixWithInfo struct {
  Categoricals []bool
  Data *mat.Dense
}

// A function used for initializing matrixWithInfo Tuple.
func DataAndInfo() *matrixWithInfo {
  return &matrixWithInfo {
    Categoricals: nil,
    Data: nil,
  }
}

// Allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *mlpackArma) allocArmaPtrMat(identifier string) {
  m.mem = C.mlpackArmaPtrMat(C.CString(identifier))
  runtime.KeepAlive(m)
}

// Allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *mlpackArma) allocArmaPtrUmat(identifier string) {
  m.mem = C.mlpackArmaPtrUmat(C.CString(identifier))
  runtime.KeepAlive(m)
}

// Allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *mlpackArma) allocArmaPtrRow(identifier string) {
  m.mem = C.mlpackArmaPtrRow(C.CString(identifier))
  runtime.KeepAlive(m)
}

// Allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *mlpackArma) allocArmaPtrUrow(identifier string) {
  m.mem = C.mlpackArmaPtrUrow(C.CString(identifier))
  runtime.KeepAlive(m)
}

// Allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *mlpackArma) allocArmaPtrCol(identifier string) {
  m.mem = C.mlpackArmaPtrCol(C.CString(identifier))
  runtime.KeepAlive(m)
}

// Allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *mlpackArma) allocArmaPtrUcol(identifier string) {
  m.mem = C.mlpackArmaPtrUcol(C.CString(identifier))
  runtime.KeepAlive(m)
}

// Allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *mlpackArma) allocArmaPtrMatWithInfo(identifier string) {
  m.mem = C.mlpackArmaPtrMatWithInfoPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

// Passes a Gonum matrix to C by using the underlying data from the Gonum matrix.
func gonumToArmaMat(identifier string, m *mat.Dense) {
  // Get the number of elements in the Armadillo column.
  r, c := m.Dims()
  blas64General := m.RawMatrix()
  data := blas64General.Data

  // Pass pointer of the underlying matrix to mlpack.
  ptr := unsafe.Pointer(&data[0])
  C.mlpackToArmaMat(C.CString(identifier), (*C.double)(ptr), C.size_t(c), C.size_t(r))
}

// Passes a Gonum matrix to C by using the underlying data from the Gonum matrix.
func gonumToArmaUmat(identifier string, m *mat.Dense) {
  // Get the number of elements in the Armadillo column.
  r, c := m.Dims()
  blas64General := m.RawMatrix()
  data := blas64General.Data

  // Pass pointer of the underlying matrix to mlpack.
  ptr := unsafe.Pointer(&data[0])
  C.mlpackToArmaUmat(C.CString(identifier), (*C.double)(ptr), C.size_t(c), C.size_t(r))
}

// Passes a Gonum matrix to C by using the underlying data from the Gonum matrix.
func gonumToArmaRow(identifier string, m *mat.Dense) {
  // Get the number of elements in the Armadillo column.
  e, err := m.Dims()
  if (err != 1 && e != 1){
    panic("Given matrix must have a single column")
  }

  // Transpose if Column vector is given
  if e == 1 {
    m = mat.DenseCopyOf(m.T())
    e = err
  }

  blas64General := m.RawMatrix()
  data := blas64General.Data

  // Pass pointer of the underlying matrix to mlpack.
  ptr := unsafe.Pointer(&data[0])
  C.mlpackToArmaRow(C.CString(identifier), (*C.double)(ptr), C.size_t(e))
}

// Passes a Gonum matrix to C by using the underlying data from the Gonum matrix.
func gonumToArmaUrow(identifier string, m *mat.Dense) {
  // Get the number of elements in the Armadillo column.
  e, err := m.Dims()
  if (err != 1 && e != 1){
    panic("Given matrix must have a single column")
  }

  // Transpose if Column vector is given
  if e == 1 {
    m = mat.DenseCopyOf(m.T())
    e = err
  }

  blas64General := m.RawMatrix()
  data := blas64General.Data

  // Pass pointer of the underlying matrix to mlpack.
  ptr := unsafe.Pointer(&data[0])
  C.mlpackToArmaUrow(C.CString(identifier), (*C.double)(ptr), C.size_t(e))
}

// Passes a Gonum matrix to C by using the underlying data from the Gonum matrix.
func gonumToArmaCol(identifier string, m *mat.Dense) {
  // Get the number of elements in the Armadillo column.
  err, e := m.Dims()
  if (err != 1 && e != 1){
    panic("Given matrix must have a single row")
  }

  // Transpose if Row vector is given
  if e == 1 {
    m = mat.DenseCopyOf(m.T())
    e = err
  }

  blas64General := m.RawMatrix()
  data := blas64General.Data

  // Pass pointer of the underlying matrix to mlpack.
  ptr := unsafe.Pointer(&data[0])
  C.mlpackToArmaCol(C.CString(identifier), (*C.double)(ptr), C.size_t(e))
}

// Passes a Gonum matrix to C by using the underlying data from the Gonum matrix.
func gonumToArmaUcol(identifier string, m *mat.Dense) {
  // Get the number of elements in the Armadillo column.
  err, e := m.Dims()
  if (err != 1 && e != 1){
    panic("Given matrix must have a single row")
  }

  // Transpose if Row vector is given
  if e == 1 {
    m = mat.DenseCopyOf(m.T())
    e = err
  }

  blas64General := m.RawMatrix()
  data := blas64General.Data

  // Pass pointer of the underlying matrix to mlpack.
  ptr := unsafe.Pointer(&data[0])
  C.mlpackToArmaUcol(C.CString(identifier), (*C.double)(ptr), C.size_t(e))
}

// GonumToArmaMatWithInfo passes a gonum matrix with info to C by 
// using it's gonums underlying blas64.
func gonumToArmaMatWithInfo(identifier string, m *matrixWithInfo) {
  // Get the number of elements in the Armadillo column.
  r, c := m.Data.Dims()
  blas64General := m.Data.RawMatrix()
  dataAndInfo := blas64General.Data
  boolarray := m.Categoricals
  // Pass pointer of the underlying matrix to mlpack.
  boolptr := unsafe.Pointer(&boolarray[0])
  matptr := unsafe.Pointer(&dataAndInfo[0])
  C.mlpackToArmaMatWithInfo(C.CString(identifier), (*C.bool)(boolptr),
      (*C.double)(matptr), C.size_t(c), C.size_t(r))
}

// ArmaToGonum returns a gonum matrix based on the memory pointer
// of an armadillo matrix.
func (m *mlpackArma) armaToGonumMat(identifier string) *mat.Dense {
  // Get the number of elements in the Armadillo row.
  c := int(C.mlpackNumRowMat(C.CString(identifier)))
  r := int(C.mlpackNumColMat(C.CString(identifier)))
  e := int(C.mlpackNumElemMat(C.CString(identifier)))

  // Allocate Go memory pointer to the armadillo matrix.
  m.allocArmaPtrMat(identifier)

  // Convert pointer to slice of data, to then pass it to a gonum matrix.
  array := (*[1<<30 - 1]float64)(m.mem)
  if array != nil {
    data := array[:e]

    // Initialize result matrix.
    output := mat.NewDense(r, c, data)

    // Return gonum vector.
    return output
  }
  return mat.NewDense(1, 1, nil)
}

// ArmaToGonum returns a gonum matrix based on the memory pointer
// of an armadillo matrix.
func (m *mlpackArma) armaToGonumArray(identifier string) (int, int, []float64){
  // Get the number of elements in the Armadillo row.
  c := int(C.mlpackNumRowMat(C.CString(identifier)))
  r := int(C.mlpackNumColMat(C.CString(identifier)))
  e := int(C.mlpackNumElemMat(C.CString(identifier)))

  // Allocate Go memory pointer to the armadillo matrix.
  m.allocArmaPtrMat(identifier)

  // Convert pointer to slice of data, to then pass it to a gonum matrix.
  array := (*[1<<30 - 1]float64)(m.mem)

  data := array[:e]
  return r, c, data
}

// ArmaToGonum returns a gonum matrix based on the memory pointer
// of an armadillo matrix.
func (m *mlpackArma) armaToGonumUmat(identifier string) *mat.Dense {
  // Get the number of elements in the Armadillo row.
  c := int(C.mlpackNumRowUmat(C.CString(identifier)))
  r := int(C.mlpackNumColUmat(C.CString(identifier)))
  e := int(C.mlpackNumElemUmat(C.CString(identifier)))

  // Allocate Go memory pointer to the armadillo matrix.
  m.allocArmaPtrUmat(identifier)

  // Convert pointer to slice of data, to then pass it to a gonum matrix.
  array := (*[1<<30 - 1]float64)(m.mem)
  if array != nil {
    data := array[:e]

    // Initialize result matrix.
    output := mat.NewDense(r, c, data)

    // Return gonum vector.
    return output
  }
  return mat.NewDense(1, 1, nil)
}

// ArmaRowToGonum returns a gonum vector based on the memory pointer
// of the underlying armadillo object.
func (m *mlpackArma) armaToGonumRow(identifier string) *mat.Dense{
  // Get the number of elements in the Armadillo row.
  e := int(C.mlpackNumElemRow(C.CString(identifier)))

  // Allocate Go memory pointer to the armadillo matrix.
  m.allocArmaPtrRow(identifier)

  // Convert pointer to slice of data, to then pass it to a gonum matrix.
  array := (*[1<<30 - 1]float64)(m.mem)
  if array != nil {
    data := array[:e]

    // Initialize result matrix.
    output := mat.NewDense(e, 1, data)

    // Return gonum vector.
    return output
  }
  return mat.NewDense(1, 1, nil)
}

// ArmaRowToGonum returns a gonum vector based on the memory pointer
// of the underlying armadillo object.
func (m *mlpackArma) armaToGonumUrow(identifier string) *mat.Dense {
  // Get the number of elements in the Armadillo row.
  e := int(C.mlpackNumElemUrow(C.CString(identifier)))

  // Allocate Go memory pointer to the armadillo matrix.
  m.allocArmaPtrUrow(identifier)

  // Convert pointer to slice of data, to then pass it to a gonum matrix.
  array := (*[1<<30 - 1]float64)(m.mem)
  if array != nil {
    data := array[:e]

    // Initialize result matrix.
    output := mat.NewDense(e, 1, data)
    // Return gonum vector.
    return output
  }
  return mat.NewDense(1, 1, nil)
}

// Passes a Gonum matrix to C by using the underlying data from the Gonum matrix.
func (m *mlpackArma) armaToGonumCol(identifier string) *mat.Dense {
  // Get the number of elements in the Armadillo column.
  e := int(C.mlpackNumElemCol(C.CString(identifier)))

  // Allocate Go memory pointer to the armadillo matrix.
  m.allocArmaPtrCol(identifier)

  // Convert pointer to slice of data, to then pass it to a gonum matrix.
  array := (*[1<<30 - 1]float64)(m.mem)
  if array != nil {
    data := array[:e]

    // Initialize result matrix.
    output := mat.NewDense(1, e, data)

    // Return gonum vector.
    return output
  }
  return mat.NewDense(1, 1, nil)
}

// Passes a Gonum matrix to C by using the underlying data from the Gonum matrix.
func (m *mlpackArma) armaToGonumUcol(identifier string) *mat.Dense {
  // Get the number of elements in the Armadillo column.
  e := int(C.mlpackNumElemUcol(C.CString(identifier)))

  // Allocate Go memory pointer to the armadillo matrix.
  m.allocArmaPtrUcol(identifier)

  // Convert pointer to slice of data, to then pass it to a gonum matrix.
  array := (*[1<<30 - 1]float64)(m.mem)
  if array != nil {
    data := array[:e]

    // Initialize result matrix.
    output := mat.NewDense(1, e, data)

    // Return gonum vector.
    return output
  }
  return mat.NewDense(1, 1, nil)
}

// Passes a Gonum matrix to C by using the underlying data from the Gonum matrix.
func (m *mlpackArma) armaToGonumMatWithInfo(identifier string) *mat.Dense {
  // Get number of rows, columns, and elements of the Armadillo matrix.
  c := int(C.mlpackArmaMatWithInfoRows(C.CString(identifier)))
  r := int(C.mlpackArmaMatWithInfoCols(C.CString(identifier)))
  e := int(C.mlpackArmaMatWithInfoElements(C.CString(identifier)))

  // Allocate Go memory pointer to the armadillo matrix.
  m.allocArmaPtrMatWithInfo(identifier)
  matarray := (*[1<<30 - 1]float64)(m.mem)

  if matarray != nil {
    data := matarray[:e]

    // Initialize result matrix.
    output := mat.NewDense(r, c, data)

    // Return gonum vector.
    return output
  }
  return mat.NewDense(1, 1, nil)
}
