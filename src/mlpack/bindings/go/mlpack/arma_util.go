package mlpack

/*
#cgo CFLAGS: -I. -I/capi -g -Wall
#cgo LDFLAGS: -L. -lmlpack -lgo_util
#include <stdlib.h>
#include <stdio.h>
#include <capi/cli_util.h>
#include <capi/arma_util.h>
*/
import "C"

import (
	"runtime"
	"time"
	"unsafe"

	"gonum.org/v1/gonum/mat"
)

type MLPACK_Arma struct {
	mem unsafe.Pointer
}

// Function alloc allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *MLPACK_Arma) allocArmaPtr_mat(identifier string) {
	m.mem = C.MLPACK_ArmaPtr_mat(C.CString(identifier))
	runtime.KeepAlive(m)
}

// Function free is used to free memory when the object leaves Go's scope.
func free_mat(m *MLPACK_Arma) {
	C.free(unsafe.Pointer(m.mem))
}

// Function alloc allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *MLPACK_Arma) allocArmaPtr_umat(identifier string) {
	m.mem = C.MLPACK_ArmaPtr_umat(C.CString(identifier))
	runtime.KeepAlive(m)
}

// Function alloc allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *MLPACK_Arma) allocArmaPtr_row(identifier string) {
	m.mem = C.MLPACK_ArmaPtr_row(C.CString(identifier))
	runtime.KeepAlive(m)
}

// Function alloc allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *MLPACK_Arma) allocArmaPtr_urow(identifier string) {
	m.mem = C.MLPACK_ArmaPtr_urow(C.CString(identifier))
	runtime.KeepAlive(m)
}

// Function alloc allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *MLPACK_Arma) allocArmaPtr_col(identifier string) {
	m.mem = C.MLPACK_ArmaPtr_col(C.CString(identifier))
	runtime.KeepAlive(m)
}

// Function alloc allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *MLPACK_Arma) allocArmaPtr_ucol(identifier string) {
	m.mem = C.MLPACK_ArmaPtr_ucol(C.CString(identifier))
	runtime.KeepAlive(m)
}

// GonumToArma passes a gonum matrix to C by using it's gonums underlying blas64.
func GonumToArma_mat(identifier string, m *mat.Dense) {
	// Get matrix dimension, underlying blas64General matrix, and data.
	r, c := m.Dims()
	blas64General := m.RawMatrix()
	data := blas64General.Data

	// Pass pointer of the underlying matrix to Mlpack.
	ptr := unsafe.Pointer(&data[0])
	C.MLPACK_ToArma_mat(C.CString(identifier), (*C.double)(ptr), C.int(c), C.int(r))
}

// GonumToArma passes a gonum matrix to C by using it's gonums underlying blas64.
func GonumToArma_row(identifier string, m *mat.VecDense) {
	// Get matrix dimension, underlying blas64General matrix, and data.
	e := m.Len()
	blas64 := m.RawVector()
	data := blas64.Data

	// Pass pointer of the underlying matrix to Mlpack.
	ptr := unsafe.Pointer(&data[0])
	C.MLPACK_ToArma_row(C.CString(identifier), (*C.double)(ptr), C.int(e))
}

// GonumToArma passes a gonum matrix to C by using it's gonums underlying blas64.
func GonumToArma_col(identifier string, m *mat.VecDense) {
	// Get matrix dimension, underlying blas64General matrix, and data.
	e := m.Len()
	blas64General := m.RawVector()
	data := blas64General.Data

	// Pass pointer of the underlying matrix to Mlpack.
	ptr := unsafe.Pointer(&data[0])
	C.MLPACK_ToArma_col(C.CString(identifier), (*C.double)(ptr), C.int(e))
}

// ArmaToGonum returns a gonum matrix based on the memory pointer
// of an armadillo matrix.
func (m *MLPACK_Arma) ArmaToGonum_mat(identifier string) *mat.Dense {
	// Armadillo row and col
	c := int(C.MLPACK_NumRow_mat(C.CString(identifier)))
	r := int(C.MLPACK_NumCol_mat(C.CString(identifier)))
	e := int(C.MLPACK_NumElem_mat(C.CString(identifier)))

	// Allocate Go memory pointer to the armadillo matrix.
	m.allocArmaPtr_mat(identifier)
	runtime.GC()
	time.Sleep(time.Second)

	// Convert pointer to slice of data, to then pass it to a gonum matrix.
	array := (*[1<<30 - 1]float64)(m.mem)
	if array != nil {
		data := array[:e]

		runtime.GC()
		time.Sleep(time.Second)

		// Initialize result matrix.
		output := mat.NewDense(r, c, data)

		// Return gonum vector.
		return output
	}
	return nil
}

// ArmaToGonum returns a gonum matrix based on the memory pointer
// of an armadillo matrix.
func (m *MLPACK_Arma) ArmaToGonum_array(identifier string) (int, int, []float64) {
	// Armadillo row and col
	c := int(C.MLPACK_NumRow_mat(C.CString(identifier)))
	r := int(C.MLPACK_NumCol_mat(C.CString(identifier)))
	e := int(C.MLPACK_NumElem_mat(C.CString(identifier)))

	// Allocate Go memory pointer to the armadillo matrix.
	m.allocArmaPtr_mat(identifier)
	runtime.GC()
	time.Sleep(time.Second)

	// Convert pointer to slice of data, to then pass it to a gonum matrix.
	array := (*[1<<30 - 1]float64)(m.mem)

	data := array[0:e]
	return r, c, data
}

// ArmaToGonum returns a gonum matrix based on the memory pointer
// of an armadillo matrix.
func (m *MLPACK_Arma) ArmaToGonum_umat(identifier string) *mat.Dense {
	// Armadillo row and col
	c := int(C.MLPACK_NumRow_umat(C.CString(identifier)))
	r := int(C.MLPACK_NumCol_umat(C.CString(identifier)))
	e := int(C.MLPACK_NumElem_umat(C.CString(identifier)))

	// Allocate Go memory pointer to the armadillo matrix.
	m.allocArmaPtr_umat(identifier)
	runtime.GC()
	time.Sleep(time.Second)

	// Convert pointer to slice of data, to then pass it to a gonum matrix.
	array := (*[1<<30 - 1]float64)(m.mem)
	if array != nil {
		data := array[:e]

		runtime.GC()
		time.Sleep(time.Second)

		// Initialize result matrix.
		output := mat.NewDense(r, c, data)

		// Return gonum vector.
		return output
	}
	return nil
}

// ArmaRowToGonum returns a gonum vector based on the memory pointer
// of the underlying armadillo object.
func (m *MLPACK_Arma) ArmaToGonum_row(identifier string) *mat.VecDense {
	// Armadillo row and col
	e := int(C.MLPACK_NumElem_row(C.CString(identifier)))

	// Allocate Go memory pointer to the armadillo matrix.
	m.allocArmaPtr_row(identifier)
	runtime.GC()
	time.Sleep(time.Second)

	// Convert pointer to slice of data, to then pass it to a gonum matrix.
	array := (*[1<<30 - 1]float64)(m.mem)
	if array != nil {
		data := (*array)[:e]

		runtime.GC()
		time.Sleep(time.Second)
		// Initialize result matrix.
		output := mat.NewVecDense(e, data)

		// Return gonum vector.
		return output
	}
	return nil
}

// ArmaRowToGonum returns a gonum vector based on the memory pointer
// of the underlying armadillo object.
func (m *MLPACK_Arma) ArmaToGonum_urow(identifier string) *mat.VecDense {
	// Armadillo row and col
	e := int(C.MLPACK_NumElem_urow(C.CString(identifier)))

	// Allocate Go memory pointer to the armadillo matrix.
	m.allocArmaPtr_urow(identifier)
	runtime.GC()
	time.Sleep(time.Second)

	// Convert pointer to slice of data, to then pass it to a gonum matrix.
	array := (*[1<<30 - 1]float64)(m.mem)
	if array != nil {
		data := array[:e]

		runtime.GC()
		time.Sleep(time.Second)
		// Initialize result matrix.
		output := mat.NewVecDense(e, data)
		// Return gonum vector.
		return output
	}
	return nil
}

// GonumToArma passes a gonum matrix to C by using it's gonums underlying blas64.
func (m *MLPACK_Arma) ArmaToGonum_col(identifier string) *mat.VecDense {
	// Get matrix dimension, underlying blas64General matrix, and data.
	e := int(C.MLPACK_NumElem_col(C.CString(identifier)))

	// Allocate Go memory pointer to the armadillo matrix.
	m.allocArmaPtr_col(identifier)
	runtime.GC()
	time.Sleep(time.Second)

	// Convert pointer to slice of data, to then pass it to a gonum matrix.
	array := (*[1<<30 - 1]float64)(m.mem)
	if array != nil {
		data := array[:e]

		runtime.GC()
		time.Sleep(time.Second)

		// Initialize result matrix.
		output := mat.NewVecDense(e, data)

		// Return gonum vector.
		return output
	}
	return nil
}

// GonumToArma passes a gonum matrix to C by using it's gonums underlying blas64.
func (m *MLPACK_Arma) ArmaToGonum_ucol(identifier string) *mat.VecDense {
	// Get matrix dimension, underlying blas64General matrix, and data.
	e := int(C.MLPACK_NumElem_ucol(C.CString(identifier)))

	// Allocate Go memory pointer to the armadillo matrix.
	m.allocArmaPtr_ucol(identifier)
	runtime.GC()
	time.Sleep(time.Second)

	// Convert pointer to slice of data, to then pass it to a gonum matrix.
	array := (*[1<<30 - 1]float64)(m.mem)
	if array != nil {
		data := array[:]

		runtime.GC()
		time.Sleep(time.Second)

		// Initialize result matrix.
		output := mat.NewVecDense(e, data)

		// Return gonum vector.
		return output
	}
	return nil
}
