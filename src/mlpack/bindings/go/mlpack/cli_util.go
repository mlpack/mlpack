package mlpack

/*
#cgo CFLAGS: -I. -I/capi -g -Wall
#cgo LDFLAGS: -L/usr/local/lib -lm -L/usr/local/lib/ -lmlpack -lgo_util
#include <capi/cli_util.h>
*/
import "C"

import (
	"runtime"
	"time"
	"unsafe"
)

func HasParam(identifier string) bool {
	return bool((C.MLPACK_HasParam(C.CString(identifier))))
}

func SetPassed(identifier string) {
	C.MLPACK_SetPassed(C.CString(identifier))
}

func SetParamDouble(identifier string, value float64) {
	C.MLPACK_SetParamDouble(C.CString(identifier), C.double(value))
}

func SetParamInt(identifier string, value int) {
	C.MLPACK_SetParamInt(C.CString(identifier), C.int(value))
}
func SetParamFloat(identifier string, value float64) {
	C.MLPACK_SetParamFloat(C.CString(identifier), C.float(value))
}

func SetParamBool(identifier string, value bool) {
	C.MLPACK_SetParamBool(C.CString(identifier), C.bool(value))
}

func SetParamString(identifier string, value string) {
	C.MLPACK_SetParamString(C.CString(identifier), C.CString(value))
}

func SetParamPtr(identifier string, ptr unsafe.Pointer, copy bool) {
	C.MLPACK_SetParamPtr(C.CString(identifier), (*C.double)(ptr), C.bool(copy))
}
func ResetTimers() {
	C.MLPACK_ResetTimers()
}

func EnableTimers() {
	C.MLPACK_EnableTimers()
}

func DisableBacktrace() {
	C.MLPACK_DisableBacktrace()
}

func DisableVerbose() {
	C.MLPACK_DisableVerbose()
}

func EnableVerbose() {
	C.MLPACK_EnableVerbose()
}

func RestoreSettings(method string) {
	C.MLPACK_RestoreSettings(C.CString(method))
}

func ClearSettings() {
	C.MLPACK_ClearSettings()
}

func GetParamString(identifier string) string {
	val := C.GoString(C.MLPACK_GetParamString(C.CString(identifier)))
	return val
}

func GetParamBool(identifier string) bool {
	val := bool(C.MLPACK_GetParamBool(C.CString(identifier)))
	return val
}

func GetParamInt(identifier string) int {
	val := int(C.MLPACK_GetParamInt(C.CString(identifier)))
	return val
}

func GetParamDouble(identifier string) float64 {
	val := float64(C.MLPACK_GetParamDouble(C.CString(identifier)))
	return val
}

type MLPACK_VectorType struct {
	mem unsafe.Pointer
}

func (v *MLPACK_VectorType) allocVecStringPtr(identifier string) {
	v.mem = C.MLPACK_GetVecStringPtr(C.CString(identifier))
	runtime.KeepAlive(v)
}

func (v *MLPACK_VectorType) allocVecIntPtr(identifier string) {
	v.mem = C.MLPACK_GetVecIntPtr(C.CString(identifier))
	runtime.KeepAlive(v)
}

func SetParamVecInt(identifier string, vecInt []int) {
	ptr := unsafe.Pointer(&vecInt[0])
	C.MLPACK_SetParamPtr(C.CString(identifier), (*C.double)(ptr), true)
}

func SetParamVecString(identifier string, vecString []string) {
	ptr := unsafe.Pointer(&vecString[0])
	C.MLPACK_SetParamPtr(C.CString(identifier), (*C.double)(ptr), true)
}

func GetParamVecInt(identifier string) []int {
	e := int(C.MLPACK_VecIntSize(C.CString(identifier)))

	var v MLPACK_VectorType
	v.allocVecIntPtr(identifier)
	runtime.GC()
	time.Sleep(time.Second)

	// Convert pointer to slice of data, to then pass it to a gonum matrix.
	data := (*[1<<30 - 1]int)(v.mem)
	output := data[:e]
	if output != nil {
		return output
	}
	return nil
}

func GetParamVecString(identifier string) []string {
	e := int(C.MLPACK_VecStringSize(C.CString(identifier)))

	var v MLPACK_VectorType
	v.allocVecStringPtr(identifier)
	runtime.GC()
	time.Sleep(time.Second)

	// Convert pointer to slice of data, to then pass it to a gonum matrix.
	data := (*[1<<30 - 1]string)(v.mem)
	output := data[:e]
	if output != nil {
		return output
	}
	return nil
}
