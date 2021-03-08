package mlpack

/*
#cgo CFLAGS: -I. -I/capi -g -Wall
#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR} -lmlpack_go_util
#include <capi/io_util.h>
*/
import "C"

import (
  "runtime"
  "unsafe"
)

func hasParam(identifier string) bool  {
  return bool((C.mlpackHasParam(C.CString(identifier))))
}

func setPassed(identifier string)  {
  C.mlpackSetPassed(C.CString(identifier))
}

func setParamDouble(identifier string, value float64)  {
  C.mlpackSetParamDouble(C.CString(identifier), C.double(value))
}

func setParamInt(identifier string, value int) {
  C.mlpackSetParamInt(C.CString(identifier), C.int(value))
}
func setParamFloat(identifier string, value float64) {
  C.mlpackSetParamFloat(C.CString(identifier), C.float(value))
}

func setParamBool(identifier string, value bool) {
  C.mlpackSetParamBool(C.CString(identifier), C.bool(value))
}

func setParamString(identifier string, value string) {
  C.mlpackSetParamString(C.CString(identifier), C.CString(value))
}

func setParamPtr(identifier string, ptr unsafe.Pointer) {
  C.mlpackSetParamPtr(C.CString(identifier), (*C.double)(ptr))
}
func resetTimers() {
  C.mlpackResetTimers()
}

func enableTimers() {
  C.mlpackEnableTimers()
}

func disableBacktrace() {
  C.mlpackDisableBacktrace()
}

func disableVerbose() {
  C.mlpackDisableVerbose()
}

func enableVerbose() {
  C.mlpackEnableVerbose()
}

func restoreSettings(method string) {
  C.mlpackRestoreSettings(C.CString(method))
}

func clearSettings() {
  C.mlpackClearSettings()
}

func getParamString(identifier string) string {
  val := C.GoString(C.mlpackGetParamString(C.CString(identifier)))
  return val
}

func getParamBool(identifier string) bool {
  val := bool(C.mlpackGetParamBool(C.CString(identifier)))
  return val
}

func getParamInt(identifier string) int {
  val := int(C.mlpackGetParamInt(C.CString(identifier)))
  return val
}

func getParamDouble(identifier string) float64 {
  val := float64(C.mlpackGetParamDouble(C.CString(identifier)))
  return val
}

type mlpackVectorType struct {
  mem unsafe.Pointer
}

func (v *mlpackVectorType) allocVecIntPtr(identifier string) {
  v.mem = C.mlpackGetVecIntPtr(C.CString(identifier))
  runtime.KeepAlive(v)
}

func setParamVecInt(identifier string, vecInt []int) {
  vecInt64 := make([]int64, len(vecInt))
  // Here we are promisely passing int64 to C++.
  for i := 0; i < len(vecInt); i++ {
    vecInt64[i] = int64(vecInt[i])
  }
  ptr := unsafe.Pointer(&vecInt64[0])
  // As we are not guaranteed  that int is always equivalent of int64_t or
  // int32_t in Go. Hence we are passing `long long` to C++.
  C.mlpackSetParamVectorInt(C.CString(identifier), (*C.longlong)(ptr),
                            C.size_t(len(vecInt)))
}

func setParamVecString(identifier string, vecString []string) {
  C.mlpackSetParamVectorStrLen(C.CString(identifier), C.size_t(len(vecString)))
  for i := 0; i < len(vecString); i++ {
    C.mlpackSetParamVectorStr(C.CString(identifier), (C.CString)(vecString[i]),
                              C.size_t(i))
  }
}

func getParamVecInt(identifier string) []int {
  e := int(C.mlpackVecIntSize(C.CString(identifier)))

  var v mlpackVectorType
  v.allocVecIntPtr(identifier)

  data := (*[1<<30 - 1]int)(v.mem)
  output := data[:e]
  if output != nil {
    return output
  }
  return []int{}
}

func getParamVecString(identifier string) []string {
  e := int(C.mlpackVecStringSize(C.CString(identifier)))

  data := make([]string, e)
  for i := 0; i < e; i++ {
    data[i] = C.GoString(C.mlpackGetVecStringPtr(C.CString(identifier),
                         C.size_t(i)))
  }
  return data
}
