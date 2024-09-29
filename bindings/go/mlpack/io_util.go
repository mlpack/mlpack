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

type params struct {
  mem unsafe.Pointer
}

type timers struct {
  mem unsafe.Pointer
}

func getParams(binding string) *params {
  ptr := C.mlpackGetParams(C.CString(binding))
  p := &params { mem: ptr }
  runtime.KeepAlive(p)
  return p
}

func getTimers() *timers {
  ptr := C.mlpackGetTimers()
  t := &timers { mem: ptr }
  runtime.KeepAlive(t)
  return t
}

func cleanParams(p *params) {
  C.mlpackCleanParams(p.mem)
}

func cleanTimers(t *timers) {
  C.mlpackCleanTimers(t.mem)
}

func hasParam(p *params, identifier string) bool {
  return bool((C.mlpackHasParam(p.mem, C.CString(identifier))))
}

func setPassed(p *params, identifier string) {
  C.mlpackSetPassed(p.mem, C.CString(identifier))
}

func setParamDouble(p *params, identifier string, value float64) {
  C.mlpackSetParamDouble(p.mem, C.CString(identifier), C.double(value))
}

func setParamInt(p *params, identifier string, value int) {
  C.mlpackSetParamInt(p.mem, C.CString(identifier), C.int(value))
}
func setParamFloat(p *params, identifier string, value float64) {
  C.mlpackSetParamFloat(p.mem, C.CString(identifier), C.float(value))
}

func setParamBool(p *params, identifier string, value bool) {
  C.mlpackSetParamBool(p.mem, C.CString(identifier), C.bool(value))
}

func setParamString(p *params, identifier string, value string) {
  C.mlpackSetParamString(p.mem, C.CString(identifier), C.CString(value))
}

func setParamPtr(p *params, identifier string, ptr unsafe.Pointer) {
  C.mlpackSetParamPtr(p.mem, C.CString(identifier), (*C.double)(ptr))
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

func getParamString(p *params, identifier string) string {
  val := C.GoString(C.mlpackGetParamString(p.mem, C.CString(identifier)))
  return val
}

func getParamBool(p *params, identifier string) bool {
  val := bool(C.mlpackGetParamBool(p.mem, C.CString(identifier)))
  return val
}

func getParamInt(p *params, identifier string) int {
  val := int(C.mlpackGetParamInt(p.mem, C.CString(identifier)))
  return val
}

func getParamDouble(p *params, identifier string) float64 {
  val := float64(C.mlpackGetParamDouble(p.mem, C.CString(identifier)))
  return val
}

type mlpackVectorType struct {
  mem unsafe.Pointer
}

func (v *mlpackVectorType) allocVecIntPtr(p *params, identifier string) {
  v.mem = C.mlpackGetVecIntPtr(p.mem, C.CString(identifier))
  runtime.KeepAlive(v)
}

func setParamVecInt(p *params, identifier string, vecInt []int) {
  vecInt64 := make([]int64, len(vecInt))
  // Here we are promisely passing int64 to C++.
  for i := 0; i < len(vecInt); i++ {
    vecInt64[i] = int64(vecInt[i])
  }
  ptr := unsafe.Pointer(&vecInt64[0])
  // As we are not guaranteed  that int is always equivalent of int64_t or
  // int32_t in Go. Hence we are passing `long long` to C++.
  C.mlpackSetParamVectorInt(p.mem, C.CString(identifier),
      (*C.longlong)(ptr), C.size_t(len(vecInt)))
}

func setParamVecString(p *params, identifier string, vecString []string) {
  C.mlpackSetParamVectorStrLen(p.mem, C.CString(identifier),
      C.size_t(len(vecString)))
  for i := 0; i < len(vecString); i++ {
    C.mlpackSetParamVectorStr(p.mem, C.CString(identifier),
        (C.CString)(vecString[i]), C.size_t(i))
  }
}

func getParamVecInt(p *params, identifier string) []int {
  e := int(C.mlpackVecIntSize(p.mem, C.CString(identifier)))

  var v mlpackVectorType
  v.allocVecIntPtr(p, identifier)

  data := (*[1<<30 - 1]int)(v.mem)
  output := data[:e]
  if output != nil {
    return output
  }
  return []int{}
}

func getParamVecString(p *params, identifier string) []string {
  e := int(C.mlpackVecStringSize(p.mem, C.CString(identifier)))

  data := make([]string, e)
  for i := 0; i < e; i++ {
    data[i] = C.GoString(C.mlpackGetVecStringPtr(p.mem,
        C.CString(identifier), C.size_t(i)))
  }
  return data
}
