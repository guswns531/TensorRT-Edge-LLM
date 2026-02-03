package main

/*
#cgo LDFLAGS: -L${SRCDIR}/../build/cpp -ledge_llm_runtime -lstdc++
#include "../cpp/runtime/c_api.h"
#include <stdlib.h>

extern void goStreamGateway(char* token, bool is_finished, void* ctx);
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
	"sync"
	"sync/atomic"
)

type Manager struct {
	handle C.EdgeLLMManagerHandle
}

func NewManager(engineDir string, numWorkers int) (*Manager, error) {
	cDir := C.CString(engineDir)
	defer C.free(unsafe.Pointer(cDir))

	handle := C.EdgeLLMManagerCreate(cDir, C.int(numWorkers))
	if handle == nil {
		return nil, errors.New("failed to create EdgeLLMManager")
	}
	return &Manager{handle: handle}, nil
}

func (m *Manager) Close() {
	if m.handle != nil {
		C.EdgeLLMManagerDestroy(m.handle)
		m.handle = nil
	}
}

type Request struct {
	Prompt       string
	MaxNewTokens int
	Temperature  float32
	TopP         float32
	TopK         int
}

type Response struct {
	Text      string
	NumTokens int
}

// Callback management for C -> Go bridge
var (
	callbackMap  sync.Map
	callbackNextID int64
)

//export goStreamGateway
func goStreamGateway(token *C.char, isFinished C.bool, ctx unsafe.Pointer) {
	id := int64(uintptr(ctx))
	if val, ok := callbackMap.Load(id); ok {
		cb := val.(func(string, bool))
		cb(C.GoString(token), bool(isFinished))
	}
}

func (m *Manager) Infer(workerIdx int, req Request) (Response, error) {
	cPrompt := C.CString(req.Prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	cReq := C.EdgeLLMRequest{
		prompt:         cPrompt,
		max_new_tokens: C.int32_t(req.MaxNewTokens),
		temperature:    C.float(req.Temperature),
		top_p:          C.float(req.TopP),
		top_k:          C.int32_t(req.TopK),
		stream_output:  false,
	}

	var cRes C.EdgeLLMResponse
	// Pass nil callback for blocking mode
	success := C.EdgeLLMManagerInfer(m.handle, C.int(workerIdx), &cReq, &cRes, nil, nil)
	if !bool(success) {
		return Response{}, fmt.Errorf("inference failed on worker %d", workerIdx)
	}
	defer C.EdgeLLMFreeResponse(&cRes)

	return Response{
		Text:      C.GoString(cRes.text),
		NumTokens: int(cRes.num_tokens),
	}, nil
}

// InferStream executes inference and calls the provided callback for each token
func (m *Manager) InferStream(workerIdx int, req Request, onToken func(string, bool)) (Response, error) {
	cPrompt := C.CString(req.Prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	cReq := C.EdgeLLMRequest{
		prompt:         cPrompt,
		max_new_tokens: C.int32_t(req.MaxNewTokens),
		temperature:    C.float(req.Temperature),
		top_p:          C.float(req.TopP),
		top_k:          C.int32_t(req.TopK),
		stream_output:  true,
	}

	// Register callback
	id := atomic.AddInt64(&callbackNextID, 1)
	callbackMap.Store(id, onToken)
	defer callbackMap.Delete(id)

	var cRes C.EdgeLLMResponse
	
	// Pass gateway function and ID
	success := C.EdgeLLMManagerInfer(m.handle, C.int(workerIdx), &cReq, &cRes, (C.EdgeLLMStreamCallback)(C.goStreamGateway), unsafe.Pointer(uintptr(id)))
	
	if !bool(success) {
		return Response{}, fmt.Errorf("inference failed on worker %d", workerIdx)
	}
	defer C.EdgeLLMFreeResponse(&cRes)

	return Response{
		Text:      C.GoString(cRes.text),
		NumTokens: int(cRes.num_tokens),
	}, nil
}
