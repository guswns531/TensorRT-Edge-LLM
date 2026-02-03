package main

/*
#cgo LDFLAGS: -L${SRCDIR}/../build/cpp -ledge_llm_runtime -lstdc++
#include "../cpp/runtime/c_api.h"
#include <stdlib.h>

extern void goStreamGateway(int batchIndex, char* token, bool is_finished, void* ctx);
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
func goStreamGateway(batchIndex C.int, token *C.char, isFinished C.bool, ctx unsafe.Pointer) {
	id := int64(uintptr(ctx))
	if val, ok := callbackMap.Load(id); ok {
		cb := val.(func(int, string, bool))
		cb(int(batchIndex), C.GoString(token), bool(isFinished))
	}
}

// InferStream executes inference and calls the provided callback for each token
func (m *Manager) InferStream(workerIdx int, req Request, onToken func(string, bool)) (Response, error) {
	// Wrapper to adapt single request to batch callback signature
	batchCb := func(idx int, token string, done bool) {
		onToken(token, done)
	}

	resps, err := m.InferBatchStream(workerIdx, []Request{req}, batchCb)
	if err != nil {
		return Response{}, err
	}
	if len(resps) == 0 {
		return Response{}, errors.New("no response")
	}
	return resps[0], nil
}

// InferBatchStream executes batch inference
func (m *Manager) InferBatchStream(workerIdx int, reqs []Request, onToken func(int, string, bool)) ([]Response, error) {
	if len(reqs) == 0 {
		return nil, nil
	}

	// Prepare C requests array
	cReqs := make([]C.EdgeLLMRequest, len(reqs))
	// Keep track of CStrings to free them
	cPrompts := make([]*C.char, len(reqs))
	
	for i, r := range reqs {
		cPrompts[i] = C.CString(r.Prompt)
		cReqs[i] = C.EdgeLLMRequest{
			prompt:         cPrompts[i],
			max_new_tokens: C.int32_t(r.MaxNewTokens),
			temperature:    C.float(r.Temperature),
			top_p:          C.float(r.TopP),
			top_k:          C.int32_t(r.TopK),
			stream_output:  true,
		}
	}

	defer func() {
		for _, p := range cPrompts {
			C.free(unsafe.Pointer(p))
		}
	}()

	// Register callback
	id := atomic.AddInt64(&callbackNextID, 1)
	callbackMap.Store(id, onToken)
	defer callbackMap.Delete(id)

	cResps := make([]C.EdgeLLMResponse, len(reqs))
	
	ptrReqs := (*C.EdgeLLMRequest)(unsafe.Pointer(&cReqs[0]))
	ptrResps := (*C.EdgeLLMResponse)(unsafe.Pointer(&cResps[0]))

	success := C.EdgeLLMManagerInferBatch(
		m.handle, 
		C.int(workerIdx), 
		ptrReqs, 
		C.int(len(reqs)), 
		ptrResps, 
		(C.EdgeLLMStreamCallback)(C.goStreamGateway), 
		unsafe.Pointer(uintptr(id)),
	)

	if !bool(success) {
		return nil, fmt.Errorf("inference failed on worker %d", workerIdx)
	}

	responses := make([]Response, len(reqs))
	for i := 0; i < len(reqs); i++ {
		// EdgeLLMResponse text needs to be freed? 
		// Yes, in C++ we strdup it. We should assume we need to free it.
		// But EdgeLLMFreeResponse only frees a single one. 
		// We should probably call cleanup manually or loop.
		// Wait, C-API `EdgeLLMFreeResponse` takes a pointer.
		// I should call it for each response.
		
		responses[i] = Response{
			Text:      C.GoString(cResps[i].text),
			NumTokens: int(cResps[i].num_tokens),
		}
		C.EdgeLLMFreeResponse(&cResps[i])
	}

	return responses, nil
}
