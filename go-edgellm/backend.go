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
	"log"
	"unsafe"
	"sync"
	"sync/atomic"
)

type Manager struct {
	handle C.EdgeLLMManagerHandle
}

type Content struct {
    Type string // "text" or "image"
    Data string // text content or image path
}

func NewManager(engineDir string, multimodalEngineDir string, numWorkers int) (*Manager, error) {
	cDir := C.CString(engineDir)
	defer C.free(unsafe.Pointer(cDir))
    
    var cMmDir *C.char
    if multimodalEngineDir != "" {
        cMmDir = C.CString(multimodalEngineDir)
        defer C.free(unsafe.Pointer(cMmDir))
    }

	handle := C.EdgeLLMManagerCreate(cDir, cMmDir, C.int(numWorkers))
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
	Contents     []Content
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
    
    // We need to keep track of allocation to free them
    type ReqAlloc struct {
        contentsPtr *C.EdgeLLMContent // Pointer to C-allocated array
        dataPtrs    []*C.char         // Pointers to strings inside contents
        typePtrs    []*C.char         // Pointers to type strings
    }
    allocs := make([]ReqAlloc, len(reqs))

	for i, r := range reqs {
		if i > 0 {
			// Check for parameter consistency warnings
			r0 := reqs[0]
			if r.Temperature != r0.Temperature || r.TopP != r0.TopP || r.TopK != r0.TopK || r.MaxNewTokens != r0.MaxNewTokens {
				log.Printf("WARNING: Request %d has different sampling params than Request 0 in current batch. C++ runtime currently uses Request 0 params for the whole batch.", i)
			}
		}
        
        // Convert Go Contents to C Contents
        numContents := len(r.Contents)
        
        // Allocate C array for contents to avoid Go-pointer-to-Go-pointer violation
        var cContentsPtr *C.EdgeLLMContent
        if numContents > 0 {
            cContentsPtr = (*C.EdgeLLMContent)(C.calloc(C.size_t(numContents), C.size_t(unsafe.Sizeof(C.EdgeLLMContent{}))))
        }
        
        allocs[i].contentsPtr = cContentsPtr
        allocs[i].dataPtrs = make([]*C.char, numContents)
        allocs[i].typePtrs = make([]*C.char, numContents)
        
        // Use unsafe.Slice to access the C array as a Go slice for easy population
        // (safe because we just allocated it)
        var cContentsSlice []C.EdgeLLMContent
        if numContents > 0 {
             cContentsSlice = unsafe.Slice(cContentsPtr, numContents)
        }
        
        for j, c := range r.Contents {
            cType := C.CString(c.Type)
            cData := C.CString(c.Data)
            
            allocs[i].typePtrs[j] = cType
            allocs[i].dataPtrs[j] = cData
            
            cContentsSlice[j] = C.EdgeLLMContent{
                _type:    cType,
                data:     cData,
                data_len: C.int(len(c.Data)),
            }
        }

		cReqs[i] = C.EdgeLLMRequest{
			contents:       cContentsPtr,
            num_contents:   C.int(numContents),
			max_new_tokens: C.int32_t(r.MaxNewTokens),
			temperature:    C.float(r.Temperature),
			top_p:          C.float(r.TopP),
			top_k:          C.int32_t(r.TopK),
			stream_output:  true,
		}
	}

	defer func() {
		for _, a := range allocs {
            // Free the C array of structs
            if a.contentsPtr != nil {
                C.free(unsafe.Pointer(a.contentsPtr))
            }
            // Free the strings
            for _, p := range a.dataPtrs {
                C.free(unsafe.Pointer(p))
            }
            for _, p := range a.typePtrs {
                C.free(unsafe.Pointer(p))
            }
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
		responses[i] = Response{
			Text:      C.GoString(cResps[i].text),
			NumTokens: int(cResps[i].num_tokens),
		}
		// Free the C string allocated by the runtime
		C.EdgeLLMFreeResponse(&cResps[i])
	}

	return responses, nil
}
