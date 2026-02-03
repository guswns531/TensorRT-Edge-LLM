package main

import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"
)

func main() {
	engineDir := "/workspace/engines/qwen3-0.6b"
	numWorkers := 4
	
	manager, err := NewManager(engineDir, numWorkers)
	if err != nil {
		log.Fatalf("Failed to initialize manager: %v", err)
	}
	defer manager.Close()
	
	// Semaphore to manage available worker indices
	workerPool := make(chan int, numWorkers)
	for i := 0; i < numWorkers; i++ {
		workerPool <- i
	}

	// Mock high-concurrency requests
	var wg sync.WaitGroup
	numRequests := 100

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(reqID int) {
			defer wg.Done()

			// Lock this goroutine to an OS thread for CUDA context stability
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()

			start := time.Now()

			// 1. Acquire worker from pool
			workerIdx := <-workerPool
			defer func() { workerPool <- workerIdx }()

			acquired := time.Now()

			// 2. Inference
			req := Request{
				Prompt:       "What is the capital of United States?",
				MaxNewTokens: 50,
				Temperature:  0.7,
				TopP:         0.9,
				TopK:         40,
			}

			var fullText string
			var ttft time.Duration
			var tokenCount int
			var totalInterTokenDuration time.Duration
			lastTokenTime := acquired

			res, err := manager.InferStream(workerIdx, req, func(token string, done bool) {
				now := time.Now()
				if tokenCount == 0 {
					ttft = now.Sub(acquired)
				} else {
					totalInterTokenDuration += now.Sub(lastTokenTime)
				}
				lastTokenTime = now
				tokenCount++
				
				// fmt.Print(token)
				fullText += token
			})

			if err != nil {
				log.Printf("[Req %d] Error: %v", reqID, err)
				return
			}
			fmt.Println("") // Newline after complete

			done := time.Now()
			
			avgITL := time.Duration(0)
			if tokenCount > 1 {
				avgITL = totalInterTokenDuration / time.Duration(tokenCount-1)
			}

			fmt.Printf("[Req %d] Worker %d | Queue: %v | TTFT: %v | Avg ITL: %v | Total: %v | Tokens: %d\n",
				reqID, workerIdx, acquired.Sub(start), ttft, avgITL, done.Sub(start), res.NumTokens)

		}(i)
	}

	wg.Wait()
}
