package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"flag"
	"sort"
)

// Config
var (
	EngineDir           string
    MultimodalEngineDir string
	NumWorkers          int
	ListenAddress       string
)

var (
    // Max requests per batch per worker. 
    // This is adjustable via /benchmark?batch_size=N, up to 4 (engine limit).
    CurrentBatchSize int64 = 4
)

// RequestJob holds the request and the channel to send tokens back to the handler
type RequestJob struct {
	Req      Request
	Response chan string // Channel to stream tokens. Closed when done.
	ErrCh    chan error
}

var (
	requestQueue = make(chan *RequestJob, 1000)
)

func main() {
	flag.StringVar(&EngineDir, "engine-dir", "/workspace/engines/qwen3-0.6b", "Path to engine directory")
    flag.StringVar(&MultimodalEngineDir, "multimodal-engine-dir", "", "Path to multimodal engine directory (optional)")
	flag.IntVar(&NumWorkers, "workers", 4, "Number of concurrent TRT-LLM runtime instances")
	flag.StringVar(&ListenAddress, "addr", ":8000", "Address to listen on")
	flag.Parse()
	// Initialize Engine Manager
	manager, err := NewManager(EngineDir, MultimodalEngineDir, NumWorkers)
	if err != nil {
		log.Fatalf("Failed to initialize manager: %v", err)
	}
	defer manager.Close()

	batchJobQueue := make(chan []*RequestJob, NumWorkers)

	// Batcher Routine
	go func() {
		var batch []*RequestJob
		// Flush batch if it gets full or timeout
		ticker := time.NewTicker(time.Millisecond * 10)
		defer ticker.Stop()

		for {
			// Read current limit
			threshold := int(atomic.LoadInt64(&CurrentBatchSize))

			select {
			case job := <-requestQueue:
				batch = append(batch, job)
				if len(batch) >= threshold {
					batchJobQueue <- batch
					batch = nil
				}
			case <-ticker.C:
				if len(batch) > 0 {
					batchJobQueue <- batch
					batch = nil
				}
			}
		}
	}()

	// Worker Routines
	for i := 0; i < NumWorkers; i++ {
		go func(workerIdx int) {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()

			for batch := range batchJobQueue {
				processBatch(manager, workerIdx, batch)
			}
		}(i)
	}

	// HTTP Server
	http.HandleFunc("/generate", handleGenerate)
	http.HandleFunc("/benchmark", handleBenchmark) 

	log.Printf("Starting server on %s with %d workers", ListenAddress, NumWorkers)
	if MultimodalEngineDir != "" {
		log.Printf("Multimodal support enabled with engine: %s", MultimodalEngineDir)
	}
	log.Fatal(http.ListenAndServe(ListenAddress, nil))
}

func processBatch(manager *Manager, workerIdx int, batch []*RequestJob) {
	if len(batch) == 0 {
		return
	}

	reqs := make([]Request, len(batch))
	for i, job := range batch {
		reqs[i] = job.Req
	}

	// Callback handles routing tokens to correct job
	onToken := func(batchIdx int, token string, done bool) {
		if batchIdx >= 0 && batchIdx < len(batch) {
			if !done {
			    batch[batchIdx].Response <- token
			}
		}
	}

	_, err := manager.InferBatchStream(workerIdx, reqs, onToken)
	
	// Close all channels
	for _, job := range batch {
		if err != nil {
				job.ErrCh <- err
		}
		close(job.Response)
		close(job.ErrCh)
	}
}

// HTTP Handler Structures
type MessageContent struct {
    Type      string `json:"type"`      // "text" or "image"
    Text      string `json:"text,omitempty"`
    Image     string `json:"image,omitempty"` // For simple image field
    ImagePath string `json:"media_path,omitempty"` // Matches user example
    Data      string `json:"data,omitempty"` // Fallback
}

type Message struct {
    Role    string           `json:"role"`
    Content []MessageContent `json:"content"`
}

type APIGenRequest struct {
	Prompt      string       `json:"prompt"` // Legacy simple prompt
    Messages    []Message    `json:"messages"` // Structured chat input
	MaxTokens   int          `json:"max_tokens"`
	Temperature float32      `json:"temperature"`
}

func handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var apiReq APIGenRequest
	if err := json.NewDecoder(r.Body).Decode(&apiReq); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}
    
    // Set defaults
	if apiReq.MaxTokens == 0 { apiReq.MaxTokens = 50 }
	if apiReq.Temperature == 0 { apiReq.Temperature = 0.7 }

    var contents []Content

    // 1. Handle Messages (VLM/Chat style)
    if len(apiReq.Messages) > 0 {
        // Flatten user messages content
        for _, msg := range apiReq.Messages {
            if msg.Role == "user" {
                for _, c := range msg.Content {
                    if c.Type == "text" {
                        text := c.Text
                        if text == "" { text = c.Data }
                         contents = append(contents, Content{Type: "text", Data: text})
                    } else if c.Type == "image" {
                        path := c.ImagePath
                        if path == "" { path = c.Image }
                        if path == "" { path = c.Data }
                        contents = append(contents, Content{Type: "image", Data: path})
                    }
                }
            }
        }
    } else {
        // 2. Handle Legacy Prompt
        contents = append(contents, Content{Type: "text", Data: apiReq.Prompt})
    }
    
    if len(contents) == 0 {
        http.Error(w, "No content provided", http.StatusBadRequest)
        return
    }

	job := &RequestJob{
		Req: Request{
			Contents:     contents,
			MaxNewTokens: apiReq.MaxTokens,
			Temperature:  apiReq.Temperature,
			TopP:         0.9,
			TopK:         40,
		},
		Response: make(chan string, 100), // Buffer slightly
		ErrCh:    make(chan error, 1),
	}

    // Enqueue
	requestQueue <- job

	// Stream Response
	w.Header().Set("Content-Type", "text/plain") // Or text/event-stream
	w.Header().Set("Transfer-Encoding", "chunked")
    
	// Flush immediately to establish connection
	if f, ok := w.(http.Flusher); ok {
			f.Flush()
	}
	
	// Wait for tokens or error
	for {
		select {
		case err := <-job.ErrCh:
				if err != nil {
							fmt.Fprintf(w, "\n[Error: %v]\n", err)
				}
				return
		case token, ok := <-job.Response:
				if !ok {
						return // Channel closed, done
				}
				fmt.Fprint(w, token)
				if f, ok := w.(http.Flusher); ok {
						f.Flush()
				}
		}
	}
}

func handleBenchmark(w http.ResponseWriter, r *http.Request) {
	// Parse params
	concurrency := int(atomic.LoadInt64(&CurrentBatchSize)) // Default to current batch size or 4
	if concurrency == 0 { concurrency = 4 }
	numRequests := 20
	
	if c := r.URL.Query().Get("concurrency"); c != "" {
		fmt.Sscanf(c, "%d", &concurrency)
	}
	if n := r.URL.Query().Get("requests"); n != "" {
		fmt.Sscanf(n, "%d", &numRequests)
	}

	// Benchmark Content Config
	imagePath := r.URL.Query().Get("image")
	promptText := r.URL.Query().Get("prompt")
	if promptText == "" {
		if imagePath != "" {
			promptText = "Describe this image."
		} else {
			promptText = "What is the capital of France?"
		}
	}

	// Update Server Batch Size if requested
	if bs := r.URL.Query().Get("batch_size"); bs != "" {
			var val int64
			fmt.Sscanf(bs, "%d", &val)
			if val > 0 && val <= 4 {
						atomic.StoreInt64(&CurrentBatchSize, val)
						fmt.Fprintf(w, "Server Batch Size Set to %d\n", val)
			} else {
						fmt.Fprintf(w, "Invalid Batch Size %d (Max 4). Ignoring.\n", val)
			}
	}
	
	currentBS := atomic.LoadInt64(&CurrentBatchSize)

	w.Header().Set("Content-Type", "text/plain")
	fmt.Fprintf(w, "Starting Benchmark: Requests=%d, Concurrency=%d, BatchSizeLimit=%d, Image=%s\n", numRequests, concurrency, currentBS, imagePath)
	if f, ok := w.(http.Flusher); ok { f.Flush() }

	start := time.Now()
	var wg sync.WaitGroup
	
	// Metrics
	var totalTokens int64
	var mu sync.Mutex
	latencies := make([]time.Duration, numRequests)
	ttfts := make([]time.Duration, numRequests)
	itls := make([]time.Duration, numRequests) // Avg ITL per request
	tpots := make([]time.Duration, numRequests) // (Lat-TTFT)/Toks

	requestsPerWorker := numRequests / concurrency
	if requestsPerWorker == 0 { requestsPerWorker = 1 } // Safety
	
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < requestsPerWorker; j++ {
				// Determine index safely
				idx := workerID*requestsPerWorker + j
				if idx >= numRequests { break }

				reqStart := time.Now()
				
				// Construct Request Content
				var contents []Content
				if imagePath != "" {
					contents = append(contents, Content{Type: "image", Data: imagePath})
					contents = append(contents, Content{Type: "text", Data: promptText})
				} else {
					contents = append(contents, Content{Type: "text", Data: promptText})
				}

				job := &RequestJob{
					Req: Request{
						Contents:     contents,
						MaxNewTokens: 50,
						Temperature:  0.7,
						TopP:         0.9,
						TopK:         40,
					},
					Response: make(chan string, 100),
					ErrCh:    make(chan error, 1),
				}
				
				requestQueue <- job
				
				var firstTokenTime time.Time
				var lastTokenTime time.Time
				var tokens int
				var sumITL time.Duration
				
				for {
					select {
					case err := <-job.ErrCh:
						if err != nil {
							fmt.Printf("Error: %v\n", err)
						}
						return
					case _, ok := <-job.Response:
						if !ok {
							goto Done
						}
						now := time.Now()
						// First token
						if tokens == 0 {
							firstTokenTime = now
						} else {
							sumITL += now.Sub(lastTokenTime)
						}
						lastTokenTime = now
						tokens++
					}
				}
			// Done
			Done:
				reqDur := time.Since(reqStart)
				ttft := firstTokenTime.Sub(reqStart)
				
				var avgITL time.Duration
				if tokens > 1 {
					avgITL = sumITL / time.Duration(tokens-1)
				}
				
				// TPOT Calculation: Time Per Output Token.
				// Typically (EndToEnd - TTFT) / (GeneratedTokens - 1)
				// Or just use avgITL.
				// Let's explicitly calculate TPOT as (Lat - TTFT) / (Tok - 1)
				var tpot time.Duration
				if tokens > 1 {
					tpot = (reqDur - ttft) / time.Duration(tokens-1)
				}

				mu.Lock()
				totalTokens += int64(tokens)
				if idx < len(latencies) {
					latencies[idx] = reqDur
					ttfts[idx] = ttft
					itls[idx] = avgITL
					tpots[idx] = tpot
				}
				mu.Unlock()
			}
		}(i)
	}
	
	wg.Wait()
	totalDur := time.Since(start)
	
	// Collect valid data
	var validLat, validTTFT, validITL, validTPOT []time.Duration
	for k := 0; k < numRequests; k++ {
		if latencies[k] > 0 {
			validLat = append(validLat, latencies[k])
			validTTFT = append(validTTFT, ttfts[k])
			validITL = append(validITL, itls[k])
			validTPOT = append(validTPOT, tpots[k])
		}
	}

	totalTokensStr := fmt.Sprintf("%d", totalTokens)
	tps := float64(totalTokens) / totalDur.Seconds()
	
	avgLat, medLat, p99Lat := calcStats(validLat)
	avgTTFT, medTTFT, p99TTFT := calcStats(validTTFT)
	avgITL, medITL, p99ITL := calcStats(validITL)
	avgTPOT, medTPOT, p99TPOT := calcStats(validTPOT)

	report := fmt.Sprintf("\nBenchmark Complete:\n"+
		"Total Time: %v\n"+
		"Total Tokens: %s\n"+
		"Throughput: %.2f tokens/sec\n"+
		"Latency: Avg %v | Median %v | P99 %v\n"+
		"TTFT   : Avg %v | Median %v | P99 %v\n"+
		"ITL    : Avg %v | Median %v | P99 %v\n"+
		"TPOT   : Avg %v | Median %v | P99 %v\n",
		totalDur, totalTokensStr, tps, 
		avgLat, medLat, p99Lat,
		avgTTFT, medTTFT, p99TTFT,
		avgITL, medITL, p99ITL,
		avgTPOT, medTPOT, p99TPOT)
		
	fmt.Fprint(w, report)
	log.Print(report)
}

func calcStats(data []time.Duration) (avg, median, p99 time.Duration) {
	n := len(data)
	if n == 0 {
		return 0, 0, 0
	}
	
	// Calculate Avg
	var sum time.Duration
	for _, d := range data {
		sum += d
	}
	avg = sum / time.Duration(n)
	
	// Sort for percentiles
	sorted := make([]time.Duration, n)
	copy(sorted, data)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	
	median = sorted[n/2]
	
	p99Index := int(float64(n) * 0.99)
	if p99Index >= n { 
		p99Index = n - 1 
	}
	p99 = sorted[p99Index]
	
	return avg, median, p99
}
