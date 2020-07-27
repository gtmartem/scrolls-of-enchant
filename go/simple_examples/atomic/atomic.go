package main

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

const numRequests = 10000

var count int64

func networkRequest(wg *sync.WaitGroup) {
	time.Sleep(time.Millisecond)
	atomic.AddInt64(&count, 1)
	wg.Done()
}
func main() {
	wg := sync.WaitGroup{}
	wg.Add(numRequests)
	for i := 0; i < numRequests; i++ {
		go networkRequest(&wg)
	}
	wg.Wait()
	fmt.Println(count)
}