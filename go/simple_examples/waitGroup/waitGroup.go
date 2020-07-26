package main

import (
	"fmt"
	"sync"
)

func main() {
	// Тут будет ошибка рантайма, так как канал не буферизирован, поэтому wg.Wait()
	// будет ожидать вечно спящие горутины: читаем мы один раз.
	ch := make(chan int)
	wg := &sync.WaitGroup{}
	wg.Add(5)
	for i := 0; i < 5; i++ {
		go func(idx int) {
			ch <- (idx + 1) * 2
			wg.Done()
		}(i)
	}
	fmt.Printf("Result: %d\n", <-ch)
	wg.Wait()
}
