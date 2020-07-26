package main

import (
"fmt"
"sync"
"time"
)

func main() {
	// Ну тут все очевидно: sleep дает планировщику мультиплексировать горутины,
	// поэтому вывод будет:
	// 2
	// 1
	// 3
	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		time.Sleep(time.Second * 2)
		fmt.Println("1")
		wg.Done()
	}()

	go func() {
		fmt.Println("2")
	}()

	wg.Wait()
	fmt.Println("3")
}
