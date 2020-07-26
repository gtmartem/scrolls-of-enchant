package main

import (
	"sync"
)

func main() {
	// Так как i < 100, запуск хотя бы нескольких горутин весьма вероятен,
	// поэтому будет ошибка конкурентной записи в map.
	data := make(map[string]int)
	wg := sync.WaitGroup{}
	wg.Add(1)
	for i := 0; i < 100; i++ {
		go func(d map[string]int, num int) {
			d[string(num)] = num
		}(data, i)
	}
	wg.Done()
}