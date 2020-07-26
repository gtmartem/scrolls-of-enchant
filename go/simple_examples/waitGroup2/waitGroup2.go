package main

import (
	"fmt"
	"sync"
)

func main() {
	// Каждая го-рутина, порожденная циклом for выведет three, так как их выполнение начнется
	// после окончания цикла for. Можно добавить sleep для принудительного переключения горутин.
	wg := sync.WaitGroup{}
	data := []string{"one", "two", "three"}
	for _, v := range data {
		wg.Add(1)
		go func() {
			fmt.Println(v)
			wg.Done()
		}()
		//time.Sleep(time.Second)
	}
	wg.Wait()
}