package main

import "fmt"

var num int

func main() {
	// Зависит от того, сколько горутин успеют отработать и какая будет последней перед принтом.
	for i := 0; i < 1000; i++ {
		go func() {
			num = i
		}()
	}
	fmt.Printf("NUM is %d", num)
}