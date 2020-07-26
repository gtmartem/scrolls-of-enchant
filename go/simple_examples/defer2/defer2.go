package main

import "fmt"

func main() {
	// вызов defer кладется в стек
	for _, i := range [5]int{1,2,3,4,5} {
		defer func(i int) {
			fmt.Println(i)
		}(i)
	}
}
