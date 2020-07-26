package main

import (
	"fmt"
)

func main() {
	// Так как defer замыкает значение во время объявления - i = 1 при выводе.
	i := 0
	i++
	defer fmt.Println(i)
	i++
	return
}
