package main

import "fmt"

func main() {
	a := []int{1,2,3}
	b := []int{4,5,6}
	ref := a
	a = b
	// [4 5 6] [4 5 6] [1 2 3]
	fmt.Println(a, b, ref)

	// если нужно копировать слайс - используем copy
	x := []int{1,2,3}
	y := []int{4,5,6}
	refNew := x
	_ = copy(x, y)
	// [4 5 6] [4 5 6] [4 5 6]
	fmt.Println(x, y, refNew)
}
