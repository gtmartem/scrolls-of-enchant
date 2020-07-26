package main

import (
"fmt"
)

func main() {
	// Поскольку в основе slice лежит массив, мы имеет доступ к элементам исходного массива даже
	// таким странным образом:

	defer func() {
		if r := recover(); r != nil {
			// runtime error: slice bounds out of range [:3] with capacity 2
			fmt.Println(r)
		}
	}()

	test1 := []int{1, 2, 3, 4, 5}
	test1 = test1[:3]
	// [1 2 3]
	fmt.Println(test1)
	test2 := test1[3:]
	// []
	fmt.Println(test2)
	// [4 5]
	fmt.Println(test2[:2])
	// PANIC!
	fmt.Println(test2[:3])
}
