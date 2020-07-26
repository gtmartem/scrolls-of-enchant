package main

import "fmt"

const (
	A = iota
	B = iota
	_
	C = iota
)

const (
	D, E, F = iota, iota, iota
)

const G = iota
const H = iota

func main() {
	// Так как iota реагирует на каждый const, предоставляя последовательность не типизированных
	// целочисленных констант, для D, E, F значения будут 0, так как вычелслено будет 1 раз
	// каждое значение iota в кортеже (аналогично с G и H).
	fmt.Println(A, B, C)
	fmt.Println(D, E, F)
	fmt.Println(G, H)
}