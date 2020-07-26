package main

import "fmt"

type Counter struct {
	value int
}

func main() {
	// В a будет попадать копия элементов из arrayOfCounters,
	// поэтому a.value = i + 2 не поменяют исходный слайс. Кроме того, в res окажутся указатели на
	// последнее значение a, полученное в результате итерирования.
	var res = make([]*Counter, 3)
	arrayOfCounters := []Counter{{1}, {2}, {3}}
	for i, a := range arrayOfCounters {
		a.value = i + 2
		res[i] = &a
	}
	fmt.Println("res:", res[0].value, res[1].value, res[2].value)
	fmt.Println("arrayOfCounters:", arrayOfCounters)
}
