package main

import "fmt"

type MyError struct{}

func (MyError) Error() string { return "MyError!" }

func errorHandler(err error) {
	if err != nil {
		fmt.Println("Error:", err)
		fmt.Printf("Type: %T \n", err)
	}
}

func main() {
	// Создаем указатель на MyError, при этом MyError - nil, но указатель не nil,
	// поэтому принт в функции errorHandler будет.
	var err *MyError
	errorHandler(err)

	err = &MyError{}
	// Создаем структуру и получаем ее адрес, таким образом метод Error у структуры будет
	// отрабатывать, а у err будет тип указатель на MyError
	errorHandler(err)
}
