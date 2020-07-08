package main

import (
	"fmt"
	"reflect"
)

type I interface { m() }

// Type Assertion - r, ok := x.(Y).
// Если Y - не интерфесный тип, то проверяем, что Y реализует интерфейс x и присваеваем r тип string
// Если Y - интерфейс, то проверяем, реализует ли x интерфейс Y

func main() {
	// anyType - динамический тип
	var anyType interface{}
	anyType = "Canada"
	fmt.Println("Variable type:", reflect.TypeOf(anyType))
	// проверяем, что string, так как это не интерфейсный тип, реализует тип anyType, но так как
	// anyType - динамический тип, любой тип реализует его. Если утверждение верно,
	// значение anyType сохраняется в str, а типом str яыляется string.
	str, ok := anyType.(string)
	if ok {
		fmt.Println("Variable type:", reflect.TypeOf(str))
	} else {
		fmt.Println("Variable is not String.")
	}

	// intType имеет конкретный тип int
	var intType = 100
	// но anyType - interface{}
	// anyType = 100
	anyType = intType
	fmt.Println("Variable type:", reflect.TypeOf(anyType))
	integer, ok := anyType.(int)
	if ok {
		fmt.Println("Variable type:", reflect.TypeOf(integer))
	} else {
		fmt.Println("Variable is not Integer.")
	}

	// Проверяем, что anyType реализует интерфейс I
	i, ok := anyType.(I)
	if ok {
		fmt.Println("Variable type:", reflect.TypeOf(i))
	} else {
		fmt.Println("Variable does not implement interface I.")
		// i имеет тип nil
		fmt.Printf("Variable type: %v\n", reflect.TypeOf(i))
	}
}