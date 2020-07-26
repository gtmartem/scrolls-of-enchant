package main

import (
	"encoding/json"
	"fmt"
)

type MyData struct {
	One int    `json:"one"`
	two string `json:"two"`
}

func main() {
	in := MyData{1, "two"}
	// main.MyData{One:1, two:"two"}
	fmt.Printf("%#v\n", in)
	encoded, _ := json.Marshal(in)

	// Так как MyData.two - неэкспортируемое поле, оно не попадет в json представление.
	// {"one":1}
	// encoded type: []uint8
	fmt.Println(string(encoded))
	fmt.Printf("encoded type: %T \n", encoded)

	var out MyData
	// main.MyData{One:1, two:""}
	err := json.Unmarshal(encoded, &out)
	if err != nil {
		panic(err)
	}

	fmt.Printf("%#v\n", out) // main.MyData{One:1, two:""}
}
