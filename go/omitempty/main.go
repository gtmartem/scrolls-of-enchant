package main

import (
	"encoding/json"
	"fmt"
)

type Dog struct {
	Breed		string
	WeightKg	int
}

type DogWithOmitempty struct {
	Breed		string
	WeightKg	int			`json:",omitempty"`
}

type DogWithEmbeddedStruct struct {
	Breed		string
	WeightKg	int			`json:",omitempty"`
	Size		dimension	`json:",omitempty"`
}

type dimension struct {
	Height	int
	Width	int
}

type DogWithEmbeddedStructV2 struct {
	Breed		string
	WeightKg	int						`json:",omitempty"`
	Size		dimensionWithOmitempty	`json:",omitempty"`
}

type dimensionWithOmitempty struct {
	Height	int		`json:",omitempty"`
	Width	int		`json:",omitempty"`
}

type DogWithPointerOnEmbeddedStruct struct {
	Breed		string
	WeightKg	int			`json:",omitempty"`
	Size		*dimension	`json:",omitempty"`
}

type Restaurant struct {
	NumberOfCustomers *int `json:",omitempty"`
}

type Response struct {
	Result json.RawMessage `json:"result"`
}

func main() {
	// Print fulfilled struct marshalled to JSON:
	d := Dog{
		Breed: 		"dalmation",
		WeightKg: 	45,
	}
	b, _ := json.Marshal(d)
	fmt.Println(string(b))

	// Now, lets print dog without weight:
	d = Dog{Breed: "pug"}
	b, _ = json.Marshal(d)
	fmt.Println(string(b))

	// Let’s add the omitempty tag to our DogWithOmitempty struct and print it:
	// Note: The same will happen if a string is empty "",
	// or if a pointer is nil,
	// or if a slice has zero elements in it.
	dwo := DogWithOmitempty{Breed: "pug"}
	b, _ = json.Marshal(dwo)
	fmt.Println(string(b))

	// In cases where an empty value does not exist, omitempty is of no use.
	// An embedded struct, for example, does not have an empty value.
	// In this case, even though we never set the value of the Size attribute,
	// and set its omitempty tag, it still appears in the output.
	// This is because structs do not have any empty value in Go.
	dwes := DogWithEmbeddedStruct{Breed: "pug"}
	b, _ = json.Marshal(dwes)
	fmt.Println(string(b))

	// Now, see what happen if dimension struct will be with omitempty fields:
	dwesv2 := DogWithEmbeddedStructV2{Breed: "pug"}
	b, _ = json.Marshal(dwesv2)
	fmt.Println(string(b))

	// To solve problem with omitempty tag in embedded structs, use struct pointer instead:
	dwpoes := DogWithPointerOnEmbeddedStruct{Breed: "pug"}
	b, _ = json.Marshal(dwpoes)
	fmt.Println(string(b))

	// If we don't want 0 or "" to be the 'empty' value, use pointers instead:
	noc := 0
	r := Restaurant{NumberOfCustomers: &noc}
	b, _ = json.Marshal(r)
	fmt.Println(string(b))

	// json.RawMessage for null Marshalling:
	resp := Response{}
	b, _ = json.Marshal(resp)
	fmt.Println(string(b))
}


