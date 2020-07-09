package main

import (
	"encoding/json"
	"fmt"
	"log"
	"reflect"
	"time"
)

const tagName = "custom"

type CK struct {
	// fields AS IS
	FieldOne	string		`json:"fieldOne"`
	FieldTwo	string		`json:"fieldTwo"`

	// fields for array
	FieldThree	string		`json:"fieldThree" custom:"fieldThree"`
	FieldFour	string		`json:"fieldFour" custom:"fieldFour"`
	FieldFive	int			`json:"fieldFive" custom:"fieldFive"`
	FieldSix	float64		`json:"fieldSix" custom:"fieldSix"`
	FieldSeven	interface{}	`json:"FieldSeven" custom:"FieldSeven"`
}

func (ck *CK) marshalToCLCK() (clck *CLCK, err error) {
	tmpType := reflect.TypeOf(*ck)
	tmpValue := reflect.ValueOf(ck).Elem()
	clck = &CLCK{
		FieldOne:    ck.FieldOne,
		FieldTwo:    ck.FieldTwo,
		CustomName:  nil,
		CustomValue: nil,
	}
	for i := 0; i < tmpType.NumField(); i++ {
		// Get the field, returns https://golang.org/pkg/reflect/#StructField
		field := tmpType.Field(i)
		// Get the field tag value
		tag := field.Tag.Get(tagName)
		fmt.Printf("%d. %v (%v), tag: '%v'\n", i+1, field.Name, field.Type.Name(), tag)
		if tag != "" {
			v := tmpValue.Field(i).Interface()
			switch v.(type) {
			case time.Time:
				clck.CustomValue = append(clck.CustomValue, v.(time.Time).String())
			case string:
				clck.CustomValue = append(clck.CustomValue, v.(string))
			default:
				bts, err := json.Marshal(tmpValue.Field(i).Interface())
				if err != nil {
					continue
				}
				if string(bts) == "\"\"" {
					clck.CustomValue = append(clck.CustomValue, "")
				} else {
					clck.CustomValue = append(clck.CustomValue, string(bts))
				}
			}
			clck.CustomName = append(clck.CustomName, tag)
		}
	}
	return
}

type CLCK struct {
	// fields AS IS
	FieldOne	string		`json:"fieldOne"`
	FieldTwo	string		`json:"fieldTwo"`

	// array fields
	CustomName	[]string	`json:"customName"`
	CustomValue	[]string	`json:"customValue"`
}

func main() {
	ck := CK {
		FieldOne: 	"field_one",
		FieldTwo:	"field_two",
		FieldThree:	"field_three",
		FieldFour:	"field_four",
		FieldFive:	1,
		FieldSix:	6.555,
		FieldSeven: []int{1,2,3,4,5},
	}

	// Get info about ck from refect
	clck, err := ck.marshalToCLCK()
	if err != nil {
		log.Fatalf("error during marshalling to clck: %s", err.Error())
	}

	// Print JSON presentation of clck
	jsonck, err := json.Marshal(clck)
	if err != nil {
		log.Fatal("error during marshalling ck struct to json")
	}
	fmt.Println(string(jsonck))
}
