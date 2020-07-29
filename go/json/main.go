package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
)

type Response struct {
	Success		bool	`json:"success"`
	Error		string	`json:"error"`
}

func main() {
	var bufEncoder bytes.Buffer
	JSONEncoder := json.NewEncoder(&bufEncoder)
	err := JSONEncoder.Encode(map[string]interface{}{
		"success": 	false,
		"error": 	"Failed to retrieve buckets",
	})
	if err != nil {
		log.Fatal(err)
	}
	JSONDecoder := json.NewDecoder(&bufEncoder)
	var resp Response
	err = JSONDecoder.Decode(&resp)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}


