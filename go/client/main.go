package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
)

type X struct {
	EventType	string 	`json:"eventType"`
}

func main() {
	uploadBuff := make([]interface{}, 2)
	uploadBuff[0] = X {
		EventType: "AppTest_1",
	}
	uploadBuff[1] = X {
		EventType: "AppTest_1",
	}
	_ = PostEvent(uploadBuff)
}

func PostEvent(chunk []interface{}) (err error) {
	c := http.Client{}
	body, err := json.Marshal(chunk)
	if err != nil {
		log.Fatal("Marshaling error")
	}
	req, err := http.NewRequest("POST", "https://httpbin.org/post", bytes.NewBuffer(body))
	if err != nil {
		err = fmt.Errorf("error during creating request: %w", err)
		return
	}
	resp, err := c.Do(req)
	if err != nil {
		err = fmt.Errorf("error during making request to offers api: %w", err)
		return
	}
	fmt.Println(resp.Status)
	return
}