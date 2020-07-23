package main

import (
	"encoding/xml"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
)

const THEGUARDIANRSS = "https://www.theguardian.com/world/americas/rss"

func main() {
	resp, err := http.Get(THEGUARDIANRSS)
	if err != nil {
		log.Fatalf("error during GET %s, error: %s", THEGUARDIANRSS, err)
	}
	body, err := ioutil.ReadAll(resp.Body)
	defer func() {
		if err := resp.Body.Close(); err != nil {
			log.Fatalf("error during closing body of GET response from: %s, error: %s",
				THEGUARDIANRSS, err)
		}
	}()
	if err != nil {
		log.Fatalf("error during reading body of GET response from: %s, error: %s",
			THEGUARDIANRSS, err)
	}
	rss := &RSS{}
	err = xml.Unmarshal(body, rss)
	if err != nil {
		log.Fatalf("errror during unmarshalling body, error: %s", err)
	}
	fmt.Println(rss)
}
