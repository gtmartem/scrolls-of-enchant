package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
)

func main() {
	done := make(chan bool)
	go func (done chan <- bool) {
		for {
			buffer, err := bufio.NewReader(os.Stdin).ReadBytes(byte('\n'))
			if err != nil {
				if err == io.EOF {
					done <- true
				}
				log.Fatal(err)
			}
			_, err = os.Stdout.Write(buffer)
			if err != nil {
				log.Fatal(err)
			}
		}
	}(done)
	<- done
	fmt.Println("EXIT, BB!")
}
