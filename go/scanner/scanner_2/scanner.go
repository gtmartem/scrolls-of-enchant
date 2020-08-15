package main

import (
	"bufio"
	"fmt"
	"os"
	"os/signal"
)

func main() {

	ch := make(chan os.Signal, 1)
	signal.Notify(ch, os.Interrupt)

	go func() {
		<- ch
		fmt.Println("Exit.")
		os.Exit(1)
	}()

	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		txt := scanner.Text()
		if txt == "exit" {
			break
		}
		fmt.Println(txt)
	}
	fmt.Println("End.")
}