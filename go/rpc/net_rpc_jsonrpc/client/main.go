package main

import (
	"bufio"
	"log"
	"net/rpc/jsonrpc"
	"os"
)

type Reply struct {
	Data string
}

func main() {
	client, err := jsonrpc.Dial("tcp", "localhost:12345")
	if err != nil {
		log.Fatal(err)
	}

	in := bufio.NewReader(os.Stdin)

	for {
		line, err := in.ReadBytes('\n')
		if err != nil {
			log.Fatal(err)
		}

		var rep Reply
		err = client.Call("Listener.GetLine", line, &rep)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Reply: %v, Data: %v", rep, rep.Data)
	}
}


