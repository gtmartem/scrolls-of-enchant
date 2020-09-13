package main

import (
	"fmt"
	"log"
	"net"
	"net/rpc"
	"net/rpc/jsonrpc"
	"strings"
)

type Listener int

type Reply struct {
	Data string
}

func (l *Listener) GetLine(line []byte, reply *Reply) error {
	rv := string(line)
	fmt.Printf("Receive: %v", rv)
	*reply = Reply{strings.TrimSuffix(rv, "\n")}
	return nil
}

func main() {
	addy, err := net.ResolveTCPAddr("tcp", "0.0.0.0:12345")
	if err != nil {
		log.Fatal(err)
	}

	inbound, err := net.ListenTCP("tcp", addy)
	if err != nil {
		log.Fatal(err)
	}

	listener := new(Listener)
	if err := rpc.Register(listener); err != nil {
		log.Fatal(err)
	}

	for {
		conn, err := inbound.Accept()
		if err != nil {
			log.Fatal(err)
		}
		jsonrpc.ServeConn(conn)
	}
}
