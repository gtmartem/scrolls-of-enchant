package main

import (
	"context"
	"log"
	"net"
	"os"
	"os/signal"
	"time"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	go handleSignals(cancel)
	if err := startServer(ctx); err != nil {
		log.Fatal(err)
	}
}

func handleSignals(cancel context.CancelFunc) {
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)
	for {
		sig := <- sigCh
		switch sig {
		case os.Interrupt:
			cancel()
			return
		}
	}
}

func startServer(ctx context.Context) error {
	listenerAddr, err := net.ResolveTCPAddr("tcp", ":8080")
	if err != nil {
		return err
	}
	listener, err := net.ListenTCP("tcp", listenerAddr)
	if err != nil {
		return err
	}
	defer func() {
		if err := listener.Close(); err != nil {
			log.Println(err)
		}
	}()
	for {
		select {
		case <- ctx.Done():
			log.Println("server shutdown")
			return nil
		default:
			if err := listener.SetDeadline(time.Now().Add(time.Second)); err != nil {
				return err
			}
			_, err := listener.Accept()
			if err != nil {
				if os.IsTimeout(err) {
					continue
				}
				return err
			}
			log.Println("new client connected")
		}
	}
}
