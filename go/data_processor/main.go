package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
)

type r string
type processor interface {
	process(data string)
	getVerbs() []r
}

type titleProcessor struct {
	verbs []r
}

func (tP *titleProcessor) process(data string) {
	fmt.Println(strings.Title(data))
}

func (tP *titleProcessor) getVerbs() []r {
	return tP.verbs
}

type toUpperProcessor struct {
	verbs []r
}

func (tUP *toUpperProcessor) process(data string) {
	fmt.Println(strings.ToUpper(data))
}

func (tUP *toUpperProcessor) getVerbs() []r {
	return tUP.verbs
}

func generateProcessors() (res []processor) {
	res = []processor{
		&titleProcessor{
			verbs: []r{"a", "b", "c"},
		},
		&toUpperProcessor{
			verbs: []r{"a", "x", "y"},
		},
	}
	return
}

func processorsToMap(processors []processor) map[r][]processor {
	res := make(map[r][]processor, len(processors))
	for _, processor := range processors {
		for _, v := range processor.getVerbs() {
			res[v] = append(res[v], processor)
		}
	}
	return res
}

type worker struct {
	processors map[r][]processor
}

func main() {
	reader := bufio.NewReader(os.Stdin)

	w := worker{processors: processorsToMap(generateProcessors())}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, os.Interrupt)
		<- sigCh
		cancel()
	}()

	for {
		select {
		case <- ctx.Done():
			log.Print("end data_processor")
			break
		default:
			text, _ := reader.ReadString('\n')
			verbs := strings.Split(text, " ")
			for _, verb := range verbs {
				processors := w.processors[r([]rune(verb)[0])]
				for _, proc := range processors {
					proc.process(verb)
				}
			}
		}
	}
}


