package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
)

func main() {
	err := readAndWrite()
	if err != nil {
		fmt.Println("readAndWrite execution err ", err)
	}
}

func readAndWrite() error {
	var buf bytes.Buffer
	pwd, _ := os.Getwd()
	dat, err := ioutil.ReadFile(pwd + "/go/buffer/buffer.txt")
	if err != nil {
		return err
	}
	_, err = buf.Write(dat)
	if err != nil {
		return err
	}
	buf.WriteString("\nnew string")
	f, err := os.Create(pwd + "/go/buffer/buffer_test_new.txt")
	if err != nil {
		return err
	}
	defer func () {
		err = f.Close()
		if err != nil {
			log.Print("err during closing file ", err)
		}
	}()
	_, err = f.Write(buf.Bytes())
	if err != nil {
		return err
	}
	return nil
}
