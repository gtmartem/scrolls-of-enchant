package main

import (
	"context"
	"fmt"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.mongodb.org/mongo-driver/mongo/readpref"
	"log"
	"time"
)

func main() {
	// run mongo container on localhost with -p 27017:27017
	client, err := mongo.NewClient(options.Client().ApplyURI("mongodb://localhost:27017"))
	if err != nil {
		log.Fatalf("error during creating mongo-client: %s", err.Error())
	}

	ctx, _ := context.WithTimeout(context.Background(), 10 * time.Second)
	err = client.Connect(ctx)
	defer client.Disconnect(context.Background())
	if err != nil {
		log.Fatalf("error during connection to mongo: %s", err.Error())
	}

	err = client.Ping(ctx, readpref.Primary())
	if err != nil {
		log.Fatalf("error during ping to mongo: %s", err.Error())
	}

	databases, err := client.ListDatabaseNames(ctx, bson.M{})
	if err != nil {
		log.Fatalf("error during getting list of databases in mongo: %s", err.Error())
	}
	fmt.Println(databases)
}