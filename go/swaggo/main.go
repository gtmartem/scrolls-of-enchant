// DOCS: https://github.com/swaggo/swag#declarative-comments-format

package main

import (
	"github.com/gorilla/mux"
	_ "github.com/gtmartem/scrolls-of-enchant/go/swaggo/docs"
	log "github.com/sirupsen/logrus"
	httpSwagger "github.com/swaggo/http-swagger"
	"net/http"
	"time"
)


// @title Swagger Example API
// @version 1.0
// @description This is swagger usage example.
// @termsOfService http://swagger.io/terms/

// @license.name Apache 2.0
// @license.url http://www.apache.org/licenses/LICENSE-2.0.html

// @host localhost:8080
// @BasePath /api

// @securitydefinitions NotExists
func main() {
	log.Println("start server ...")
	router := mux.NewRouter()
	router.PathPrefix("/api/swagger/").Handler(httpSwagger.Handler())
	router.HandleFunc("/api/ping", PingHandler).Methods("GET", "OPTIONS")
	// Create server
	srv := &http.Server{
		Addr:         "127.0.0.1:8080",
		Handler:      router,
		ReadTimeout:  time.Duration(5) * time.Second,
		WriteTimeout: time.Duration(5) * time.Second,
		IdleTimeout:  time.Duration(5) * time.Second,
	}
	log.Infoln("start server at", "127.0.0.1")
	log.Fatal(srv.ListenAndServe())
}

// PingHandler return pong
// @Summary ping example
// @Description do ping
// @Tags example
// @Produce plain
// @Success 200 {string} string "pong"
// @Router /ping [get]
func PingHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, err := w.Write([]byte("pong"))
	if err != nil {
		log.Error("error during pong response")
	}
}


