all: run

run:
	go run cmd/main.go

build:
	go build -o neural-network cmd/main.go
