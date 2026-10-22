# Runs the stack as the host user, so the files the containers create aren't owned by root (Linux and macOS)
export MLBOX_UID := $(shell id -u)
export MLBOX_GID := $(shell id -g)
export PWD := $(CURDIR)

.PHONY: build up down clean

build:
	docker compose build

# The socket's group as containers see it, which on Docker Desktop is the VM's, not the Mac's
up: build
	MLBOX_DOCKER_GID=$$(docker run --rm --entrypoint stat -v /var/run/docker.sock:/var/run/docker.sock mlbox:latest \
		-c %g /var/run/docker.sock) docker compose up -d
	@echo "MLflow: http://localhost:5001"
	@echo "API:    http://localhost:8080"
	@echo "Logs:   docker compose logs -f"

down:
	docker compose down

# Everything `up` created: the containers, the network, MLflow's data, the mlbox and pipeline images and the runs
clean:
	docker compose down --volumes --remove-orphans
	docker image rm $$(docker images --filter reference='mlbox*' --format '{{.Repository}}:{{.Tag}}') 2>/dev/null || true
	rm -rf .mlbox
