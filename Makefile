SHELL := /bin/bash

DOCKER_COMPOSE := docker compose

.PHONY: build up exec down

build:
	$(DOCKER_COMPOSE) build

up:
	$(DOCKER_COMPOSE) up -d
	$(DOCKER_COMPOSE) exec -T dino bash -c "cd models/dino/ops && python setup.py build install && python test.py"

exec:
	$(DOCKER_COMPOSE) exec dino bash

down:
	$(DOCKER_COMPOSE) down
