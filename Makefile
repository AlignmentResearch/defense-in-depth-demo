APPLICATION_URL ?= ghcr.io/alignmentresearch/robust-llm
RELEASE_PREFIX ?= 2025-05-20-22-04-43-oskar-transformers-4.52
COMMIT_HASH ?= $(shell git rev-parse HEAD)
BRANCH_NAME ?= $(shell git branch --show-current)
CPU ?= 4
MEMORY ?= 60G
SHM_SIZE ?= 4Gi
GPU ?= 1
# e.g., set to "mig-2g.20gb" to request a 20GB GPU slice
GPU_RESOURCE ?= "gpu"
SLEEP_TIME ?= 2d
USER ?= $(shell whoami)
DEVBOX_NAME ?= rllm-devbox-$(USER)
YAML_FILE ?= k8s/auto-devbox.yaml
KUBECTL_OPTIONS ?=
KANIKO_PRIORITY ?= cpu-normal-batch

.PHONY: devbox devbox/% large cpu ian tom oskar image dry-run

# Make and launch a devbox
devbox/%:
	git push
	python -c "print(open('${YAML_FILE}').read().format( \
                NAME='${DEVBOX_NAME}', \
                USER='${USER}', \
                IMAGE='${APPLICATION_URL}:${RELEASE_PREFIX}', \
                COMMIT_HASH='${COMMIT_HASH}', \
                CPU='${CPU}', \
                MEMORY='${MEMORY}', \
                SHM_SIZE='${SHM_SIZE}', \
                GPU='${GPU}', \
                GPU_RESOURCE='${GPU_RESOURCE}', \
                SLEEP_TIME='${SLEEP_TIME}', \
        ))" | kubectl create -f - ${KUBECTL_OPTIONS}

large:
	$(eval CPU := 8)
	$(eval MEMORY := 100G)
	$(eval GPU := 2)
	$(eval DEVBOX_NAME := $(DEVBOX_NAME)-large)

cpu:
	$(eval CPU := 4)
	$(eval MEMORY := 48G)
	$(eval GPU := 0)
	$(eval DEVBOX_NAME := $(DEVBOX_NAME)-cpu)

mig-10gb:
	$(eval GPU := 1)
	$(eval GPU_RESOURCE := mig-1g.10gb)
	$(eval DEVBOX_NAME := $(DEVBOX_NAME)-mig-10gb)

mig-20gb:
	$(eval GPU := 1)
	$(eval GPU_RESOURCE := mig-2g.20gb)
	$(eval DEVBOX_NAME := $(DEVBOX_NAME)-mig-20gb)

ian:
	$(eval YAML_FILE := k8s/auto-devbox-ian.yaml)

tom:
	$(eval CPU := 1)
	$(eval YAML_FILE := k8s/auto-devbox-tom.yaml)

oskar:
	$(eval YAML_FILE := k8s/auto-devbox-oskar.yaml)

aaron:
	$(eval YAML_FILE := k8s/auto-devbox-aaron.yaml)

devbox: devbox/main

# Build the Docker image for a specific branch
image:
	python -c "print(open('k8s/kaniko-build.yaml').read().format( \
                BRANCH_NAME='${BRANCH_NAME}', \
				PRIORITY_CLASS_NAME='${KANIKO_PRIORITY}', \
        ))" | kubectl create -f - ${KUBECTL_OPTIONS}

dry-run:
	$(eval KUBECTL_OPTIONS := --dry-run=client -o yaml)
