# Change these variables to match your setup.
REMOTE_HOST ?= your-remote-host
REMOTE_DIR ?= /path/to/swebench
LOCAL_DIR ?= ../

push:
	rsync -avz --exclude="build_state.json" --exclude='.git' --exclude=".venv" --exclude="logs" --exclude="nv-SWE-Rebench-V2" ./ $(REMOTE_HOST):$(REMOTE_DIR)

pull:
	rsync -avz --exclude='.git' --exclude='.venv' --exclude='__pycache__' $(REMOTE_HOST):$(REMOTE_DIR)/ ./
