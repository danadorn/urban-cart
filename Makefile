PYTHON := python
PIP := $(PYTHON) -m pip
REQ := requirements.txt

.PHONY: install run clean

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQ)

run:
	$(PYTHON) main.py

clean:
	rm -rf data/processed/* figures/*
	find . -type d -name "__pycache__" -print0 | xargs -0 rm -rf
