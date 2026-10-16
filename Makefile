.PHONY: all install data run test clean
all: data run test
install:
	pip install -e ".[test]"
data:
	python -m cgpipe simulate -o data/example
run:
	python -m cgpipe run -c config/config.yaml
test:
	python -m pytest -q tests
clean:
	rm -rf results/example
