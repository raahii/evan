.PHONY: build

all: init test

init:
	pip install -r requirements.txt

test:
	python -m unittest discover -s tests

build:
	python setup.py bdist_wheel

clean:
	rm -rf build dist evan.egg-info

publish: clean build
	twine upload --repository pypi dist/*
