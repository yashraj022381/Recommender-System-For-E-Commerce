PYTHON ?= python
S ?=

.PHONY: help install data data-real build evaluate app test test-fast all clean freeze

help:
	@echo Targets: install data data-real build evaluate app test test-fast all clean freeze

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

data:            # synthetic data, about 10 seconds
	$(PYTHON) $(S)prepare_data.py --synthetic

data-real:       # real Amazon reviews, downloads a few hundred MB
	$(PYTHON) $(S)prepare_data.py

build:           # fit every model, save artifacts/bundle.pkl (4-6 minutes)
	$(PYTHON) $(S)build_artifacts.py

evaluate:        # artifacts/leaderboard.csv (2-4 minutes)
	$(PYTHON) $(S)evaluate_all.py

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py

test:
	$(PYTHON) -m pytest -q

test-fast:       # skips the slow app tests
	$(PYTHON) -m pytest -q --ignore=tests/test_app.py --ignore=tests/test_app_interactions.py

all: data build evaluate test

clean:           # removes only the saved models, not your data
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('artifacts').glob('*.pkl')]"

freeze:
	$(PYTHON) -m pip freeze > requirements-lock.txt