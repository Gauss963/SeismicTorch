PYTHON ?= python3
TECTONIC ?= tectonic

.PHONY: all figures report verify refresh replay sensitivity continuous
all: verify

figures:
	$(PYTHON) tools/make_figures.py
	$(PYTHON) tools/make_extended_figures.py

report: figures
	$(TECTONIC) Project_Report.tex

verify: report
	$(PYTHON) tools/verify_report.py

# These targets need the original local waveform archives; replay also needs MPS.
refresh:
	$(PYTHON) tools/collect_evidence.py

replay:
	$(PYTHON) tools/reproduce_training.py

sensitivity:
	$(PYTHON) tools/run_sensitivity.py

continuous:
	$(PYTHON) tools/collect_continuous.py
