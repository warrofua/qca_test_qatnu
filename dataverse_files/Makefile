# ---- SRQID & QATNU reproducibility ----
# Usage:
#   make run_srqid   # SRQID numerics (LR velocity, no-signalling, energy drift)
#   make run_qatnu   # QATNU PoC (unitarity, light-cone, 1/k^2 tail, MERA, dchi~T)
#   make run_all     # both of the above
#   make run_nb      # execute the notebook if you include one at notebooks/srqid_numerics.ipynb
#   make clean       # remove venv and outputs

VENV_DIR := .venv
PY := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
REQS := requirements_qatn.txt

.PHONY: run_srqid run_qatnu run_all venv run_nb clean figures

run_all: run_srqid run_qatnu

# Backward-compat: 'make run' maps to SRQID only
run: run_srqid

venv: $(PY)
$(PY):
	python3 -m venv $(VENV_DIR)
	$(PIP) install -U pip
	$(PIP) install -r $(REQS)

run_srqid: venv
	mkdir -p outputs figures
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
	$(PY) src/srqid_numerics.py

run_qatnu: venv
	mkdir -p outputs figures
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
	$(PY) src/qatnu_poc.py

run_nb: venv
	mkdir -p outputs figures
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
	$(PY) -m jupyter nbconvert \
	    --to notebook --execute notebooks/srqid_numerics.ipynb \
	    --output outputs/srqid_numerics.executed.ipynb \
	    --ExecutePreprocessor.timeout=1200

figures:
	@echo "Figures are generated into ./figures by the scripts."

clean:
	rm -rf $(VENV_DIR) outputs figures
