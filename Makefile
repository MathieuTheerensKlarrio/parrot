## Makefile: activate virtualenv if present
# Usage: `make activate`

.PHONY: activate
activate:
	@if [ -d venv ]; then \
		if [ -f venv/Scripts/Activate.ps1 ]; then \
			echo "Launching PowerShell with venv activated..."; \
			powershell -NoExit -ExecutionPolicy Bypass -Command "& { . '$(CURDIR)\\venv\\Scripts\\Activate.ps1' }"; \
		elif [ -f venv/Scripts/activate ]; then \
			echo "Sourcing venv/Scripts/activate in the current shell..."; \
			. venv/Scripts/activate; \
			exec $$SHELL; \
		else \
			echo "No activation script found in venv/Scripts"; \
		fi; \
	else \
		echo "Virtualenv not found at venv. Create one with: python -m venv venv"; \
	fi
