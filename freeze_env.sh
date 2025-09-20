#!/bin/bash
# Freezes the current Python environment to a requirements.txt file.

echo "Freezing environment to requirements.txt..."
pip freeze > requirements.txt
echo "Done."
