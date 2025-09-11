#!/usr/bin/env python3
"""
Simplify matplotlib configuration to avoid backend issues
"""

import json

# Read the current notebook
with open('trial_tracking_analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# Fix the matplotlib configuration cell - remove explicit backend setting
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'matplotlib.use(' in str(cell['source']):
        # Simplify to just use magic commands and basic configuration
        cell['source'] = [
            "# Jupyter magic commands for matplotlib display\n",
            "%matplotlib inline\n",
            "%config InlineBackend.figure_format = 'retina'\n",
            "\n",
            "# Configure matplotlib for optimal notebook display\n",
            "import matplotlib.pyplot as plt\n",
            "plt.rcParams['figure.max_open_warning'] = 0\n",
            "plt.rcParams['figure.dpi'] = 100\n",
            "plt.rcParams['figure.figsize'] = [12, 8]\n",
            "plt.rcParams['font.size'] = 10\n",
            "\n",
            "print(\"✅ Matplotlib configured for notebook display\")\n"
        ]
        break

# Write the fixed notebook
with open('trial_tracking_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✅ Simplified matplotlib configuration!")
print("✅ Removed explicit backend setting")
print("✅ Using only %matplotlib inline magic command")
print("✅ This should work reliably in Jupyter notebooks")
