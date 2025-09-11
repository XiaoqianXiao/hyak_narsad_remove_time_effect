#!/usr/bin/env python3
"""
Fix matplotlib backend error in the notebook
"""

import json

# Read the current notebook
with open('trial_tracking_analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# Fix the matplotlib configuration cell
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'matplotlib.use(' in str(cell['source']):
        # Replace the matplotlib configuration with correct backend
        cell['source'] = [
            "# Jupyter magic commands for matplotlib display\n",
            "%matplotlib inline\n",
            "%config InlineBackend.figure_format = 'retina'\n",
            "\n",
            "# Configure matplotlib for optimal notebook display\n",
            "import matplotlib\n",
            "import matplotlib.pyplot as plt\n",
            "# Use nbAgg backend for Jupyter notebooks\n",
            "matplotlib.use('nbAgg')\n",
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

print("✅ Fixed matplotlib backend error!")
print("✅ Changed to 'nbAgg' backend (correct for Jupyter notebooks)")
print("✅ Kept %matplotlib inline magic command")
print("✅ Figures should now display properly")
