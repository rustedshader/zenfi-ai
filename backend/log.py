import asyncio

from langchain_sandbox import PyodideSandbox

sandbox = PyodideSandbox(
    allow_net=True,
    allow_run=True,
    allow_write=True,
    allow_read=True,
    allow_ffi=True,
    allow_env=True,
)


async def main():
    result = await sandbox.execute("""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib import pyplot as plt
import numpy as np
import io
import base64

# Generate example data
np.random.seed(42)
X = np.random.rand(100)
y = 2 * X + 1 + np.random.normal(0, 0.1, 100)

# Create scatter plot
plt.scatter(X, y, label='Data')

# Fit a line
coeffs = np.polyfit(X, y, deg=1)
y_pred = np.polyval(coeffs, X)
plt.plot(X, y_pred, color='red', label='Fit: y={:.2f}x+{:.2f}'.format(coeffs[0], coeffs[1]))

# Add labels and title
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Fit Example')
plt.legend()
plt.tight_layout()

# Save plot to bytes buffer and encode as base64
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
plt.close()

# Return base64 string
print(img_str)
""")
    print(result)


asyncio.run(main())
