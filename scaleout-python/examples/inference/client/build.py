import os
from scaleoututil.helpers.helpers import get_helper
import numpy as np

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


def build():
    output_dir = os.environ.get("SCALEOUT_BUILD_OUTPUT_DIR", ".")
    np.random.seed(42)
    params = np.random.rand(10).astype(np.float32)
    helper.save([params], os.path.join(output_dir, "seed.npz"))
    print(f"Created seed.npz with 10 random parameters.")

if __name__ == "__main__":
    build()
