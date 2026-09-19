"""Records the interpreter, platform and package versions of the run in data/env.json."""
import json, platform
from importlib.metadata import version

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pillow",
            "dilithium-py", "cryptography"]
env = {"python": platform.python_version(), "platform": platform.platform()}
env.update({p: version(p) for p in PACKAGES})
json.dump(env, open("../data/env.json", "w"), indent=2)
print(json.dumps(env, indent=2))
