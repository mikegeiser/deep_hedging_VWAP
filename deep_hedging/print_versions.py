import importlib

packages = [
    "tensorflow",
    "tensorflow_probability",
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "h5py",
    "joblib",
    "tqdm",
    "numba",
]

for pkg in packages:
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "unknown")
        print(f"{pkg}=={ver}")
    except ImportError:
        print(f"{pkg} not installed")
