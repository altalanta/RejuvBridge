import importlib
import platform


def check_env() -> None:
    print("RejuvBridge environment check")
    print("Python:", platform.python_version())
    for pkg in ["numpy", "pandas", "torch", "fastapi", "mlflow"]:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"- {pkg}: {ver}")
        except Exception as e:
            print(f"- {pkg}: not installed ({e})")

