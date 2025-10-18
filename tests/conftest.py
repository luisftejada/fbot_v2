# Permite que pytest encuentre los módulos en la carpeta scripts
def pytest_configure():
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
