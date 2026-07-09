from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

# imagecodecs uses lazy loading via __getattr__ for all codec functions
# PyInstaller cannot trace these, so we must collect all submodules explicitly
hiddenimports = collect_submodules('imagecodecs')

# Also collect the binary .pyd extension modules
binaries = collect_dynamic_libs('imagecodecs')
