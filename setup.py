import setuptools

setuptools.setup(
    name="mechafil_jax",
    version="0.1",
    packages=["mechafil_jax"],
    install_requires=[
        "jax", 
        "jaxlib", 
        "numpy", 
        "scipy", 
        "matplotlib",
        "pystarboard @ git+ssh://git@github.com:protocol/pystarboard.git",
    ],

    tests_require = [
        'pytest',
        'pandas==1.5.3',
        'mechaFIL @ git+ssh://git@github.com:protocol/filecoin-mecha-twin.git@mechafil_jax'  # get the branch that is built for comparisons w/ jax
    ]
)