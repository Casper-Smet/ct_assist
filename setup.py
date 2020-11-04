import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ct_assist",
    version="0.5",
    author="Casper Smet",
    author_email="casper.smet@gmail.com",
    description="Automatically finding head-feet pairs for CameraTransform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Casper-Smet/ct_assist",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Windows 10, Linux",
    ],
    python_requires='>=3.7.7',
)