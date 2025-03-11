from setuptools import setup, find_packages

setup(
    name="tensorcore",
    version="0.1.0",
    author="Aakash",
    author_email="aakashpal1183@gmail.com",
    description="An elementary deep-learning library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Aakash-Engineer/tensorcore",
    package_dir={"": "src"},  # Tell setuptools that the code is inside "src/"
    packages=find_packages(where="src"),  # Find packages inside "src/"
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
