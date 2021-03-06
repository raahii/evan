from setuptools import find_packages, setup

with open("requirements.txt") as requirements_file:
    install_requirements = requirements_file.read().splitlines()

setup(
    name="evan",
    version="0.0.2",
    description="Evaluation framework for GANs for video generation",
    author="raahii",
    url="https://github.com/raahii/video-gans-evaluation",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=install_requirements,
    entry_points={"console_scripts": ["evan=cli.evan.main:main"]},
    python_requires=">=3.6",
    test_suite="tests",
)
