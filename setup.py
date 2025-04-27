from setuptools import setup, find_packages

setup(
    name="your_package_name",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of your project",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/your-repo-name",
    packages=find_packages(exclude=["tests*", "docs"]),
    include_package_data=True,  # Include files from MANIFEST.in
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.7",
    install_requires=[     #  install_requires=requirements,and mention all in requirements text file
        "requests>=2.20",
        "numpy>=1.18.0"
    ],
    extras_require={
        "dev": ["black", "flake8", "pytest"],  # Optional dependencies for development
        "docs": ["sphinx", "sphinx_rtd_theme"],  # Optional dependencies for docs
    },
    entry_points={
        "console_scripts": [
            "your-command=your_package.module:main_function",
        ],
    },
    project_urls={  # Optional, nice for PyPI
        "Bug Tracker": "https://github.com/yourusername/your-repo-name/issues",
        "Documentation": "https://your-repo-name.readthedocs.io/",
    },
)
