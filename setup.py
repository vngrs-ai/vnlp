from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as readme_file:
    README = readme_file.read()

setup(
    name="vngrs-nlp",
    version="0.2.3",
    description="Turkish NLP Tools developed by VNGRS.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Meliksah Turker",
    author_email="turkermeliksah@hotmail.com",
    license="GNU Affero General Public License v3.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    project_urls={
        "Homepage": "https://github.com/vngrs-ai/vnlp",
        "Documentation": "https://vnlp.readthedocs.io/en/latest/",
        "Website": "https://vnlp.io",
    },
    packages=find_packages(exclude=["turkish_embeddings"]),
    include_package_data=True,
    setup_requires=[
        "swig==3.0.12",
    ],
    install_requires=[
        'tensorflow<2.6.0; python_version < "3.8"',
        'tensorflow>=2.6.0; python_version >= "3.8"',
        "swig==3.0.12",
        "regex",
        "requests",
        "sentencepiece",
        "jamspell"
    ],
    extras_require={"extras": ["gensim", "spacy"], "develop": ["pre-commit", "pytest"]},
    entry_points={"console_scripts": ["vnlp=vnlp.bin.vnlp:main"]},
)
