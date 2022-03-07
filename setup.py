from setuptools import setup, find_packages

setup(
    name='VNLP',
    version='0.1',
    description='NLP Tools for Turkish Language.',
    author='Meliksah Turker',
    author_email='turkermeliksah@hotmail.com',
    license='Apache License 2.0',
    classifiers=[
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Information Technology',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Text Processing',
    'Topic :: Text Processing :: Linguistic',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',

    'Programming Language :: Python :: 3.7'
    'Programming Language :: Python :: 3.8'
    'Programming Language :: Python :: 3.9'
],
    packages=find_packages(exclude=['turkish_embeddings']),
    include_package_data=True,
    install_requires=['tensorflow<2.6.0; python_version < "3.8"',
                      'tensorflow>=2.6.0; python_version >= "3.8"',
                      'regex==2021.8.28', 'cyhunspell'],
    extras_require={"extras": ['gensim', 'spacy']},
    entry_points={"console_scripts": ["vnlp=vnlp.bin.vnlp:main"]}
    )