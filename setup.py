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
],
    packages=find_packages(), #exclude=['turkish_embeddings']
    package_data={'': ['_resources', 'resources', '*.txt', '*.hdf5', '*.md', '*.pickle']},
    include_package_data=True,
    install_requires=['regex==2021.8.28', #'numpy==1.19.5', 
                      'tensorflow>2.3.0', 'cyhunspell==2.0.2'],
    entry_points={"console_scripts": ["vnlp=vnlp.bin.vnlp:main"]}
    )