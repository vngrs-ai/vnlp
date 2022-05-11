Quickstart
===================================

Installation
------------

Installation is possible via pip.

.. code-block:: console

   $ pip install vngrs-nlp

Usage
----------------

VNLP provides 2 very intuitive and simple API options.

Python API
----------------

You can simply import, initialize and predict!

For example:

>>> from vnlp import NamedEntityRecognizer
>>> ner = NamedEntityRecognizer()
>>> ner.predict("Ben Melikşah, 29 yaşındayım, İstanbul'da ikamet ediyorum ve VNGRS AI Takımı'nda çalışıyorum.")
[('Ben', 'O'),
('Melikşah', 'PER'),
(',', 'O'),
('29', 'O'),
('yaşındayım', 'O'),
(',', 'O'),
('İstanbul', 'LOC'),
("'", 'O'),
('da', 'O'),
('ikamet', 'O'),
('ediyorum', 'O'),
('ve', 'O'),
('VNGRS', 'ORG'),
('AI', 'ORG'),
('Takımı', 'ORG'),
("'", 'O'),
('nda', 'O'),
('çalışıyorum', 'O'),
('.'), 'O']


- See Main Classes in :ref:`index:Contents` for the rest of the functionality.


Command Line API
----------------
Command Line API is even simpler than Python API!

The format is

.. code-block:: console

    $ vnlp --task TASK_NAME --text INPUT_TEXT

Example:

.. code-block:: console

    $ vnlp --task sentiment_analysis --text "Sipariş geldiğinde biz karnımızı atıştırmalıklarla doyurmuştuk."
    0

To list available tasks/functionality:

.. code-block:: console

    $ vnlp --list_tasks

    # List of available tasks:
    stemming_morph_analysis
    named_entity_recognition
    dependency_parsing
    part_of_speech_tagging
    sentiment_analysis
    split_sentences
    correct_typos
    convert_numbers_to_words
    deasciify
    lower_case
    remove_punctuations
    remove_accent_marks
    drop_stop_words
