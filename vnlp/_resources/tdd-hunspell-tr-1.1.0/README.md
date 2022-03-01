# hunspell-tr

[Hunspell](http://hunspell.github.io/) dictionary for Turkish language developed by Turkish Data Depository group.

This dictionary was constructed based on word lists that were extracted from Turkish text corpora and dictionaries. You can find details about the performance of this spell checker [here](https://github.com/tdd-ai/spell-checking-and-correction/blob/main/README.md#turkish-spell-checker-benchmark).

## Evaluation

When compared to other spellcheckers on the [trspell-10](https://data.tdd.ai/#/3477863a-9a7d-4b96-b13f-7afac1490ce0) dataset, Hunspell-tr performs much better in terms error correction accuracy.

| Spell Checker | Error detection Precision | Error detection Recall | Error detection F1-Score | Correction accuracy |
| --- | --- | --- | --- | --- |
| [zemberek-python](https://github.com/Loodos/zemberek-python)                                            | **99.94** | 94.21 | 96.99 | 93.05 |
| Hunspell-based |||||
| [hunspell-tr](https://github.com/hrzafer/hunspell-tr) (hrzafer)                                         | 99.63 | **99.36** | **99.50** | 79.61 |
| [TDD hunspell-tr](https://github.com/tdd-ai/hunspell-tr) (ours)                                         | 99.90 | 97.30 | 98.58 | **94.67** |

**Contributors** (alphabetically):

- Ali Safaya
- Arda Göktoğan 
- Deniz Yuret
- Emirhan Kurtuluş
- Taner Sezer
