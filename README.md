# violin_pitch_tracking
 Carnatic violin pitch tracking model that applies a modified version of the [CREPE](https://github.com/marl/crepe) pitch tracking model to estimate pitch contours for selected multi-track violin recordings from the [Saraga Carnatic dataset](https://mtg.github.io/saraga/).

Multi-track recordings have been source-separated beforehand using the state-of-the-art [Hybrid Demucs](https://github.com/adefossez/demucs) model.

The modified CREPE model uses the same architecture as the original, but is trained using [*Violin Etudes*](https://github.com/nctamer/violin-etudes).

Further details are provided in the following paper:

> [Violin etudes: A comprehensive dataset for f0 estimation and performance analysis](https://ismir2022program.ismir.net/poster_247.html)<br>
> Nazif Can Tamer, Pedro Ramoneda, and Xavier Serra.<br>
> Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), 2022
 
## Contents
This repository contains the notebook `pitch_extraction.ipynb` which is used to compute activation matrices for each track in `\audio_multitrack`.

The activation matrices can then be utilised to compute pitch and confidence tracks using three Viterbi algorithms

* `viterbi=True`: Custom viterbi setting specific to the model which uses `viterbi_discriminative` from [librosa](https://github.com/librosa/librosa)
* `viterbi=False`: Uses local averaging
* `viterbi='weird'`: Uses the default CREPE viterbi setting.

There is an option of applying a voicing threshold based on confidence values to the computed pitch tracks.

Sections of the obtained pitch tracks can be plotted and sonified to compare against the vocal pitch annotations for the corresponding tracks from Saraga (`\vocal_pitch_annotations`). Plots for two example tracks are saved in `\plots`.

`utils.py` contains the utility functions to accompany the main notebook. `core.py` is a modeified version of the script from the original CREPE repository.
