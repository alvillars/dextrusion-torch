DeXtrusion (PyTorch)
====================

Detection of cellular events (extrusion / cell death, division, sensory organ precursor
differentiation...) in 2D+t movies of epithelial tissues, using sliding windows classified by a
small CNN + GRU network (a *DeXNet*).

This is a rewrite of the original TensorFlow/Keras implementation with **PyTorch**, **MONAI**
and **uv**. Networks trained with the original version load here unchanged and give the same
outputs (see `Reproducibility`_).

Install
-------

.. code-block:: bash

    git clone <this repository> && cd dextrusion-torch
    uv sync                      # python >= 3.11, creates .venv from uv.lock
    uv run dextrusion --help

Detect events
-------------

Movies are gray-scale tif stacks shaped ``(T, Y, X)`` (for 3D tissues, use a 2D projection).

.. code-block:: bash

    uv run dextrusion detect movie.tif -m models/notum_all -o results/

``-m`` is either one network folder or a folder containing several networks, which are then used
as an ensemble (recommended; ``models/notum_all`` holds two). For every event class, the results
folder gets the raw probability map (``movie_cell_death_rawproba.tif``) and the detected events
as Fiji point ROIs (``movie_cell_death.zip``, one file per event type). Useful options:
``--cell-diameter`` and ``--extrusion-duration`` (rescale movies whose cell size / event
duration differ from the training data), ``--volume-threshold`` / ``--proba-threshold`` (event
filtering), ``--save-proba``, ``--save-cleaned``, ``--device cuda``.

Shipped networks (``models/``): ``notum_all`` (extrusion, SOP, division; the default choice),
``notum_ExtSOPDiv``, ``notum_ExtSOP`` and ``notum_Ext``, each an ensemble of two.

Use networks trained with the original TensorFlow version
---------------------------------------------------------

Conversion can be done at any time, on any legacy DeXNet folder or zip (including your own
retrained ones):

.. code-block:: bash

    uv run dextrusion convert path/to/legacy_deXNet            # -> path/to/legacy_deXNet/torch
    uv run dextrusion convert network.zip -o converted_network

The legacy weights are read with TensorFlow, which is *not* a dependency of this project:
``convert`` runs a small script (``src/dextrusion/convert/extract_tf.py``) in an isolated
``uv run`` environment (the first call downloads TensorFlow; ``uv`` must be installed). If you
prefer TensorFlow inside the project environment, ``uv sync --extra convert`` works too.
``detect``, ``train --init-from`` and ``evaluate`` also accept a legacy folder directly: it is
converted on first use and cached in a ``torch`` sub-folder.

A converted network is ``model.safetensors`` (weights) plus ``config.json`` (window size,
classes, scale parameters).

Train / retrain
---------------

Training data is a folder with ``name.tif`` movies and ROI zips named ``name_cell_death.zip``,
``name_cell_sop.zip``, ``name_cell_division.zip`` (and optionally ``name_nothing.zip`` with typical
false positives). The data used for the published networks is on Zenodo
(https://doi.org/10.5281/zenodo.7586394).

.. code-block:: bash

    uv run dextrusion train data/ -o my_net --nb-filters 16 --epochs 40 --naug 3 --add-nothing 15
    uv run dextrusion train new_data/ -o my_net_v2 --init-from my_net --epochs 10   # retrain

Training is seeded (``--seed``); epoch metrics go to ``my_net/history.csv``. Results are
reproducible for a given seed, device and number of data-loader workers (``--workers``); the
random streams of the workers differ from those of the main process, so changing the number of
workers changes the result.

Custom event classes are given with ``--catnames``: the ROI file suffix of every class, the first
one empty for "no event", for example ``--catnames "" _cell_delamination.zip _cell_division.zip``
(this sets the number of classes).

Fine-tuning with a new class: combine ``--init-from`` with ``--catnames`` listing the classes of
the starting network in the same order, followed by the new ones. The output layer is widened:
the existing classes keep their output weights and the new class starts from a fresh row, so the
probabilities shift slightly before training. ``--freeze-cnn`` keeps the per-frame CNN fixed and
trains only the GRU and the decision head, which suits small datasets, and a smaller ``--lr``
(for example 0.01) than the 0.1 used from scratch is advisable:

.. code-block:: bash

    uv run dextrusion train data/ -o my_net_v2 --init-from models/notum_all/notumAll0 \
        --catnames "" _cell_death.zip _cell_sop.zip _cell_division.zip _cell_delamination.zip \
        --freeze-cnn --lr 0.01 --epochs 10 --naug 3

Without ``--catnames`` a retrained network keeps exactly the classes of the network it starts from.

A small annotated movie gets few windows next to a large dataset (the sampler balances classes
within each movie, not across movies). ``--oversample MOVIE=K`` samples a movie K times, with
independent position jitter, augmentation and random "nothing" windows; only training windows are
repeated, validation windows are not, and every annotated event stays on one side of the
train/validation split. K copies of the same few events can make the network memorise them, so
watch the validation loss and prefer a moderate K together with ``--freeze-cnn``.

Label new training data
-----------------------

A napari tool writes the ROI files that ``train`` reads. It needs the optional ``label`` extra:

.. code-block:: bash

    uv sync --extra label
    uv run dextrusion label movie.tif -o annotations/
    uv run dextrusion label movie.tif -o annotations/ --classes division=_cell_division.zip \
        delamination=_cell_delamination.zip       # these two are the default classes

The movie is shown with one points layer per class. Press ``q`` (first class) or ``w`` (second
class) to switch to that class in add mode, then click to add an event at the current frame; select
points and press Backspace to delete them. ``k`` saves, and by default every change is saved
automatically to ``annotations/<movie name><suffix>`` (ImageJ point ROIs). The previous version of a
file is kept as ``<name>.bak.zip``, existing files are loaded when the tool starts, and the movie is
symlinked into the folder, which is then directly a training folder. Further classes use the keys
``g``, ``h`` and ``j`` (at most five classes).

Mark every event at the frame and cell position where it is most recognisable, always on the same
landmark of the event: a training window spans 5 frames before to 4 frames after the marked frame,
and the sampler moves the position by up to 2 frames. For delaminations seen from the basal side
(an emerging cell instead of a shrinking one) use a class of their own, since the shipped networks
only know the apical appearance.

The tool saves coordinates of the movie you annotate and does not rescale. The networks work with
cells of about 25 px, so a movie with much larger (or smaller) cells has to be rescaled before
training; ``prepare`` writes a training copy of the movie together with correspondingly scaled
ROI files, using exactly the zoom that ``detect`` applies for the same ``--cell-diameter``
(diameter of the cells in *your* movie, in pixels). The originals are not modified:

.. code-block:: bash

    uv run dextrusion prepare movie.tif -o train_copy/ --cell-diameter 50 --rois-dir annotations/
    # -> train_copy/movie.tif (downscaled 2x), train_copy/movie_cell_division.zip, ...
    #    and train_copy/movie.prepare.json with the ratios and ROI counts

``--extrusion-duration`` does the same along time for movies whose events last more or fewer
frames than the 4.5 of the networks (rescaling is only applied when the value differs by more than
30 %, as in ``detect``). Annotate on the original movie, prepare, then train on the prepared folder.

Evaluate
--------

.. code-block:: bash

    uv run dextrusion evaluate events results/movie_cell_death.zip manual/movie_cell_death.zip
    uv run dextrusion evaluate windows data/ -m my_net

``events`` prints true/false positives, false negatives, precision and recall (a detection
matches a manual event within ``--distance-xy`` pixels and ``--distance-t`` frames).

Reproducibility
---------------

* **Inference is numerically equivalent to the original**: converted networks reproduce the Keras
  outputs to float32 precision (``tests/test_parity_model.py``, all eight shipped networks), and
  the whole detection (window scaling, tiling, ensemble shifts, float16 accumulation,
  probability maps, watershed ROI extraction) was checked against the *unchanged original code*
  (``tests/test_parity_e2e.py``). On the synthetic test movie the probability maps are
  bit-identical. On real movies (8 movies of the Zenodo dataset: the 7 of the table below plus
  one training movie) at most 0.006 % of the probability-map pixels differ, by one of the 256 levels at most (float32
  rounding differences between TensorFlow and PyTorch), and the detected events and their
  precision / recall against the manual annotations are identical.
  Reference data is generated by ``tests/fixtures/make_golden.py`` and
  ``tests/fixtures/make_e2e_reference.py``, which need the original code and models.
* Quirks of the original that influence results are kept by default and documented in
  ``inference.py`` (e.g. the ensemble shift formula); ``--consistent-ensemble-shift`` selects the
  corrected one.
* **Training cannot be bit-identical to the original** (different framework and random number
  streams), but it is deterministic here for a given ``--seed`` and device, with the same
  architecture, initialisation, optimiser (SGD, lr 0.1), sampling and augmentation recipe.

Same results as the original on held-out movies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original TensorFlow code and this version were run on the 7 test movies of the Zenodo
dataset (movies 2, 7, 15, 18, 22, 26, 30, never seen by the ``notum_ExtSOPDiv`` networks), with
the 2-network ensemble, the default parameters of the original interface (25 px cell diameter,
4.5 frames event duration, window steps 10 px / 2 frames, volume threshold 800, probability
threshold 180) and event matching within 15 px / 4 frames of a manual event. Only extrusions are
scored here, as the division and SOP annotations of these movies are not exhaustive. Both
versions detect exactly the same events and get exactly the same scores:

.. list-table::
   :header-rows: 1

   * - Movie
     - Manual extrusions
     - Detected (original / this version)
     - Precision (original / this version)
     - Recall (original / this version)
   * - 2
     - 883
     - 966 / 966
     - 0.72 / 0.72
     - 0.91 / 0.91
   * - 7
     - 11
     - 10 / 10
     - 0.80 / 0.80
     - 0.80 / 0.80
   * - 15
     - 63
     - 64 / 64
     - 0.81 / 0.81
     - 0.96 / 0.96
   * - 18
     - 123
     - 483 / 483
     - 0.17 / 0.17
     - 0.72 / 0.72
   * - 22
     - 59
     - 56 / 56
     - 0.91 / 0.91
     - 0.91 / 0.91
   * - 26
     - 682
     - 793 / 793
     - 0.75 / 0.75
     - 0.94 / 0.94
   * - 30
     - 499
     - 508 / 508
     - 0.68 / 0.68
     - 0.80 / 0.80
   * - **Median**
     -
     -
     - **0.75 / 0.75**
     - **0.91 / 0.91**

The detected events of the division and SOP classes are also identical in all 21 movie / class
combinations. Recall is the original definition (true positives are matched one-to-one, false
negatives are not), kept for comparability.

Differences from the original version
-------------------------------------

* The train/validation split is done per annotated event *before* augmentation. The original
  shuffled augmented copies, so validation windows leaked into training. Validation windows are
  not augmented.
* Window intensities are min-max normalised per window, as at inference. The original used the
  batch-wide min/max during training (``--legacy-batch-normalization`` restores it).
* Window-level evaluation is deterministic (no random flips / noise).
* No GUI and no temporary tif files; Fiji macros and notebooks of the original are not ported.

Citation
--------

When you use DeXtrusion code, networks or data, please cite the original paper: *DeXtrusion:
automatic recognition of epithelial cell extrusion through machine learning in vivo*,
Development 150(13), 2023, https://doi.org/10.1242/dev.201747

License
-------

BSD 3-Clause, see ``LICENSE``.
