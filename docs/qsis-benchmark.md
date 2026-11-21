# QSIS waveform benchmark

The target is 95% accuracy for P, S and noise on independently labelled QSIS
waveform clips. Report accuracy, per-class recall, confusion matrices, the
number of earthquake groups and noise dates, and the coverage of readable
references. Picking arrivals from continuous recordings is a separate task.

## Current verified result

The retained model is the 33,795-parameter CNN at
`runs/qsis-ccul-extra/gravity-record-candidate`. It uses the original
16/32/64 convolution channels and 384/64/16/3 fully connected dimensions,
CWA initialization, steady-start high-pass preprocessing and measured
per-record gravity alignment. Training used 226 sources for 20 epochs
on Apple MPS. The normal fitting command reproduces its state tensors
exactly, and the prediction entry point has been checked on P, S and
noise records from the new confirmation cohort.

| Frozen current CNN evaluation | -0.2 s | Center | +0.2 s |
| --- | --- | --- | --- |
| New IES archive confirmation | 27/27 | 27/27 | 26/27 |
| Earlier prespecified public confirmation | 17/17 | 17/17 | 17/17 |
| Combined primary cohorts | 44/44 | 44/44 | 43/44 |
| Including three additional array-context references | 46/47 | 46/47 | 45/47 |

The new confirmation model identity and event/station selection were
fixed before waveform review or classifier evaluation. Its 27 scored
clips comprise 9 P, 6 S and 12 pre-event backgrounds across 12 UTC dates
absent from training and all previous evaluation cohorts. The two primary
cohorts cover 21 scored dates. Including the additional array references,
the centered accuracy is **97.87% (46/47)** and the lowest fixed-offset
accuracy is **95.74% (45/47)**. The centered per-class results for that
47-clip set are P 13/14, S 12/12 and noise 21/21. The weak CCUL P error
and the newly observed positive-offset P-to-S error remain in the scores.
The 215 archived public references are reported separately as descriptive
reuse, with 210/215 at the current model's center.

These measured results meet the 95% clip-classification target on the
independently reviewed reference set. They do not assign an accuracy to
unresolved phase slots or continuous arrival detection. Two independent
model-blind AI reviewers provided the references, as authorized for the
waveform audit; no expert QSIS arrival file was supplied. On the new
archive, 15 of 58 enrolled phase slots had clear agreeing onsets and
12 of 14 screened background candidates had two noise judgments. The
new archive covers one building and magnitude-5-or-larger events, while
the earlier prespecified cohort adds other sites and smaller events.
See `runs/qsis-ies-hybrid/verification.json` for the completion checks,
provenance hashes, actual inference checks and full coverage limits.

## Freeze the evaluation groups

```bash
seismictorch tools split --review runs/qsis-review-balanced/manifest.json \
  --pilot runs/qsis-onset-pilot-2026-11-12/metrics.json \
  --output runs/qsis-benchmark/split.json
seismictorch tools reference --split runs/qsis-benchmark/split.json \
  --partition validation --output runs/qsis-benchmark/validation-review
seismictorch tools reference --split runs/qsis-benchmark/split.json \
  --partition test --output runs/qsis-benchmark/test-review
```

The split groups SAC origin times within five seconds across stations and
keeps previously reviewed events and pilot noise dates in the training set.
The evaluation pool includes weaker recordings as well as records above the
historical 2 gal threshold. The review packets pick up to two available
stations per event and suppress candidate arrival lines. Use validation to
select the model and settings. Evaluate the final checkpoint on the frozen
test after the references are available.

## Train and compare

Use `--model tcn` with `seismictorch train pretrain` and `src/seismictorch/training/pilot.py` to compare the
small TCN against the original CNN. The TCN mixes all three axes at the first
layer, uses depthwise temporal filters with dilations 1, 2, 4 and 8, and pools
only at the classifier. It has 5,955 parameters versus 33,795 for the CNN.

RED-PAN is a separate TensorFlow inference dependency. The experiment used
the [author's 60-second checkpoint](https://github.com/tso1257771/RED-PAN/blob/1bcfb672dd184b1cb1e98491dcc5bd827614dcdb/pretrained_model/REDPAN_60s_240107/train.hdf5).
Its SHA-256 is `664c3023e9ca94659279d2264ee69f97db47668ef704157b9a8559a221031919`.

```bash
seismictorch tools redpan --split runs/qsis-benchmark/split.json \
  --weights runs/external-weights/redpan-60s.hdf5 \
  --output runs/qsis-benchmark/redpan-train.json
seismictorch tools transfer --device mps --review runs/qsis-review-balanced \
  --split runs/qsis-benchmark/split.json \
  --proposals runs/qsis-benchmark/redpan-train.json \
  --init runs/waveform-2026-11-12/stead/best.pth \
  --output runs/qsis-benchmark/proposal-transfer
```

Proposal inference only reads training events. A proposal passes when raw
and band-pass predictions both score at least 0.5 and agree within 0.35 s.
The transfer comparison excludes proposals from any event represented in the
development evaluation. The proposals are weak training labels and the
development scores are from earlier AI-reviewed clips.

## Development experiments

Use `--augment --embargo 120 --jitter-eval` with `src/seismictorch/training/transfer.py` to compare
±0.2 s training jitter and ±30° horizontal rotation. The raw jitter
evaluation re-cuts every development clip at both offsets. The embargo
excludes nearby event proposals in addition to exact group matches.
`--fit-seed` changes the model-training randomness while preserving the
development cohort. `--hard-noise` adds separate STA/LTA triggers and excludes
any trigger from a held-out noise date within each fold.

The augmented CNN with proposals scored 42, 42 and 41 out of 46 at training
seeds 42–44. At seed 42, shifting the evaluation center by −0.2 s reduced
accuracy to 38/46. This development cohort contains only 11 earthquake
groups and 22 noise dates. The frozen benchmark remains unevaluated.
With 64 additional STA/LTA triggers and date exclusions per fold, the
centered result remained 42/46. The −0.2 s result improved to 40/46 and
the +0.2 s result fell from 42/46 to 41/46 at seed 42.

The polarization CNN adds a 27-weight linear residual head using early and
late vertical energy fractions, concentration of three-component covariance,
and onset energy contrast. With the same proposal, augmentation and hard-noise
settings, the centered and −0.2/+0.2 s scores were 41/46 and 39/46, 42/46.
The original CNN scored 42/46 and 40/46, 41/46 in that comparison.
Averaging the original CNN's probabilities at −0.2, 0, and +0.2 s
also scored 42/46, with and without the extra STA/LTA triggers.

Adding real background noise to training phase clips at 8–20 dB
(`--mix-noise`) scored 42, 40 and 42 out of 46 at seeds 42–44. The
matched comparison without mixing scored 42, 42 and 41. The noise donor
pool contains only training dates for each fold. Raw shifted-window scores
also did not consistently improve with mixing.

For the U-Trans comparison, use TensorFlow and the upstream package at
commit `46a12fc2e35ecd9b273f73e950b0c40629c2cb52`. Run
`seismictorch tools utrans --development .../augmentation-jitter/metrics.json --weights .../UTrans_Foundation.h5 --output .../utrans-development.json`.
The backbone stays frozen and each fold fits its own regularized linear head.
On the 41 clips with 60 seconds of context, U-Trans scored 35/41 and the
original CNN scored 36/41 on the same subset. Five noise windows lacked
enough recorded context.

With `--long-context`, the CNN kept its weights and used adaptive temporal
pooling for four-second clips at 40 Hz. At the same centers and with the
same folds, it scored 40/46 versus 42/46 for the two-second CNN. The three
phase misses persisted, and more noise clips were classified as P or S.
Two of the phase misses came from the same earthquake. Their early
waveforms and the third missed phase showed more energy after the
two-second window ended.

### Taiwan strong-motion adaptation

The [CWA Benchmark TSMIP archive](https://huggingface.co/datasets/NLPLabNTUST/NoMerged-CWA)
supplies labelled 200 Hz three-component acceleration. The first waveform
member of `TSMIP.tar.gz` is `chunks_2011.hdf5`. The authors' [2011 sample metadata](https://huggingface.co/datasets/NLPLabNTUST/Merged-CWA/blob/main/sample.csv)
matches all 5,882 HDF5 names and sample counts. This period predates every
earthquake in the frozen QSIS split.

```bash
seismictorch tools cwa --waveforms runs/cwa-tsmip-2011/chunks_2011.hdf5 \
  --metadata runs/cwa-tsmip-2011/merged-cwa-sample.csv \
  --init runs/waveform-2026-11-12/stead/best.pth \
  --output runs/qsis-benchmark/cwa-pretrain-42 --device mps
```

The pilot uses the published manual picks on 2,748 single-event traces
that passed completeness and waveform checks. Its background class is
cut three seconds before P. The validation score was 97.99% on 73 CWA
events after training on 292 CWA events. This score describes the CWA
validation clips, whose pre-event background differs from QSIS triggers.
To compare transfer to QSIS, use this checkpoint as `seismictorch tools transfer --init`
with the same folds, augmentation and training seeds as the STEAD baseline.

On the same 46 QSIS development clips, CWA initialization followed by
reviewed-only fine-tuning scored 42, 43 and 43 correct (seeds 42–44).
With RED-PAN proposals and augmentation it scored 43, 41 and 43,
compared with 42, 42 and 41 for the matched STEAD initialization.
Two weak P windows were repeatedly classified as noise.

With `--later-context`, the two-second window starts 0.5 s before the
nominal arrival and ends 1.5 s after. It recovered both weak P examples
in all three runs, with two to three noise false positives per run.
Equal averaging of the centered and delayed CNN probabilities scored
44, 43 and 42 out of 46. The single 95.65% result comes from a development
fold experiment and does not establish reproducibility or performance on
the frozen QSIS test.

The reviewed-only candidate from this initialization is saved in
`runs/qsis-benchmark/cwa-reviewed-candidate`. It uses centered two-second
clips, inverse class frequencies and the default seed 42. To reproduce it,
use `seismictorch qsis fit` with the CWA initialization and `--no-augment`.
Omit `--proposals` for reviewed-only training.

The CWA candidate and the earlier STEAD candidate both classified all
17 previously reviewed public P/S clips and nine of ten noise clips
from the frozen validation dates. These are separate diagnostics. The
formal validation packet still lacks confirmed QSIS P arrivals.
In the nearby public batch, the CWA candidate missed two P arrivals
that the earlier STEAD candidate detected. This comparison favors retaining
the earlier candidate pending a larger evaluation.

With extra STA/LTA noise and CWA initialization, the centered development
scores were 43, 42 and 43 out of 46. The noise ablation did not recover
either of the two weak P cases.

### Additional QSIS P-wave review

`src/seismictorch/qsis/array_proposals.py` uses a cluster of at least three agreeing
RED-PAN picks at the same site to guide a separate PhaseNet search on
the remaining instruments. It yielded two extra S picks and no extra
P picks on the training split. The output remains weak labels.

Two reviewers independently examined 12 training events selected for
readable arrays and absent RED-PAN P picks. They agreed on one P and
two S arrival intervals at array level. A separate review of 15 station
plots from the P event gave two agreed P times at 5AE28 and 5AF0F.
This event followed another by 13 seconds. The two station windows were
used in a training ablation with a 120-second embargo around evaluation
events. Every run scored 43/46 on centered development clips, compared
with 42, 42 and 41 correct without the extra picks. At −0.2 s the new
scores were 36, 38 and 39, and at +0.2 s all three scored 41. The
candidate trained with these picks classified 45 of 46 agreed public
P/S clips, versus 46 of 46 for the previous candidate. The public audit
still covers selected readable phases and has no noise class.

`--raw-shifts` replaces reflected padding with SAC recuts at −0.2, 0
and +0.2 s during training. It scored 39, 42 and 41 of 46 centered,
with no gain at −0.2 s. The QSIS station registry lists different sensor
azimuths and horizontal handedness, and some change across installations.
`--full-rotation` tests uniform rotations with random horizontal
reflection. With the two new picks it scored 42, 42 and 43 of 46,
compared with 43 in all three runs under the earlier ±30° augmentation.

### Noise label and cut review

`src/seismictorch/qsis/noise_audit.py` compares training and validation noise triggers
with the [CWA historical earthquake catalog](https://data.gov.tw/en/datasets/97095)
using QSIS station coordinates for the recording date. It found six
possible matches among 696 checked recordings. Three stations recorded
the same magnitude 4.9 event on June 22, 2023. Two independent readers
agreed on an S arrival at 20.75–20.85 seconds after the origin for each
station. They found two other candidates resembled isolated impulses
and left one unresolved. The corrected copies are stored separately
from the original SAC files.

`seismictorch tools transfer --augment --hard-noise --noise-corrections .../corrections.json`
uses the reviewed phases and samples other training noise from those days.
Its centered scores were 40, 41 and 40 of 46, versus 42, 42 and 40
with the original labels. The correction remains valid even though
this particular training recipe performed worse.

Six of ten validation noise windows were centered in the recording's
taper. Moving the trigger search past the taper and the five-second
STA/LTA warm-up changed these six windows. Three fixed checkpoints
each scored 10/10 on the interior windows and 9/10 on the original ones.
This compares different clips from the same recordings across four
validation dates. The training ablation with both the corrected labels
and interior cuts scored 40, 42 and 41 of 46 on the original development
windows. Keep the original scoring protocol for comparisons.

A three-second shared CNN evaluated centered and delayed two-second
views jointly. It scored 41, 42 and 40 out of 46 with the earlier
training set and 39, 41 and 40 with the additional station picks.

With `--downweight-proposals`, each training epoch samples with
replacement, assigning the RED-PAN clips half the sampling weight of
reviewed clips and noise. With the same two additional station picks,
it scored 42, 43 and 44 out of 46, versus 43 in all three runs with
uniform shuffling. At −0.2 s it scored 38, 39 and 39 out of 46. This
comparison changes both sample weights and the sampling procedure.

Freezing the STEAD CNN and updating only its final linear layer with
`--freeze-backbone` scored 31/46 at each of three seeds. Full fine-tuning
on the same data with the two reviewed station picks scored 43/46.

An exploratory multiscale residual uses 3-, 7- and 15-sample filters
across all three axes and pools the two halves of the waveform separately.
This adds 912 parameters. A zero-initialized output projection preserves
the CWA CNN logits at initialization. With matched development clips,
folds and training seeds on MPS, the augmented proposal runs scored 43,
41 and 43 out of 46 with either model. Predictions also matched after
shifting the windows by ±0.2 s in those runs. The branch remains an
experiment in `runs/qsis-benchmark/multiscale-pilot`.

The [Italian MEMS archive](https://data.mendeley.com/datasets/zsrk4cngtr/2)
provides analyst picks and separate noise traces. After filtering by
pick uncertainty, the experiment used 63 P, 77 S and 25 noise clips for
adaptation and held out complete dates with 20 P, 19 S and 18 noise clips.
On the same 46 grouped QSIS development clips and three seeds, both the
centered and jittered MEMS initializations scored 42, 41 and 41 at the
nominal centers. The matched STEAD controls scored 42, 42 and 41.
At −0.2 s, all three initializations scored 38, 38 and 39. The Italy
experiments are in `runs/mems-italy/qsis-comparison.json`.

A separate training experiment labeled all three real cuts at −0.2, 0,
and +0.2 s and penalized divergence between their predictions. With the
consistency penalty, centered scores were 41, 41 and 42 out of 46,
compared with 42, 41 and 42 under the same training without the penalty.
P recall across the three offsets matched at each seed. See
`runs/qsis-benchmark/shift-consistency/comparison.json`.

A factorized objective balances seismic phase versus noise separately
from the conditional P/S loss. At −0.2 s it scored 39, 39 and 40 out of
46 versus 38, 38 and 39 for the matched controls. Centered accuracy was
42, 42 and 41 out of 46 for both. See
`runs/qsis-benchmark/factorized-loss/comparison.json`.

The optional `--model smoothed` CNN applies a short temporal filter to
the dense max-pool responses before each stride, adapting
[Zhang's pooling approach](https://proceedings.mlr.press/v97/zhang19a.html).
This preserves the trainable tensor shapes. At −0.2 s it scored 40, 39
and 39 out of 46 versus 38, 38 and 39. At the center it scored 42, 42 and
42 versus 42, 42 and 41. At +0.2 s it scored 44, 44 and 43 versus 42, 43
and 42. The fixed versions agreed on all 46 centered phase
classifications from the earlier public audits. On ten separate
validation noise clips across offsets −0.2, 0 and +0.2 s, the smoothed
model scored 7, 8 and 9 versus 6, 9 and 8. The original CNN remains the
default. See `runs/qsis-benchmark/antialias-pilot`.

[Pacific Northwest strong-motion data](https://doi.org/10.26443/seismica.v2i1.368)
provided an alternative to the Raspberry Shake archive. The experiment
read 156 three-component acceleration records, one per earthquake, with
analyst P/S picks and a background window before each catalog origin.
The 120 adaptation events precede 2020. The 36 external validation events
occurred in 2020–2022. The external macro recall averaged across window
offsets went from 90.12% to 94.44%. On QSIS, the adapted initialization
scored 40, 40 and 41 out of 46 centered, versus 42, 42 and 41 for
the matched controls. See `runs/pnw-acceleration/qsis-comparison.json`.

## Score an independent review

`seismictorch qsis fit` writes a checkpoint plus a hash of the frozen split
and the data sources used to fit it. The candidate is unvalidated until an
independent evaluation succeeds. `seismictorch qsis score` accepts the
JSON export from a blind review packet, verifies the split and model
hashes, and reports per-class recall, coverage and earthquake/noise-date
counts. The score includes a 95% cluster bootstrap interval that resamples
earthquake groups and noise dates separately. With few groups the interval
is exploratory. Fill in every recording in the packet and leave unresolved arrivals
blank. Use the validation packet for model selection. Score the test
packet only after fixing the checkpoint and protocol.

Generate unmarked detail views for the frozen test packet:

```bash
seismictorch tools array-review --split runs/qsis-benchmark/split.json \
  --partition test --review-manifest runs/qsis-benchmark/test-review/manifest.json \
  --output runs/qsis-benchmark/test-waveform-context
```

Two visual readings covered every original test recording (40 recordings
from 24 events). No recording had an agreed P or S first break. Only one
original recording reached 2 gal, while the full test archive contains
such recordings for nine events. A separate supplement added 19 recordings
from 19 of the same frozen groups with `seismictorch tools reference --supplement`.
The supplement prioritizes peak amplitude relative to background. The
two readers found no agreed arrivals in those recordings either. The original
packet's hash and its designated checkpoint were verified. No three-class
test result was computed.

A follow-up downloaded 40 Hz IU.TATO broadband waveforms for all 24 groups
from [EarthScope](https://service.earthscope.org/fdsnws/dataselect/1/).
Seven groups also had 100 Hz waveforms. Two AI readers independently
reviewed all 40 original QSIS recordings using the TATO trace, two filter
bands and local QSIS array energy. TATO was shifted by the phase-specific
TauP travel-time difference. TEST-039 used its own SAC origin, five
seconds after the group key, and was reviewed again. No phase met the
two-reader standard for a clear first arrival within 0.6 s.

Running PhaseNet and RED-PAN on TATO retained five P and six S proposals
after requiring agreement across models and between unfiltered and
band-pass views. The shifted proposals defined 17 search windows on the
QSIS recordings. RED-PAN retained one local S proposal (TEST-004,
39.20 s after the QSIS origin), and PhaseNet retained none. The combined
shortlist contains one P and three S candidates, including the separate
visual suggestions. See
`runs/qsis-benchmark/test-tato/qsis-candidate-shortlist.json` and
`visual-consensus.json` in the same directory. These candidates have
not entered training or formal test scoring.

The [2026 Taiwan tomography release](https://zenodo.org/records/21194749)
contains P arrivals for origins dated 1991–2018 in its local event
catalog. The 2022–2023 QSIS test groups fall outside that catalog's dates.

The [Reviewed ISC Bulletin](https://www.isc.ac.uk/iscbulletin/search/arrivals/)
gave unique matches for 24 of 40 QSIS recordings across 15 groups. The
context plots use time-defining picks within 25 km that have residuals
within ±2 s. After adjusting for travel time, the nearby picks defined
22 P windows and 13 S windows. Two independent AI readers agreed on
TEST-007 S at 24.35 and 24.40 s. The target horizontal onset is later
than the 21.13–23.40 s estimates from nearby stations. All P candidates
and the remaining S candidates are unresolved. PhaseNet and RED-PAN
retained no local arrivals in the 29 windows with at least two
sufficiently close nearby estimates. The shortlist keeps the ISC and
earlier TATO evidence in `runs/qsis-benchmark/test-isc/qsis-candidate-shortlist.json`.
The ISC review provides context and the QSIS arrival is an AI-reviewed
candidate. No classifier score was computed on the test partition.

P-Alert supplied close-station acceleration for three recordings
across the original and supplemental frozen packets. Independent
readers both marked the TEST-034 P candidate tentative, at 39.77 and
39.82 s after origin. The other tentative suggestion is SUP-018 S
at 69.12 s from one reader. No phase met the clear two-reader rule.
See `runs/qsis-benchmark/test-palert/consensus.json`.

```bash
seismictorch qsis score \
  --split runs/qsis-benchmark/split.json \
  --manifest runs/qsis-benchmark/validation-review/manifest.json \
  --labels /path/to/validation-review-export.json \
  --candidate runs/qsis-benchmark/frozen-packet-candidate \
  --label-provenance "Describe the actual reviewer and review procedure" \
  --output runs/qsis-benchmark/validation-metrics.json
```

Independent AI reviewers agreed on no single-record validation phases.
Same-site array review found one agreed S arrival among 19 earthquake
groups. Nearby ISC catalog picks can inform review and should not be used
as station-specific QSIS reference times.

Generate validation-only context plots with:

```bash
seismictorch tools array-review --split runs/qsis-benchmark/split.json \
  --review-manifest runs/qsis-benchmark/validation-review/manifest.json \
  --catalog runs/qsis-benchmark/validation-review/isc-context.json \
  --output runs/qsis-benchmark/validation-context
```

The optional catalog view overlays broad search intervals after correcting
nearby station picks by the TauP travel-time difference. Use it to inspect
the actual onset on the target recording and compare its coherence across
co-located instruments. Record unresolved picks as such. The original blind
packet remains available without any catalog overlays.

For a catalog-assisted second reading, open `validation-context/annotate.html`.
It uses the frozen validation record IDs and exports a separate file from
the blind review. For two-reader agreement, merge the review JSON files
using `seismictorch tools consensus --catalog-context .../validation-context/details.json`.
The scorer accepts agreed recording-level consensus on validation only.
The three AI readers agreed on two S picks from one earthquake and no P
picks in the 43-record validation packet. It cannot yet provide a
three-class validation score.

A separate review used [IU.TATO broadband recordings from EarthScope](https://service.earthscope.org/fdsnws/dataselect/1/)
for six validation events. The nearby station waveforms were shifted by
the TauP travel-time difference and compared with the QSIS array. Two
readers agreed on two QSIS array S times and no P times. This supplied
array context without producing independent station references.

## Later-period public QSIS waveform audit

```bash
seismictorch tools public --start 2026-08-12 --end 2026-11-09 --events 12 \
  --output runs/qsis-public-2026-audit
seismictorch tools array-review --public-audit runs/qsis-public-2026-audit \
  --output runs/qsis-public-2026-audit/array-review
```

This uses the [Academia Sinica QSIS event portal](https://smartsensors.earth.sinica.edu.tw/Events).
The download keeps the portal's 100 Hz time axis and records unavailable
stations. It does not provide P/S labels. The limited magnitude range,
date range and station list are intended for checking whether newer
recordings can support annotation. The audit is separate from the frozen
split and must not be used to claim benchmark accuracy.

The 2026 download has 69 readable recordings from 12 earthquakes. Three
independent array readers agreed on paired P/S arrivals for two events.
On those two events, two independent station readers agreed within 0.6 s
on nine P and eight S picks across 12 reviewed recordings. Seven of the
24 phase slots stayed unresolved. The fixed CNN classified all 17
reviewed clips correctly at the agreed centers and at ±0.2 s. This is a
selected two-event diagnostic without noise references or analyst labels.

To repeat the diagnostic after generating `station-review` with
`seismictorch tools array-review --public-audit … --array-consensus …/array-review/consensus.json`
and merging the independent station reviews with `src/seismictorch/qsis/consensus.py`:

```bash
seismictorch tools public-score \
  --split runs/qsis-benchmark/split.json \
  --download runs/qsis-public-2026-audit/download.json \
  --manifest runs/qsis-public-2026-audit/station-review/manifest.json \
  --labels runs/qsis-public-2026-audit/station-review/consensus.json \
  --candidate runs/qsis-benchmark/frozen-packet-candidate \
  --output runs/qsis-public-2026-audit/station-review/diagnostic.json
```

Generate public candidates separately from training proposals:

```bash
seismictorch tools redpan --public-download runs/qsis-public-2026-audit/download.json \
  --weights runs/external-weights/redpan-60s.hdf5 \
  --output runs/qsis-public-2026-audit/redpan-candidates.json
```

The raw/band-pass agreement filter retained nine P and 11 S candidates
across five events. Five candidates from three previously unresolved
events have unmarked waveform views in `redpan-shortlist/index.html`.
For the 17 agreed station picks, RED-PAN retained 11 candidates; those 11
were within 0.28 s of the station consensus. Two readers then reviewed
the five shortlisted views. Both picked S at 55.4 s on REDPAN-001 and
31.8 s on REDPAN-005. Three arrivals remained unresolved. These views
used RED-PAN to select the time interval, so their agreement remains
model-assisted. See `redpan-shortlist/consensus.json`.

To increase coverage, a second public batch sampled 12 catalog events from
2025-01-01 to 2026-08-11 within 80 km of the Taipei site, at magnitude
4.0 or greater and at most 60 km deep. The portal provided 60 aligned
recordings. This selection prioritizes signals suitable for review and
belongs to the annotation audit. RED-PAN kept 12 P and 40 S candidates
from the 60 recordings for further checking.
Two independent reviewers agreed on array P times for four events and
S times for ten. They then agreed on 12 P and 17 S times across 19
individual recordings from the four events with both array phases.
The STEAD-initialized CNN correctly classified 29/29 centered P/S clips.
The CWA-initialized CNN classified 27/29 correctly, with two P clips
classified as noise. These clips were selected by agreement on the
visible waveforms, and this audit has no noise references.

The Hualien HW01 audit used `seismictorch tools public --site HW01` to use registered
station coordinates outside Taipei. Ten recordings were available from
eight of ten queried events. Only two events had two instruments for
array review. The two readers agreed on four station-specific S times.
All three fixed candidates correctly classified those four clips at
0 and ±0.2 s. This batch supplied no confirmed P or noise references.

For the 2024 RCEC audit, `seismictorch tools public --min-duration 100` retained 17
complete recordings shorter than 400 seconds alongside 18 longer ones.
The array and station review additionally required pre-event background
and coverage through the search windows. Two readers agreed on 16 P
and 14 S arrivals from 18 station recordings across four events.
The original CNN classified 26 of these 30 centered phase clips correctly.

The 30 clips were then added to a separate fine-tuning experiment with
`seismictorch tools transfer --public-review DOWNLOAD MANIFEST CONSENSUS`. On the
original grouped development folds the three seeds scored 43, 41 and 40
out of 46, compared with matched controls at 42, 42 and 41. The later
2025–2026 phase diagnostics covered 50 clips from eight events. At the
nominal centers the new candidates classified 50, 50 and 49 correctly,
versus 50, 49 and 49 for the matched controls. On the ten original
validation noise windows they classified nine, eight and nine correctly,
versus nine on all three controls. The results show no consistent gain
across earthquake and noise clips. Public phase windows come from
recordings with readable first arrivals and the noise scores come from
separate validation dates.

The candidate trainer and diagnostic both check the public waveform and
review provenance. The diagnostic rejects public earthquake origins within
120 seconds of a public training event. The training-data seed stays
fixed when the optimizer seed changes.

## Prespecified 2026 QSIS cohort

The cohort in `runs/qsis-prospective-2026/cohort.json` fixes the checkpoint
and samples 24 earthquakes before downloading waveforms. It assigns
events within 120 km to the nearer of RCEC or HW01 and samples six
earthquakes per site and magnitude band (3.7–4.19 and 4.2+). It excludes
origins within two minutes of the historical split or earlier public
audits. Waveforms were available for 19 groups. The five missing groups
were all at HW01. The primary station was the lowest code with an aligned
recording that covered the search windows. Waveform strength did not
determine which recording to review.

Two independent AI readers agreed on six P and nine S arrivals across
13 of the 24 prespecified earthquakes. The same protocol checked 25
legacy test-partition triggers from four 2023 dates. The CWA catalog
screen flagged none. Two visual readers agreed that 22 clips were
noise-like and left three taper-obscured windows unresolved. The 22
usable noise clips represent two dates, with 21 clips on one date.

The fixed STEAD-initialized CNN classified **36 of 37 scored clips
(97.30%)** at the specified centers. It classified 6/6 P, 9/9 S and 21/22
noise. With either a −0.2 or +0.2 s shift, accuracy was 35/37 (94.59%) and
P recall fell to 5/6. The centered and shifted results are in
`runs/qsis-prospective-2026/fixed-checkpoint-diagnostic.json` and
`sensitivity.json`. This is a diagnostic on the identifiable portion
of the cohort. Only 15 of the 48 enrolled phase slots have references.
Noise recall is based mainly on one trigger date. Neither result
establishes 95% on the full QSIS population.

A separate background check used the 2026 cohort's pre-event recordings.
The [CWA GDMS catalog](https://gdms.cwa.gov.tw/catalogDownload.php) was
queried for each UTC date. Candidate centers needed a matched catalog
event, separation from nearby listed events, and a complete two-second
clip. Two visual readers found ten of the first eleven windows overlapped
the taper applied to the full recording. A corrected search excludes the
full start taper, the five-second STA/LTA lookback and the clip margin.
It yielded ten windows, one per date. The observers agreed nine were
noise-like and left one with a probable signal gap in the lookback
unresolved. The fixed CNN classified all nine as noise. These are
pre-event backgrounds. The legacy trigger test measures a different
noise domain. See `runs/qsis-prospective-2026/background-interior`.

## Prespecified 2025 QSIS comparison

The cohort in `runs/qsis-prospective-2025/cohort.json` pins both CNN
checkpoints before waveform download. It samples 30 previously unused
events, five per site and magnitude band at RCEC, HW01 and CCUL. All 30
primary waveforms could be plotted. Two independent AI visual readings
agreed on 5 P and 15 S arrivals across 15 events. RCEC and HW01 provided
all agreed arrivals. All ten CCUL phase records remained unresolved.
The separate CWA-catalog screen and two-reader review yielded 22
pre-event background clips across 22 dates.

At the registered centers, both CNNs scored **40/42 (95.24%)** on
the reviewed clips, with 5/5 P, 13/15 S and 22/22 background. Both S
errors are from the lower magnitude band at HW01. At −0.2 s, the
original and smoothed models scored 42/42 and 41/42. At +0.2 s, both
scored 41/42. The favorable smoothing ablation on the historical
development clips did not repeat here. See
`runs/qsis-prospective-2025/comparison.json`. Phase coverage is 20 of
60 enrolled slots and noise covers event prehistory. The result
describes the identifiable subset under this protocol.
The 95% interval from resampling dates is 88.64–100% for both models at
the center. It covers the scored subset and is recorded in
`runs/qsis-prospective-2025/cluster-uncertainty.json`.

A separate CCUL review used simultaneous [BATS broadband waveforms](https://doi.org/10.7914/SN/TW)
from TPUB for all ten events and SSLB for nine. Two readers agreed on
QH25-011 S at 15.60 s and QH25-025 P and S at 4.90 s and 8.53 s.
Both fixed models classified only QH25-011 correctly at the center and
at either ±0.2 s offset. The original registered score does not include
these later-reviewed candidates. See
`runs/qsis-prospective-2025/ccul-context/supplemental-diagnostic.json`.

## 2024 CCUL and HW01 training candidates

`runs/qsis-training-sites-2024/cohort.json` froze ten unused earthquakes
per site, split evenly above and below magnitude 4.5. The cohort placed
a 120-second embargo around earthquakes from the historical split and
earlier public cohorts. Two AI reviewers independently examined all 20
records. They agreed within 0.6 s on 6 P and 5 S arrivals, all at HW01.
They then reviewed 15 records with nearby BATS waveform context. That
added two CCUL P arrivals and one HW01 S arrival, for 8 P and 6 S total.
CCUL S remains unresolved. The consensus and the review inputs are in
`runs/qsis-training-sites-2024/consensus-with-broadband.json`.

In five date-grouped development folds on Apple GPU, at seeds 42–44,
the STEAD-initialized baseline classified 12/14 phases correctly at each
seed. Training with the new references from other dates scored 13, 13
and 14 out of 14. At −0.2 s both variants scored 11/14 for all seeds.
At +0.2 s the baseline scored 12/14 at all seeds, and adaptation scored
13, 14 and 14. The two CCUL P cases account for the gain at the center.
On the historical grouped three-class development clips, the matched
STEAD-initialized baseline scored 42, 42 and 41 out of 46. Adding the
14 arrivals scored 41, 42 and 42. These development results use small
samples and report separate datasets. See
`runs/qsis-training-sites-2024/site-grouped-development-with-broadband.json`
and `runs/qsis-training-sites-2024/legacy-dev-{42,43,44}/metrics.json`.

With the CWA initialization, the phase development baseline again
scored 12/14 at the center in all three seeds. Adaptation scored 14/14 in
all three, with 12, 12 and 13 at −0.2 s and 14 at +0.2 s. The historical
three-class development scores were 43, 41 and 43 for the baseline and
43, 42 and 43 for adaptation. The seed-42 CWA pair was refit on the full
development pool and saved under `runs/qsis-training-sites-2024/` as
`cwa-baseline` and `cwa-adapted`, with 74 and 88 training clips respectively.
The refit uses the original settings of 20 epochs and ±0.2 s shift and
±30° horizontal rotation. See
`runs/qsis-training-sites-2024/site-grouped-development-with-cwa-init.json`
and `runs/qsis-training-sites-2024/cwa-legacy-dev-{42,43,44}/metrics.json`.

The [Reviewed ISC Bulletin](https://doi.org/10.31905/D808B830) adds
neighbor-station context for ten of the 20 enrolled earthquakes. Four
events have reported arrivals at stations with rendered broadband
context. A separate query for the six QSIS instrument codes found no
matching stations. The Bulletin records times at the reporting instruments.
The projected times in `runs/qsis-training-sites-2024/isc/matches.json`
use the ak135 differential travel time and serve as cues for QSIS
waveform review. The compatible tentative QSIS candidates are CCUL
P at 4.55 s and S at 6.03 s on SITE24-012 and HW01 S at 17.60 s on
SITE24-006. They remain outside the fit and development scores. See
`runs/qsis-training-sites-2024/isc/candidates.json`.

The refit pair and the original CNN were fixed before the later
site-adaptation cohort downloaded waveforms. Of the 17 enrolled
earthquakes, six of seven at CCUL and zero of ten at HW01 had eligible
public recordings. Both BATS neighbors were available for all six
CCUL records. Two independent blind AI readers found no clear
station-specific first break. They independently suggested S near
16.65 s on SA26-003 and 15.55–15.65 s on SA26-014, both tentative.
The sole catalog-screened background candidate also lacked agreement.
The scored coverage is zero. See
`runs/qsis-prospective-site-adaptation/comparison.json` and
`runs/qsis-prospective-site-adaptation/tentative-arrivals.json`.

### Nearby P-Alert context

The [P-Alert network](https://palert.earth.sinica.edu.tw/help) has a
separate accelerometer at CCU (W18D, 0.23 km from CCUL) and at Jian
Elementary School (W475, 2.66 km from HW01). The public event plots
provide 25 Hz three-component traces. The context renderer matches
origins within four seconds and epicenters within 10 km and applies
the ak135 differential travel time between instruments. The references
still require a visible QSIS onset and two independent, model-blind
AI readings that agree within 0.6 s.

The new context covered 14 of 20 training-era events and added three
P and four S arrivals, including three CCUL S arrivals. The combined
training set now has 11 P and 10 S. The same-date grouped comparison
with CWA initialization uses the same 21 development arrivals for all
three conditions. At seeds 42–44, the centered baseline scores were
15, 15 and 14; the previous 14-reference adaptation scored 17 in all
three; the 21-reference adaptation scored 17, 17 and 18. At −0.2 s,
the previous adaptation scored 16, 15 and 17, and the new one scored
17, 18 and 17. Both scored 17 at +0.2 s across all seeds. In the
separate historical three-class development set, the expanded
adaptation scored 42, 43 and 43, compared with 43, 42 and 43 for the
14-reference adaptation. See
`runs/qsis-near-station-pilot/qsis-training-sites-2024/site-grouped-development.json`.

Four CCUL recordings in the already-enrolled site-adaptation test
also have nearby P-Alert context. The repeat blind review yielded
six time-compatible tentative candidates and zero confidently agreed
phase labels. For SA26-014, both reviewers located a candidate at
19.60 s with the closer data, after the earlier 15.55–15.65 s
suggestions. See
`runs/qsis-near-station-pilot/qsis-prospective-site-adaptation/consensus.json`.

### Prespecified three-site P-Alert comparison

`runs/qsis-prospective-palert/cohort.json` fixes the original
STEAD-initialized CNN and the paired CWA checkpoints before waveform
download. The adapted checkpoint uses the earlier 14 site arrivals.
The cohort samples 26 earthquakes from 26 dates, January 2025 to
May 2026, with public P-Alert catalog coverage and an epicentral
distance within 65 km. It excludes days from earlier public cohorts.
Every enrolled earthquake had an eligible QSIS recording and
nearby P-Alert context. Two independent AI readers agreed on 12 P
and 16 S picks across 18 earthquakes. They left 24 of the 52 enrolled
phase slots unresolved or overlapping. The catalog screen and a
separate two-reader review yielded 19 pre-event background clips
from 19 dates.

| Model | −0.2 s | Center | +0.2 s | Center P | Center S | Center noise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original CNN | 45/47 | 45/47 | 44/47 | 10/12 | 16/16 | 19/19 |
| CWA baseline | 45/47 | 45/47 | 44/47 | 10/12 | 16/16 | 19/19 |
| CWA site adaptation | 46/47 | 45/47 | 45/47 | 11/12 | 15/16 | 19/19 |

The adapted model's centered result was 95.74% with a date-cluster
95% interval of 89.13–100%. It classified 14/16 clips at CCUL,
21/21 at HW01 and 10/10 at RCEC. Both centered errors were
P/S confusions at CCUL. Date-resampled intervals of the paired
accuracy difference include zero for all offsets. The intervals
describe the scored subset. See
`runs/qsis-prospective-palert/comparison.json`.

A separate comparison on the archived local trigger windows scored
21/22 noise windows for all three models at all offsets. Those
windows come from two dates, with 21 windows on one date. See
`runs/qsis-prospective-palert/legacy-trigger-comparison.json`.

### Pre-arrival waveform history

Phase picking models such as [PhaseNeXt](https://doi.org/10.1186/s40623-026-02578-6)
and [DTPP](https://doi.org/10.1016/j.aiig.2026.100189) use temporal
context beyond a local onset. This experiment checks whether preceding
motion helps a two-second QSIS waveform classifier distinguish P and
S. The additional head uses nine seconds of earlier waveform and the
following second. A linear model learns low- and high-band energy,
vertical fractions and onset contrast from labelled CWA acceleration.
Its log-odds contribution is bounded. The original CNN makes the
noise decision and supplies the result when the waveform lacks history.

The clean history head classified 1,079/1,084 CWA validation phases.
Adding synthetic band-pass backgrounds to training gave 1,074/1,084
on the same clean phase clips and 1,024/1,084 on their separately
noised counterparts. This is a CWA source-domain check. In the QSIS
development comparison, neither history variant changed the centered
scores across three seeds. At −0.2 s, the clean variant raised the
21-phase site scores from 17, 18 and 17 to 19, 19 and 19. The
noise-trained variant scored 19, 19 and 18, and improved the shifted
historical three-class scores by one clip at both offsets in every
seed. Increasing the log-odds cap from 1.25 to 2.5 made no additional
prediction changes. See `runs/qsis-context-history/stress-development.json`.

The seven-site comparison in `runs/qsis-prospective-broader/cohort.json`
froze the original CNN, the 14- and 21-arrival CWA adaptations, and
the 21-arrival adaptation with the noise-trained history head before
downloading waveforms. It enrolled 47 earthquakes on 47 dates across
RCEC, CCUL, HW01, HW02, CHY01, PT01 and Fy0. All earlier public
benchmark dates and training-era site dates were excluded. There
were 43 usable QSIS recordings, with nearby P-Alert context for 40.
Two independent AI readers agreed on 13 P and 22 S picks from 23
earthquakes. Two separate readers agreed on 31 pre-event background
clips. The scored records represent 35 dates.

| Model | −0.2 s | Center | +0.2 s |
| --- | ---: | ---: | ---: |
| Original CNN | 57/66 | 62/66 | 63/66 |
| CWA adaptation with 14 site arrivals | 59/66 | 62/66 | 63/66 |
| CWA adaptation with 21 site arrivals | 62/66 | 63/66 | 63/66 |
| CWA adaptation with 21 site arrivals and history | 64/66 | 64/66 | 63/66 |

The new centered score was 96.97%, with a date-cluster 95% interval
of 92.19–100% across the scored subset. It classified 13/13 P,
20/22 S and 31/31 pre-event backgrounds. Both centered errors
were CCUL S-to-P confusions. Of 94 enrolled phase slots, 35 could
be scored. PT01 has background-only coverage. See
`runs/qsis-prospective-broader/comparison.json` and
`runs/qsis-near-station-pilot/qsis-prospective-broader/consensus.json`.

### Hard QSIS trigger training

The historical QSIS trigger diagnostic holds out whole UTC dates
in five folds. It uses 672 original-center clips from 22 training
dates after excluding possible catalog events and unusable records.
Labels retain the archive's noise assignment. At seeds 42–44, the
standard CWA adaptation rejected 592, 580 and 575 triggers out of
672. On the same 519 recordings with a full interior recut, scores
at the original centers were 488, 484 and 482. Moving the centers
past the taper and STA/LTA warmup raised those scores to 506, 504
and 503. See `runs/qsis-context-history/trigger-paired-centers.json`.

Adding 124 catalog-screened trigger clips from other training
dates raised original-center noise recall to 628, 651 and 644
out of 672. Repeating original noise clips for the same number of
additional training examples scored 549, 587 and 582. The new
model scored 42/46 in all three trials on the centered historical
three-class development set. It scored 42/46 at −0.2 s in each trial,
compared with 37, 38 and 38 without the extra trigger clips.
See `runs/qsis-context-history/hard-trigger-development.json` and
`hard-noise-phase-development.json` in the same directory. Use
`seismictorch qsis fit --trigger-pool .../hard-trigger-development.json`
to reproduce the 219-clip CWA-initialized refit using all 21 accepted
2024 phase clips. Its separate frozen archival trigger check scored
22/22 at the center, compared with 21/22 before the trigger training.
Those examples come from two dates, with 21 from one date.
See `runs/qsis-context-history/archived-trigger-test.json`.

A separate legacy development comparison held out both the target
noise region and test UTC dates. The hard-trigger recipe scored
455, 445 and 453 of 476 campus triggers, 96, 106 and 107 of 116
SPHSa triggers, and 75, 76 and 76 of 80 triggers from the remaining
sites. Each fold used noise examples from other regions and dates.
The phase training pool was fixed. These retain the catalog-screened
archival noise labels. See `runs/qsis-context-history/noise-site-development.json`.

### Combined history and trigger training

The original CNN, the 21-arrival CWA adaptation and the hard-trigger
refit were fixed before new waveforms were downloaded. Both CWA
checkpoints were scored with and without the same noise-trained
history head. The primary batch enrolled 44 earthquakes on unique
dates. It yielded 28 aligned QSIS and nearby P-Alert records.
An extension enrolled another 20 dates after the observed archive
unavailability and before any model scoring. It yielded 17 records.
Earlier public and training-era event dates were excluded. Phase
reviews by two independent AI readers identified 17 P and 14 S
arrivals across 22 earthquakes. A separate two-reader review
accepted 30 pre-event background clips. All 35 rendered background
candidates had a STA/LTA ratio below the legacy threshold of 3.01.

| Fixed model | −0.2 s | Center | +0.2 s |
| --- | ---: | ---: | ---: |
| Original CNN | 58/61 | 59/61 | 58/61 |
| 21-arrival CWA adaptation | 61/61 | 60/61 | 60/61 |
| 21-arrival CWA adaptation with history | 61/61 | 59/61 | 59/61 |
| Hard-trigger refit | 60/61 | 60/61 | 60/61 |
| Hard-trigger refit with history | 60/61 | 60/61 | 59/61 |

The combined model scored 42/42 in the primary batch and 18/19
in the extension at center. This yields 98.36% across 36 scored
dates, with 16/17 P, 14/14 S and 30/30 pre-event backgrounds.
The date-resampled 95% interval for the centered scored subset is
94.55–100%. The history head offered no additional accuracy on
the new batches. Phase references cover 31 of 128 enrolled phase
slots. See `runs/qsis-prospective-combined/pooled-comparison.json`.

## Recording-edge trigger audit

The standard QSIS preprocessing tapers the first and last 5% of the
recording before applying a causal fourth-order 0.5 Hz high-pass filter.
At SPHSa, 73 of 116 catalog-screened archived trigger windows with usable
components overlapped this tapered region. With the same filter cutoff
and order, first-sample filter initialization and the taper removed, none
of these 73 windows crossed the 3.01 STA/LTA threshold within one second
of the old center. Another 43 windows were interior and all still crossed
near the old center. On a separate set of reviewed P and S arrivals, the
same threshold was reached from 0.2 s before through 2 s after 24/30 P and
23/36 S arrivals with either filter. Those numbers measure trigger
coverage and are separate from classification at a specified center.

Two independent readers inspected the original and both filtered views
of 19 SPHSa clips across 17 dates. They agreed that 18 were noise-like.
Fourteen showed an amplitude ramp created by the taper and four showed
isolated impulses. The remaining clip, SPT-14, showed coherent motion
across the three components. Its source remains unresolved and its
archived label was left in place.

In three exploratory runs, excluding SPHSa and its evaluation dates from
the training noise, the tapered classifier called 98, 105 and 107 of the
116 archived windows noise. The steady-start classifier called 108, 113
and 112 noise at the same centers. For the 18 independently accepted
noise clips, the respective counts were 13, 14 and 15 versus 17, 18 and 18.
Both variants classified all 43 interior windows as noise in every run.
These site experiments used archived class labels that can include
unresolved transient signals. See `runs/qsis-filter-boundaries`.

The fixed steady-start CNN used 219 training clips and 20 epochs on
Apple MPS, with the same source windows as the hard-trigger CNN. Its
checkpoint and receipt are in
`runs/qsis-filter-boundaries/cwa-hard-steady-start`. Use
`seismictorch qsis fit --waveform-filter steady-start` with the existing
hard-trigger fit arguments. The classifier and frozen scorer read the
filter from the receipt. Earlier receipts default to the tapered version.

A supplemental blind review recovered 13 additional archived trigger
noise windows from seven dates absent from either candidate's noise
training. Both readers marked all 13 as noise-like. Including the earlier
22 independently reviewed trigger windows, both models classified 33/35,
35/35 and 35/35 as noise at −0.2 s, center and +0.2 s. The 35 clips span
eight dates, with 22 on one date. See
`runs/qsis-filter-boundaries/trigger-review-extension/comparison.json`.

### Fixed comparison on later QSIS events

The original CNN, the CWA hard-trigger CNN and the steady-start refit were
locked before enrolling 25 additional earthquakes on separate UTC dates.
Usable QSIS and nearby P-Alert context was available for 23 earthquakes
at six sites. Two independent AI waveform reviews agreed on 7 P and
13 S arrivals across 14 earthquakes. They also agreed on 18 of the 19
catalog-screened pre-event background clips. All 19 had STA/LTA below
3.01. There were no scored CHY01 observations and PT01 contributed one
background clip. The remaining phase slots lack a scored reference.

| Fixed model | −0.2 s | Center | +0.2 s |
| --- | ---: | ---: | ---: |
| Original CNN | 36/38 | 35/38 | 36/38 |
| CWA hard-trigger CNN | 37/38 | 36/38 | 36/38 |
| CWA steady-start CNN | 37/38 | 36/38 | 36/38 |

The two CWA models made identical predictions on this cohort at all three
offsets. At center they correctly classified 6/7 P, 12/13 S and 18/18
background clips. One CCUL event accounts for both phase errors. Its
P and S labels were interchanged. The date-resampled interval is
82.86–100% at center, across 20 scored dates. The 14 tentative arrivals
with compatible timing are available alongside the agreed picks in
`runs/qsis-near-station-pilot/qsis-prospective-filter/consensus.json`.

Across this and the two earlier date-disjoint combined cohorts, the
same original CNN classified 94/99 and the same tapered CWA hard-trigger
CNN classified 96/99 at center. The latter has 22/24 P, 26/27 S and
48/48 backgrounds, with a date-resampled interval of 91.67–100% across
56 scored dates. The pooled score is descriptive and represents readable
phase arrivals and screened pre-event backgrounds. The steady-start
checkpoint was frozen only for the latest cohort. See
`runs/qsis-prospective-filter/shared-checkpoint-summary.json` for the
original and hard-trigger aggregate.

For candidate follow-up, the [Reviewed ISC Bulletin](https://isc.ac.uk/iscbulletin/)
uniquely matched 12 of the 13 waveform dates within its review coverage.
Nine targets had nearby time-defining P context and ten had S context.
Two new readers reviewed ten target plots with that context. The linked
candidate list records 35 phase slots. Six clear picks agree across both
contexts, two more have closely spaced tentative readings, and four
have materially different times. The ISC context added a tentative P
candidate for SFH-010 at 15.85 s. See
`runs/qsis-prospective-filter/arrival-candidates.json`. These entries
retain the separate times where the reviewers disagreed.

Two further prespecified October–November 2026 dates yielded aligned
records at PT01 and RCEC. Two readers agreed on one P and two S
candidates, plus a tentative P at PT01. Each fixed checkpoint correctly
classified the three agreed phase clips at all three offsets. Neither
date yielded a CWA-matched background example under the same screening
rule. See `runs/qsis-prospective-nov2026/comparison.json`.

The original `dada_QSIS_test` examples also contain the CWA events of
2022-09-18 and 2023-08-10. For the latter, two readers agreed on the
SPHS S onset at 32.675 s; the 18.51 s P onset is tentative.
The 2022 RCEC arrivals remain unresolved after review in two passbands
with wider coverage. See
`runs/qsis-filter-boundaries/legacy-event-candidates/candidate-arrivals.json`
and the expanded review in that directory.

## CWA pretraining with real QSIS backgrounds

The CWA source-domain experiment used the manually picked TSMIP records.
It held out 73 earthquakes for validation and 73 for testing, with 219
earthquakes used for training. Noise donors were taken from quiet clips
on archived QSIS training dates after the catalog and waveform checks.
The donor splits were disjoint by date, with 14 training, four validation
and four test dates. Both models started with the same STEAD weights.
The augmented model adds a random background to half of the P and S
training windows at 4–18 dB relative RMS. Both variants use the same
shift and rotation augmentation. Checkpoints were chosen by the mean
balanced accuracy on the original and 10 dB validation conditions.
Signal-to-noise ratios compare the original recorded phase clip with
the added donor waveform.

| CWA training | Seed | Original clips | Plus 10 dB noise | Plus 0 dB noise | Low-contrast P |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 42 | 1399/1422 | 1387/1422 | 1043/1422 | 115/119 |
| With QSIS backgrounds | 42 | 1401/1422 | 1386/1422 | 1097/1422 | 116/119 |
| Standard | 43 | 1402/1422 | 1389/1422 | 1044/1422 | 116/119 |
| With QSIS backgrounds | 43 | 1404/1422 | 1387/1422 | 1104/1422 | 117/119 |

Only the phase clips received extra noise. The test outcomes at 0 dB
reflect a controlled mixture of the recorded CWA phases and real QSIS
waveforms. The low-contrast P stratum is the lowest quartile of the
ratio of post-center to pre-center RMS on original CWA P clips. Source
weights, event groups, donor dates and confusion matrices are under
`runs/qsis-cwa-noise`.

Using the same set of 219 QSIS development training sources and
preprocessing for each variant, QSIS fine-tuning held out groups of
entire event dates and excluded training arrivals within 120 s.
Only previously reviewed P and S labels were scored. The available
reference set consisted of 24 historical arrivals and 21 later public
arrivals. The two CWA pretraining variants both scored 20/24 and 18/21
at the original centers, at each seed. The earlier CWA initialization
scored 20/24 and 21/24 on the historical phases, and 18/21 at both seeds
on the later phases. See `runs/qsis-cwa-noise/transfer.json`.

The November 2026 [MGPP paper](https://doi.org/10.1016/j.engappai.2026.116339)
learns across explicit temporal scales for arrival picking. A local
two-second classifier experiment paired 0.2- and 0.5-second waveform
chunks in a gated residual branch. It adds 1,695 parameters to the CNN.
The CWA test comparison reused the earlier conditions for an exploratory
architecture check. The branch changed the 0 dB correct count by zero
and one at the two seeds, and made the same decisions as the
noise-trained CNN on the grouped QSIS development phases at all three
offsets. Both variants remain development experiments. Model weights
for the independent QSIS benchmarks are unchanged. See
`runs/qsis-cwa-noise/granularity-transfer.json`.

## CCUL orientation and additional arrival references

A further event selection reserved five CCUL dates before waveform
acquisition and assigned 17 dates to development. Earlier training and
evaluation dates were excluded. Nearby events within 48 hours and 15 km
were thinned before assigning the two sets. Three primary QSIS instruments
provided six readable development recordings. Searching the remaining
five instruments on the originally unavailable dates recovered three
more. Two independent AI readers agreed on three P and four S references.

Adding those seven references to the existing 219 training sources yielded
226 windows. On the 15 available CCUL development phases, holding out full
dates with a 120 s embargo, the centered score stayed 10/15 and 11/15 at
training seeds 42 and 43. The reserved dates yielded three recordings,
one agreed S reference and three reviewed pre-event backgrounds. Both
the previous and extra-reference CNNs scored 4/4 at all three offsets.
The remaining reserved dates lacked waveforms even after checking the
other instruments. See `runs/qsis-ccul-extra/development.json` and
`runs/qsis-ccul-holdout/comparison.json`.

### Physical vertical in the target waveforms

Raw development recordings at CCUL include approximately 1 g on X for
5AE10, 5AE35 and 5AE48. Several P/S confusions coincide with the model's
assumption that Z is vertical. A stable baseline gives a local physical
vertical before detrending. The [Phidgets accelerometer documentation](https://www.phidgets.com/?prodid=1026)
describes its static gravity response. The public QSIS station metadata
also reports azimuth and inclination for each component.

The deployment metadata and gravity agree for 5AE10, 5AE35, 5AE48 and
5AE62. The inclination for 5AE78 and 5AEB8 differs from the observed
raw gravity in the downloaded development waveforms. A sensor code's
deployment interval alone therefore does not establish a consistent
waveform coordinate convention. The metadata and baseline observations
are preserved in `runs/qsis-ccul-extra/axis-metadata-diagnostic.json`.

The gravity preprocessing uses the median of six one-second baseline
medians from seconds 1–7 of the original recording. It accepts norms of
900–1100 gal with at most 8 gal drift. This baseline must precede the
two-second classification window. A proper three-dimensional rotation
aligns the estimated upward direction with Z. The waveform QC still
uses the original components. Fitting, scoring and direct inference
share the implementation in `src/seismictorch/data/orientation.py`.

| CCUL development preprocessing | Seed 42 | Seed 43 |
| --- | ---: | ---: |
| Original axes | 10/15 | 11/15 |
| Gravity from each recording | 13/15 | 12/15 |
| Gravity plus dated station fallback | 14/15 | 14/15 |

This comparison uses the same source recordings, labels and event folds.
At −0.2 and +0.2 s, the per-record method scored 13/15 and 12/15
at both seeds. The fallback scored 13/15 and 15/15 at seed 42,
and 13/15 and 14/15 at seed 43. The dated station fallback uses
gravity from other development dates. Using the verified inclination
values in those intervals yielded the same counts. See
`runs/qsis-ccul-extra/canonical-orientation-development.json` and
`metadata-orientation-development.json`.

The held-out target plots were also rendered with corrected vertical
and horizontal components. Two new independent readers agreed on P
at 22.45 s and S at 38.0 s for CCH-003. The other two phase records
remain unresolved or tentative. On the corrected-reference set with
three backgrounds, the uncorrected CNNs scored 3/5 at every offset.
The station-fallback candidate scored 4/5, 4/5 and 5/5 at −0.2 s,
center and +0.2 s. This is a follow-up assessment with a new review
context. The prior reference packet and its 4/4 score are retained.
See `runs/qsis-ccul-holdout/oriented-comparison.json`.

### Transfer across archived public references

Six previously reviewed, date-disjoint public cohorts contain 215
scored clips. The fixed original CNN scored 204/215, and the hard-trigger
CWA CNN scored 208/215 at center. Applying the station-fallback
orientation candidate scored 200/215. Most new errors occurred on CCUL
recordings whose static gravity had already been removed. The matching
station epoch metadata was insufficient to validate their coordinates.
See `runs/qsis-ccul-extra/orientation-generalization.json`.

Refitting with gravity from each recording and the same 226 training
sources scored 211/215, 210/215 and 209/215 at the three offsets.
At center this is 47/50 P, 65/67 S and 98/98 pre-event backgrounds.
The per-record candidate is saved under
`runs/qsis-ccul-extra/gravity-record-candidate`. Its receipt carries
the preprocessing choice and has an empty station calibration list.
The six-cohort result reuses prior evaluation references for an
exploratory transfer check. Its event dates remain outside the
candidate's training set. See
`runs/qsis-ccul-extra/record-orientation-generalization.json`.

The matched refit with the same 226 training sources and original axes
scored 210/215, 208/215 and 208/215 at the three offsets. This isolates
the per-record correction from the seven-reference addition in the
archived transfer comparison. See
`runs/qsis-ccul-extra/record-orientation-generalization-controlled.json`.

The same per-record candidate classified 34/35, 35/35 and 35/35 of
the previously reviewed archived trigger windows as noise at −0.2 s,
center and +0.2 s. Those windows span eight dates absent from training,
with 22 windows on one date. See
`runs/qsis-ccul-extra/record-gravity-archived-noise.json`.

### New waveform acquisition after fixing the per-record candidate

An additional 18 earthquake dates were fixed after training, before
downloading new evaluation waveforms. The selection used the archived
QSIS and P-Alert catalogs, excluded every earlier training and public
evaluation date, and placed a 120 s embargo around earlier events.
It included magnitude 3.2 and above, depth at most 130 km and distance
at most 130 km across seven intended sites. Nine dates yielded usable
QSIS recordings and nearby P-Alert context, at CCUL, CHY01, RCEC, HW02
and Fy0. The target plots use physical vertical alignment only where
the recording has a stable gravity baseline. The reviewers remained
blind to classifier outputs.

Two independent readers agreed on three P and five S onsets across
five dates and accepted nine pre-event background windows. None of
those backgrounds reached the 3.01 STA/LTA trigger threshold. There
were no readable SPHSa or PT01 records. CHY01 contributed S and
background windows, and Fy0 contributed backgrounds.

| Fixed checkpoint | −0.2 s | Center | +0.2 s |
| --- | ---: | ---: | ---: |
| Original CNN | 17/17 | 16/17 | 16/17 |
| CWA hard-trigger CNN | 16/17 | 17/17 | 16/17 |
| Gravity from each recording plus seven CCUL references | 17/17 | 17/17 | 17/17 |

The candidate's score covers all three classes on nine scored dates.
Eight of the 36 enrolled phase slots have clear agreed references.
This small sample and the quiet background selection limit the scope
of the result. See `runs/qsis-prospective-gravity/comparison.json` and
the candidate suggestions in
`runs/qsis-near-station-pilot/qsis-prospective-gravity/consensus.json`.

### Same-building follow-up on unresolved phases

The [October 2026 EQNet2 preprint](https://arxiv.org/html/2610.17404v1)
uses station geometry and shared multi-station features to recover weak
arrivals. Its reported experiments concern Northern California. The
local follow-up applies the simpler idea of comparing contemporaneous
motion when checking QSIS references. It provides additional review
context for the existing single-recording classifier.

Eight records from the latest enrolled cohort were plotted with all three
target components, three frequency bands, vector-energy envelopes from
two or three same-building QSIS instruments, and the closest usable
independent P-Alert trace. Six of the records had unresolved phases, and
two HW02 records were controls with clear prior arrivals. All clocks
were aligned from their waveform acquisition metadata. Two new,
independent AI readers viewed these plots without classifier outputs.

The additional agreements are Fy0 P at 20.0 s on GPH-012, RCEC S at
33.3 s on GPH-013 and CCUL P at 22.05 s on GPH-015. Five earlier clear
arrivals were confirmed within the original 0.6 s tolerance. Six
phase slots have compatible tentative times and two remain unresolved.
See `runs/qsis-prospective-gravity/array-followup/candidates.json`.

Keeping the original reference times and backgrounds and appending the
three new agreements yields 20 clips across the same reserved dates.
The per-record gravity candidate scores 19/20 at all three offsets,
with 4/5 P, 6/6 S and 9/9 backgrounds at center. GPH-015 P is classified
as S. The original CNN scores 20/20, 19/20 and 19/20, and the hard-trigger
CWA CNN scores 18/20, 19/20 and 18/20. The initial frozen 17-clip result
is retained separately. This supplemental comparison reports the
additional reference context. See
`runs/qsis-prospective-gravity/array-followup/supplemental-comparison.json`.

A further development comparison uses all 52 reviewed training phase
references in five date-held folds with a 120 s embargo. It compares
the gravity CNN with the existing clean and noise-trained CWA history
heads, using their original 1.25 log-odds bound.

| Development variant | Seed | −0.2 s | Center | +0.2 s |
| --- | ---: | ---: | ---: | ---: |
| Gravity CNN | 42 | 45/52 | 46/52 | 46/52 |
| With clean history | 42 | 47/52 | 47/52 | 45/52 |
| With noise-trained history | 42 | 48/52 | 49/52 | 47/52 |
| Gravity CNN | 43 | 46/52 | 47/52 | 46/52 |
| With clean history | 43 | 46/52 | 47/52 | 45/52 |
| With noise-trained history | 43 | 46/52 | 48/52 | 46/52 |

Selecting the noise-trained history variant from these development
results left all 20 supplemental evaluation predictions unchanged
at each offset, including the GPH-015 P-to-S error. See
`runs/qsis-history-orientation/evaluation.json`.
The history helper checks original-channel repeated samples before
applying the physical rotation. The 156 evaluated development contexts
have identical features before and after this QC-order correction.

### Reproduce the gravity candidate

The regular training entry point accepts the two additional reviewed
CCUL cohorts. It reproduces the same ordered 226 windows, initial
weights, MPS settings and all learned parameter values as the earlier
experiment. The verified run is saved in
`runs/qsis-ccul-extra/cli-gravity-reproduction`.

```sh
seismictorch qsis fit \
  --split runs/qsis-benchmark/split.json \
  --review runs/qsis-review-balanced \
  --init runs/qsis-benchmark/cwa-pretrain-42/best.pth \
  --validation-manifest runs/qsis-benchmark/validation-review/manifest.json \
  --test-manifest runs/qsis-benchmark/test-review/manifest.json \
  --proposals runs/qsis-benchmark/redpan-train.json \
  --public-review runs/qsis-training-sites-2024/download.json \
    runs/qsis-training-sites-2024/phase-manifest.json \
    runs/qsis-near-station-pilot/qsis-training-sites-2024/merged-training-consensus.json \
  --trigger-pool runs/qsis-context-history/hard-trigger-development.json \
  --extra-public-review runs/qsis-ccul-extra/download.json \
    runs/qsis-ccul-extra/phase-manifest.json runs/qsis-ccul-extra/references.json \
  --extra-public-review runs/qsis-ccul-extra-supplement/download.json \
    runs/qsis-ccul-extra-supplement/phase-manifest.json \
    runs/qsis-ccul-extra-supplement/references.json \
  --waveform-filter steady-start --gravity-align \
  --device mps --epochs 20 --seed 42 --output runs/qsis-gravity-refit
```

Three off-catalog waveform requests at two timestamps did not yield
data from the public QSIS event endpoint. A control request for an
existing catalog event returned 60,000 aligned samples. The request
receipts are in `runs/qsis-arbitrary-time/waveform-only-availability.json`.
This check did not supply a new set of ambient trigger recordings.

### Independently reviewed interior trigger audit

A fixed selection used at most two recordings from each of the 22
historical development noise dates. Every chosen center exceeded
3.01 in a one-second/five-second STA/LTA on the steady-start filtered
vector magnitude. Centers began at least eight seconds into the
recording and passed the two-second raw waveform QC. The selection
excluded any catalog origin within 300 s and 400 km using all
available magnitudes in the archived CWA catalog. It was fixed before
review or model scoring.

Two independent AI readers reviewed all 43 sampled windows in raw,
untapered high-pass and wider context views. They both accepted 40
as noise. QIT-010 had coherent decaying motion, and QIT-023 and
QIT-042 lacked a second confident noise judgment. Those three remain
outside the noise score. The accepted set spans 22 dates and six
sites, with 27 of the originally sampled records from RCEC.

In five date-held folds, every training source on an evaluation date
was excluded, including phase sources from that date. Both training
seeds 42 and 43 used the same 226-window source recipe and CWA
initialization. The original tapered, steady-start and gravity-aligned
variants each classified all 40 accepted windows as noise at −0.2 s,
center and +0.2 s. This is a development cross-validation of noise
rejection, with manual waveform screening by the two AI readers.
See `runs/qsis-interior-trigger-audit/comparison.json` and its linked
manifest and reviews.

## Three-dimensional rotation augmentation

A fixed development comparison used the same 52 reviewed phase clips,
five whole-date folds, 120-second event embargo and two training seeds.
Each fold used CWA initialization and 20 MPS epochs. The additional
rotation used an independent NumPy random stream to preserve the existing
PyTorch sampling and temporal augmentation stream. Half of the training
examples received a proper signed axis permutation or a continuous SO(3)
rotation. Both variants were compared with the unchanged training recipe.

| Recipe | Seed 42 CNN, center | Seed 43 CNN, center | Seed 42 with history | Seed 43 with history |
| --- | --- | --- | --- | --- |
| Existing gravity recipe | 46/52 | 47/52 | 49/52 | 48/52 |
| Signed axis permutations | 48/52 | 50/52 | 50/52 | 50/52 |
| Continuous rotations | 49/52 | 50/52 | 49/52 | 48/52 |
| Permutations only without measured gravity | 49/52 | 49/52 | 50/52 | 50/52 |

The discrete recipe was fixed at seed 42 before supplemental scoring.
Despite the development improvement, its CNN scored 203, 207 and 201
out of 215 reused public reference clips at -0.2 s, center and +0.2 s.
The existing gravity candidate scored 211, 210 and 209. On the recent
20-clip supplemental set the new CNN scored 20, 20 and 17, while the
history combination scored 20, 19 and 17. Two background windows became
P predictions at the positive timing offset. All three offsets use
re-cuts from the original recording.

A follow-up restricted permutations to training recordings without
measurable gravity. This retained physical vertical alignment on 104
training clips, including 99 noise clips. It scored 201, 208 and 202
out of the same 215 references, and 19, 20 and 17 out of the supplemental
20. Neither rotation recipe replaces the existing gravity candidate.
These results show that the development phase improvement did not
transfer reliably to time-shifted noise and the larger public reference
set. The historical reference scores are descriptive reuse, since those
cohorts had already been inspected in earlier experiments.

The protocols, source hashes, development fold decisions, frozen weights
and supplemental comparisons are archived in
`runs/qsis-rotation-augmentation` and `runs/qsis-conditional-rotation`.
The history model was fixed from the earlier CWA stress training, with
a 1.25 log-odds cap. No supplemental reference entered training.

## Later November 2026 candidate arrivals

A QSIS catalog refresh for November 17–20 returned three events on two
new dates. Before any waveform download, a confirmation cohort fixed
the largest catalog event per date, up to three closest sites within
90 km and up to three active instruments per site. Every enrolled date
was absent from waveform training and earlier public cohorts. The fixed
model remained the original per-record gravity candidate. Neither of the
new rotation variants was promoted.

The frozen selection contained five event/site pairs. Seven recordings
were available across four pairs. HW01 was unavailable. The two event
dates were November 17 and 18, 2026. The P-Alert query for the same period
returned no listed event, so the review used the target XYZ waveforms,
three frequency bands and contemporaneous same-site instruments where
available. Target selection used the lowest usable station code, before
phase review. Single-instrument sites remain in the coverage count.

Two independent AI readers each viewed all eight review images. Both
marked two S onsets as tentative and all remaining phases as unresolved.
Their S candidate times agreed within 0.1 s.

| Reference | Target/site | S candidate, seconds after origin | Reader times | Status |
| --- | --- | --- | --- | --- |
| LNP-001 | 5AF43 / HW02 | 22.15 | 22.1, 22.2 | Tentative |
| LNP-003 | 5AEA0 / CHY01 | 11.25 | 11.2, 11.3 | Tentative |

The gradual energy increase prevented either reader from assigning a
clear first arrival. There are zero scoring references among eight
available phase slots and ten enrolled slots. No accuracy is calculated
for this cohort. The candidate times remain available for further
arrival analysis. They do not enter model training or the 95% claim.

See `runs/qsis-prospective-late-nov2026/candidates.json` for both readers'
reasons, `comparison.json` for explicit coverage and scoring status, and
`cohort.json` for the frozen model and enrollment source hashes.

## Adjacent-window probability averaging

A development-only comparison retained the frozen gravity training recipe
and compared one two-second cut with equal softmax probability averaging
at center minus 0.2 s, center and center plus 0.2 s. The five recut centers
from -0.4 to +0.4 s also allowed the same comparison at nominal offsets
of -0.2, 0 and +0.2 s. If a neighbor failed raw QC, that case used its
original center prediction. No case was discarded for a missing neighbor.
The fixed CWA stress history head was tested on the nominal-center
context, with the same 1.25 log-odds cap.

| Phase development recipe | Seed 42 center | Seed 43 center |
| --- | --- | --- |
| Single cut | 46/52 | 47/52 |
| Single cut plus history | 49/52 | 48/52 |
| Three-cut mean | 47/52 | 46/52 |
| Three-cut mean plus history | 47/52 | 46/52 |

The phase folds exclude whole dates and a 120-second event neighborhood.
A separate date-held evaluation of the 40 independently reviewed interior
noise triggers scored 40/40 for every variant, at both seeds and all
three offsets. The single-cut results reproduce the earlier phase and
noise development baselines. Averaging did not improve the centered
phase result consistently, so it was not evaluated on the reserved
cohort or added to production prediction. The protocol and individual
fold decisions are in `runs/qsis-temporal-average`.

## Extra training stations with independent waveform review

The original training split contains additional instruments recording the
same earthquakes and same building as existing manually reviewed P/S
references. A training-only scan found 117 event/station/phase candidates
with support from such a reviewed instrument within 300 m. The source
arrival times had to agree within 0.6 s after aligning absolute SAC
clocks. Previously used station/phase labels and raw one-second channel
dropouts were excluded. Candidate selection used up to two highest-SNR
recordings per event and phase, without classifier predictions. It
produced 24 target plots across ten training earthquakes, with eight P
and sixteen S candidates.

Two independent AI reviewers each viewed all 24 target plots. Every
plot included target XYZ in three frequency bands and the contemporaneous
reviewed reference instruments. They agreed on eight clear target arrivals
within 0.6 s, comprising three P and five S from four events. The other
sixteen remain tentative, unresolved or without two clear judgments.
The eight agreed labels are additional training references. Their source
hashes, target times, images and both readers' reasons are retained in
`runs/qsis-training-array-expansion/references.json`.

The development comparison kept the original 52 phase references as its
headline set. Each fold excluded every source on the held-out dates and
extra phase records within 120 s of the held events. New phase references
were also evaluated separately in their held-date fold. The CWA
initialization, 20 epochs, augmentation and gravity preprocessing were
unchanged.

| Recipe, original 52 phases at center | Seed 42 | Seed 43 |
| --- | --- | --- |
| Existing CNN | 46/52 | 47/52 |
| CNN with eight additional arrivals | 48/52 | 47/52 |
| Existing CNN plus history | 49/52 | 48/52 |
| Additional-arrival CNN plus history | 49/52 | 48/52 |

Both seeds classified the eight extra held-date arrivals correctly at
all three offsets. The separate 40-trigger, date-held noise evaluation
remained 40/40 for both seeds and every offset. Exact centered input
hashes confirmed that no added phase input duplicates an existing input.

A fixed seed-42 MPS refit used 234 sources, comprising 36 P, 52 S and
146 noise clips. The CNN classified all 20 recent supplemental references
correctly at -0.2 s, center and +0.2 s. With the fixed history head it
scored 20, 20 and 18. The recent result reuses the reserved-date
references that have already been inspected, so it is a supplemental
comparison rather than a new untouched confirmation.

On the larger 215-clip archived transfer set, the new CNN scored 207,
208 and 207 across the same offsets. At center it classified 48/50 P,
62/67 S and 98/98 background clips correctly. The existing gravity
candidate scored 47/50 P, 65/67 S and 98/98 background at center, or
210/215 overall. Because the recent weak-P gain accompanied a larger
S-to-P regression on the archived references, the existing candidate
remains the default. On the 35 archived held-out trigger windows, the
new candidate scored 32, 35 and 35, versus the existing candidate's 34,
35 and 35. Those eight noise dates were verified absent from training.
The additional negative-offset false positives also argue against
promotion. The new model is retained for further development at
`runs/qsis-training-array-expansion/candidate`. Its evaluation receipt
links the raw reference, fold, checkpoint and comparison hashes.

## Fixed training-pool probability ensemble

The 226-source and 234-source gravity recipes were refit separately in
identical held-date folds at seeds 42 and 43. The ensemble used the equal
mean of their softmax probabilities. It fitted no mixing coefficient.
The original 52 phase references stayed fixed, and the separate noise
folds excluded every training source on each evaluated noise date.

| Original 52 phases, center | Seed 42 | Seed 43 |
| --- | --- | --- |
| Current single CNN | 46/52 | 47/52 |
| Expanded single CNN | 48/52 | 47/52 |
| Equal probability mean | 48/52 | 47/52 |
| Current CNN plus history | 49/52 | 48/52 |
| Expanded CNN plus history | 49/52 | 48/52 |
| Equal mean plus history | 49/52 | 48/52 |

All six variants scored 40/40 on the reviewed interior noise at both
seeds and all three timing offsets. The component predictions reproduce
the earlier single-model experiments exactly. The ensemble did not
improve the centered phase development score over the expanded model.
At the negative offset, its history combination improved seed 43 from
46/52 to 48/52 relative to current-history, with seed 42 unchanged.

The two already frozen checkpoints were also compared descriptively on
the existing public reference sets. On the 215-clip archived set the
equal mean scored 210, 211 and 207 at -0.2 s, center and +0.2 s,
versus the current model's 211, 210 and 209. On the recent supplemental
20 references it scored 20, 19 and 19, both with and without history.
The centered gain on archived references accompanied weaker positive
offset behavior and no improvement in the recent centered score.
The default remains the current single gravity model. See
`runs/qsis-training-pool-ensemble` for fixed protocols, individual
component logits, comparisons and reproduction checks.

## IES hybrid archive confirmation

The primary [IES Building Seismic Response Dataset](https://zenodo.org/records/17469923)
was published in October 2025. Its 225,569,391-byte archive passed the
publisher's MD5 checksum and was inventoried without executing any supplied
scripts. It contains 3,069 QSIS SAC components covering 112 event IDs
on 64 dates, plus SMA-1 MiniSEED, text and metadata. It supplies event
recordings rather than an established continuously labelled background
benchmark. No QSIS P/S arrival label file was found in the supplied metadata.

Header and catalog screening found 15 event dates absent from the original
split, all candidate training dates and every previous public cohort.
The model identities were frozen before archive acquisition. Before
waveform inspection, enrollment chose the largest listed event per date,
two lowest complete instrument codes as targets and up to two further
codes as context. It required synchronized 100 Hz XYZ triplets, at least
30 seconds before origin, 120 seconds afterward and a 120-second embargo
from previous events. All selected component clocks and SAC reference
origins agreed with the supplied event metadata. Some SAC headers lack
station coordinates; those search-window estimates use the archive
README's documented IES coordinates, with the source recorded explicitly.
The original SAC bytes are preserved.

There were 29 enrolled target recordings across 15 dates. One failed raw
waveform QC, leaving 28 rendered targets and 58 enrolled phase slots.
Two independent AI reviewers each viewed every target plot, using three
frequency bands, local multi-instrument energy and a wider target overview.
Both agreed on nine P and six S onsets within 0.6 seconds, from six event
dates. The other phase slots remain unresolved in the coverage record.
No classifier output was provided to the reviewers.

A separate CWA catalog screen queried all available magnitudes for each
date and confirmed the enrolled event before selecting a background.
It excluded candidate centers within 300 seconds after nearby catalog
origins or two seconds before them, within 400 km. Selection took the
maximum valid 1-second/5-second STA/LTA center in the first eligible target
per date, between -104 and -12 seconds from origin, at least eight seconds
into the recording. Raw dropouts were excluded over detector support and
the classification clip. It yielded 14 plots. Two independent readers
accepted 12 as noise, left the quantized staircase at IBG-026 unclear and
disagreed on the nearby transient at IBG-020. Both exclusions are retained.

Only after consensus was frozen were the current and expanded models
scored on the same 27 references, comprising 9 P, 6 S and 12 backgrounds.
Every cut passed raw QC at all three fixed offsets, and no centered input
exactly duplicated a current training input. Both frozen models scored
27/27 at -0.2 seconds and center, and 26/27 at +0.2 seconds. IHP-017 P
became S at the positive offset. The checkpoint identity is unchanged, and all 15 evaluation
dates remain outside waveform fitting. The expanded model's earlier
S and noise regressions still favor retaining the current single model.

The new cohort contributes 12 scored dates, disjoint from the earlier
nine-date primary cohort. The same unchanged weights score 44/44 at
center across the two primary cohorts. Including the three earlier
array-context additions gives 46/47 at center and 45/47 at +0.2 seconds.
The sources, both sets of reviews, unchanged checkpoint identities and
recomputed metrics are linked in `runs/qsis-ies-hybrid/verification.json`.
The normal command-line inference path also reproduced one P, one S and
one noise result from this cohort. These are independently reviewed
arrival-window results; unresolved phases and continuous picking remain
outside the accuracy denominator.

## 2026 research

- [DTPP](https://doi.org/10.1016/j.aiig.2026.100189) uses dilated depthwise
  temporal filters to retain a large receptive field at low cost. This
  motivated the local TCN classifier. DTPP itself predicts phase times.
- [U-Trans](https://doi.org/10.1038/s41598-026-41454-x) provides frozen
  representations learned from three-component 60-second seismograms.
  The development comparison above uses the authors'
  [published weights](https://github.com/omarmohamed15/U-Trans).
- [Regional fine-tuning of phase pickers](https://doi.org/10.1785/0220260259)
  evaluates PhaseNet, EQTransformer and GPD on analyst-reviewed held-out
  events and sets thresholds on validation. Reported scores vary by phase
  and matching tolerance. Its P/S picking F1 and the QSIS clip accuracy
  measure different tasks.
- [SeismoDual](https://doi.org/10.1016/j.cageo.2025.106080) combines waveform and
  time-frequency inputs and reports results on Taiwan CWA data. Its primary
  task is P picking, so it is a reference for feature design and regional
  evaluation.
- [2026 multistation picking study](https://doi.org/10.1093/gji/ggag326)
  reports that coordinate-frame changes can affect transfer between surface
  and downhole recordings. This motivates testing rotation sensitivity on
  QSIS. It does not establish a benefit for indoor MEMS sensors.
- [Network-specific phase picking and waveform denoising](https://doi.org/10.1007/s00024-026-04025-9)
  adds real background waveforms during training and tests on later events.
  The noise-mixing comparison above adapts this training idea to QSIS clips.
- [2026 PhaseNO evaluation](https://doi.org/10.1093/gji/ggag242) compares
  single-station and multistation picking on four local networks. It
  motivates using coherent motion across QSIS instruments when checking
  weak arrivals. The QSIS classifier still takes one three-component
  recording at inference.
- [Learning from inaccurate arrival labels](https://doi.org/10.1038/s41598-026-57601-3)
  studies training pickers with timing label noise. The QSIS audit above
  found a different label error, in which S arrivals appeared in recordings
  treated as background.
- [New Zealand strong-motion CNN](https://doi.org/10.1038/s41598-026-42568-y)
  uses two-second, three-component segments. Its training data used
  PhaseNet, EQTransformer and available GeoNet arrival information to
  screen candidate phases. The QSIS TATO follow-up above similarly uses
  multiple pickers for proposals and reviews the target sensor separately.
- [SymPhase](https://doi.org/10.3390/sym18071229) uses multiscale filters
  and global and local attention to pick weak arrivals. This motivated
  the small temporal residual experiment on the QSIS classifier.
- [XiaoNet](https://doi.org/10.1093/pnasnexus/pgag322) distills PhaseNet
  into an edge picker. Its reported P/S performance varies substantially
  across datasets. The [RSDB dataset paper](https://doi.org/10.1016/j.eqs.2026.06.001)
  describes weak labels generated with PhaseNet for Raspberry Shake
  records. These are potential training sources. Local analyst labels
  and independent sensor/date coverage are stronger evaluation references.
- [PhaseNeXt, published in November 2026](https://doi.org/10.1186/s40623-026-02578-6),
  scales ConvNeXt and multiscale temporal context to the JMA catalog.
  Briefly reviewed small-event picks helped, while adding fully
  automatic picks was less effective. The QSIS timing pilot used the
  August 2026 PhaseNeXt-M weights pinned to the authors' release.
  Across 46 previously reviewed public arrivals, the raw/band-pass
  agreement filter retained 27 candidates. Of these, 26 fell within
  0.6 s of the visual reference. Rejected phases are part of the coverage
  count. See `runs/qsis-benchmark/phasenext-pilot/timing-diagnostic.json`.
  The earlier RED-PAN run matched 36 arrivals on the same clips,
  and at least one of the two matched 40. The runs used different
  preprocessing and search windows. A separate search of 242 historical
  training station-windows yielded no new arrival that passed both
  co-located array and PhaseNet agreement. See
  `runs/qsis-benchmark/phasenext-pilot/training-candidates.json`.

The [RSDB archive](https://zenodo.org/records/17265621) contains 46,661
three-component MEMS metadata rows, with paired automatic picks on
13,096 of them. A pilot plan selected 104 distinct earthquake IDs
outside Taiwan, with pre-2024 events assigned to adaptation and 2024
events assigned to external validation. The public waveform service
returned six aligned records and rate-limited most of the remaining
requests. No RSDB adaptation was run. The archived selection and
waveform availability results are in `runs/rsdb/sample-download.json`.
