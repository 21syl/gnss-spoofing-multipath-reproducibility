# Experimental datasets

The experiments use self-collected BDS B1I recordings and the public TEXBAT GPS
L1 C/A dataset. Self-collected recordings can be obtained by contacting the
corresponding author at the email address listed in the paper. Because of their
file size, the recordings are provided separately from this repository.

## Self-collected recordings

The signal settings are BDS B1I, authentic C/N0 of 45 dB·Hz, spoofing power
advantage of 3 dB and asynchronous code pull-off at 0.1 chips/s from -4 to 4 chips.
Dynamic recordings use a platform speed of 15 m/s. Each experiment processes one
tracked satellite. Multipath generation follows the models cited in the paper
([reference 1](https://doi.org/10.1109/TVT.2020.3038646),
[reference 2](https://doi.org/10.1186/s43020-020-00027-7)).

| Record | Platform | Signal sequence |
|---|---|---|
| bs1 | Static | Authentic → authentic + spoofing |
| bs2 | Dynamic | Authentic → authentic + spoofing |
| bs3 | Static | Authentic → authentic + multipath |
| bs4 | Dynamic | Authentic → authentic + multipath |
| bs5 | Static | Authentic → authentic + multipath → authentic + multipath + spoofing |
| bs6 | Dynamic | Authentic → authentic + multipath → authentic + multipath + spoofing |
| bs7 | Static | Authentic → authentic + spoofing → authentic + spoofing + multipath |
| bs8 | Dynamic | Authentic → authentic + spoofing → authentic + spoofing + multipath |

The original pairwise signals last 200 s, with interference injected at 100 s.
Receiver-output selections are aligned using the original time of the first
selected epoch. The mixed records contain 198 s of processed epochs on channel
8. Their first and second interference types enter at 80 and 120 s, measured from
the first tracking epoch; the first interference remains present afterward.

## TEXBAT

Obtain TEXBAT from the [UT Austin Radionavigation Laboratory](https://radionavlab.ae.utexas.edu/texbat/)
or its [official download directory](https://rnl-data.ae.utexas.edu/datastore/texbat/).
The cross-scenario experiment uses ds2/ds4/ds5 for training and ds3/ds6/ds7/ds8
for testing. Receiver processing produces the correlator MAT inputs used by the
export script.

## Input construction and partitions

Neural-network inputs consist of I-branch amplitudes at 11 correlators over
[-2,2] chips, with positions -2, -1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6
and 2 chips. Each input contains ten 1 ms epochs, with a 10 ms window and step.
The tensor layout is [sample, correlator, time]. Labels are 0 for authentic only,
1 for spoofing present (including simultaneous multipath), and 2 for multipath
without spoofing. Event timestamps determine the labels; windows stay within
their signal-state intervals.

Pairwise and TEXBAT within-scenario experiments partition each continuous
signal-state interval chronologically at 60/20/20 before constructing windows.
Class balance is obtained by sampling within each partition. Standardization is
fitted on training samples and applied to validation and test samples.

Mixed-record transfer trains on bs5 and tests bs6, or trains on bs8 and tests bs7.
Source windows are split by class-stratified sampling at 70/15/15. One min/max
pair is fitted on the source training samples and applied to the remaining
samples. TEXBAT transfer uses an 80/20 source training/validation split and
training-fitted standardization. Independent target records are used for testing.

Detailed signal parameters and record identifiers are in
`configs/dataset_parameters.json`. The export configurations specify local file
paths, channels, available durations and time offsets. Set these values for the
receiver outputs being processed. MATLAB reads `trackData.<system>.channel` and
exports the selected raw correlator epochs to HDF5; Python constructs the windows
and fits the configured normalization.
