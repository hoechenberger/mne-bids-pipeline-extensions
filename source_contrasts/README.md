# Produce power contrasts at the source level for custom frequency bands.

This script executes the following processing steps:

- Apply the inverse operator to the covariances of each condition. This yields
  a source estimate of the power.
- Calculate the difference of the log10 powers of two conditions (i.e., form a
  contrast). This yields a ratio of the log10 power: log10(cond_1 / cond_2).
- Average those contrasts across subjects, yielding a grand average power
  contrast at the source level.

The output are STCs and PNG screenshots.
