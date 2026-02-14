# Tangram Reimplementation in PyTorch Lightning

**Tangram** is a probabilistic mapping framework that aligns single-cell RNA-seq profiles and spatial transcriptomics measurements through a learned mapping matrix that assigns cells to spatial locations by optimizing a similarity-based objective between predicted and observed spatial gene expression.

This repository contains a reimplementation of Tangram in PyTorch Lightning that:

* Builds upon the original Tangram implementation by the Broad Institute: [https://github.com/broadinstitute/Tangram](https://github.com/broadinstitute/Tangram); and the refinement extension: [https://github.com/daisybio/Tangram_Refinement_Strategies](https://github.com/daisybio/Tangram_Refinement_Strategies)

* Provides a modular trainer and utilities for inspecting training dynamics
* Adds tools for downstream mapping analysis.

---

## Supported Modes

Currently supported:

* `cell` mode as default
* `constrained` mode, controlled by the `filter` flag in the default mode

Not supported:

* `cluster` mode

---

## Intended Use Case

This implementation is designed for imaging-based spatial transcriptomics technologies that provide:

* Single-molecule resolution
* Single-cell spatial expression profiles

Accordingly:

* Only a uniform density prior over spots is implemented
* `rna_count_based` priors are not supported
* Custom density priors are not yet implemented