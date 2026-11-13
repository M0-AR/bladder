# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-13

### Added

- Installable `bladder_vision` package with modules for annotations, frame extraction, dataset splitting, enhancement, classification, detection, video inference, panorama stitching, metrics, and a CLI.
- Seeded, stratified dataset splitting with a reproducible `split_manifest.json`.
- Pascal VOC export from VIA JSON annotations.
- Dependency-light classification and screening metrics.
- A full interactive website (`preview.html`) with a 3D hero scene, pipeline explorer, threshold simulator, 3D learning game, quiz, searchable glossary, business brief, and research brief.
- The 3D CystoQuest learning game with a growing concept network, narrated explanations, scoring, and a final rank.
- CI workflow for linting, formatting, and tests.
- Project documentation: architecture, dataset contract, evaluation protocol, glossary, business brief, research context, model facts, medical disclaimer, security policy, contribution guide, and code of conduct.
- MIT licence and citation metadata.

### Changed

- Reorganised the original research scripts into `legacy/` and moved the detector model to `models/`.
- Corrected the detector label display from `Suspected_Regain` to `Suspected_Region`.
- Added a canonical class order and a class-name sidecar for trained models.
- Made the detector confidence threshold configurable, with a documented default of 0.30.

### Fixed

- Removed hard-coded Windows paths from the supported pipeline.
- Made frame extraction and annotation conversion use the same deterministic filename scheme.
- Replaced the legacy one-frame panorama replacement logic with OpenCV's stitcher and a documented two-frame fallback.

### Security

- Added a private vulnerability reporting policy and explicit patient-data rules.

## [0.1.0] - 2023-08-07

### Added

- Initial research scripts for VIA annotation conversion, frame extraction, dataset splitting, CNN classification, TFLite detection, contour enhancement, and panorama stitching.
- Sample TFLite detector and label map.