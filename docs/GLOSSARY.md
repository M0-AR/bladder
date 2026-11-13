# Glossary

Plain-language definitions for readers who are new to cystoscopy, machine learning, or both.

## Clinical terms

**Bladder cancer.** A cancer that starts in the lining of the urinary bladder. Most cases are urothelial carcinoma.

**Cystoscopy.** A procedure in which a thin camera is passed through the urethra to inspect the bladder. It is central to diagnosis and surveillance.

**Urothelium.** The lining of the urinary tract. Most bladder cancers arise from it.

**Urothelial carcinoma.** The most common bladder cancer type.

**NMIBC (non-muscle-invasive bladder cancer).** Cancer that has not grown into the muscle wall. It accounts for roughly three quarters of new bladder cancer diagnoses. It is often treatable but recurs frequently, so surveillance matters.

**MIBC (muscle-invasive bladder cancer).** Cancer that has grown into the muscle layer. Treatment is more aggressive.

**Tis (carcinoma in situ).** A flat, high-grade lesion confined to the surface. It can be easy to miss.

**Ta.** A non-invasive papillary tumour. It is a stage within NMIBC, not a separate disease.

**T1.** A tumour that has grown into the connective tissue under the surface but not into muscle. It is a stage within NMIBC.

**Grade.** How abnormal the cells look. Low-grade and high-grade have different risks.

**Recurrence.** The cancer comes back after treatment. Surveillance schedules exist because recurrence is common.

**Histopathology.** Examination of tissue under a microscope. It is the reference standard for cancer diagnosis.

**Sensitivity.** Of the people who truly have the condition, the proportion the test flags. High sensitivity means fewer false negatives.

**Specificity.** Of the people who truly do not have the condition, the proportion the test clears. High specificity means fewer false positives.

**PPV / NPV.** Positive and negative predictive value: the chance that a positive or negative result is correct in the population being tested. They depend on prevalence.

**False negative.** A missed case.

**False positive.** A false alarm.

## Imaging and data terms

**Frame.** One still image from a video.

**Annotation.** A label or outline added to an image or frame.

**Bounding box.** A rectangle around a region of interest, usually stored as x, y, width, height.

**Pascal VOC.** An XML annotation format with an image size and a list of objects and boxes.

**VIA / VGG Image Annotator.** A browser tool for drawing annotations and exporting JSON.

**Timestamp.** The time in the video at which an annotation applies.

**Class imbalance.** Some labels are much more common than others. Accuracy becomes misleading.

**Domain shift.** The test data looks different from the training data: new device, lighting, population, or protocol.

**Data leakage.** Information from test data influences training, producing optimistic results.

**Patient-level split.** All frames from one patient go into exactly one split. This is the credible default.

**Ground truth.** The reference label used for evaluation. For cancer, histopathology is the strongest reference.

## Machine learning terms

**Model.** A function learned from data that maps an input to an output.

**Classifier.** A model that assigns a label to an input, such as a frame.

**Object detector.** A model that finds and localises objects, usually with boxes and confidence scores.

**CNN (convolutional neural network).** A neural network designed for images.

**Transfer learning.** Starting from a model trained on a large dataset and adapting it to a smaller one.

**Fine-tuning.** Continuing training on new data, often with a small learning rate.

**TensorFlow Lite.** A compact runtime for deploying models on edge devices.

**Softmax.** A function that turns raw scores into values that sum to one.

**Confidence score.** A model's self-reported certainty. It is not automatically a calibrated probability.

**Threshold.** The cut-off that turns a score into a positive or negative decision.

**Calibration.** How well predicted probabilities match observed frequencies.

**Augmentation.** Synthetic variation such as flips, crops, and brightness changes used during training.

**Overfitting.** Learning the training set too closely and failing on new data.

**Stratified split.** A split that preserves class proportions.

**Seed.** A number that makes random operations repeatable.

**Confusion matrix.** A table of true versus predicted classes.

**AUC / AUROC.** The probability that a randomly chosen positive ranks above a randomly chosen negative. It is threshold-free and does not by itself prove clinical utility.

**Bootstrap.** Resampling with replacement to estimate uncertainty, often by patient.

**External validation.** Evaluation on data from a different site, device, or period. It is the strongest evidence of generalisation.

## Regulatory and product terms

**SaMD (software as a medical device).** Software intended for a medical purpose without being part of a hardware device.

**FDA 510(k) / De Novo / PMA.** U.S. regulatory pathways with different evidence and risk requirements.

**EU MDR.** Regulation (EU) 2017/745, the European framework for medical devices, including software.

**CE mark.** A conformity mark for products sold in the EU. This project does not have one.

**QMS (quality management system).** The documented processes that govern design, risk, and production.

**ISO 14971.** A risk management standard for medical devices.

**IEC 62366.** A usability engineering standard for medical devices.

**Clinical evaluation.** The evidence that a device is safe and performs as intended in its clinical context.

**Post-market surveillance.** Ongoing monitoring after a device is deployed.

**Intended use / indication for use.** The formal statement of what a device does, for whom, and in what context. It determines the regulatory pathway.

**Automation bias.** The tendency of people to over-trust a machine's output.

**Human-in-the-loop.** A design in which a person reviews or confirms the system's output.

## Project-specific terms

**Bladder Vision.** This repository: a research pipeline and education layer for cystoscopy AI.

**CystoQuest.** The 3D learning game on the project website. It teaches orientation and concepts, not clinical skill.

**Concept network.** The growing graph in the game that links the concepts you have encountered.

**Teaching simulator.** The threshold-based demo on the website. It is a simplified visual model, not the shipped detector.

**Model facts.** A short, structured statement of a model's purpose, data, limitations, and status.

**Split manifest.** A JSON file recording the seed, ratios, and exact files in each split.