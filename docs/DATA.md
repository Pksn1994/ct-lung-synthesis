# Data provenance and preprocessing

The pilot uses convenience-selected LIDC-IDRI cases 0001–0020, obtained through the NCI Imaging Data Commons browser. This subset is not representative of the whole collection. Patient identifiers here are public dataset identifiers.

Collection: https://www.cancerimagingarchive.net/collection/lidc-idri/

License for the collection images: Creative Commons Attribution 3.0 Unported, https://creativecommons.org/licenses/by/3.0/

Dataset citation: Armato, S. G., III, et al. (2015). Data from LIDC-IDRI. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX

Publication: Armato, S. G., III, et al. (2011). The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. Medical Physics, 38(2), 915–931. https://doi.org/10.1118/1.3528204

The authors acknowledge the National Cancer Institute and the Foundation for the National Institutes of Health, and their critical role in the creation of the free publicly available LIDC/IDRI Database used in this study.

Only CT objects were used. SEG and SR objects in the download were excluded. No nodule annotation or pathology label was used for patch selection. Four cases (0010, 0015, 0017, 0020) have derived/secondary CT metadata, so this pilot mixes original and processed CT. Source series and SOP instance identifiers should be retained with the original download manifest. The pilot archive had 3,592 CT objects and one CT series per patient.

Preprocessing applies DICOM rescale slope and intercept, corrects unsigned encodings of signed padding values, excludes padding, and resamples approximately to 1 mm in-plane spacing using Gaussian smoothing and linear interpolation. Rounded array dimensions mean the grid is not an exact physical 1 mm grid. Intensities are clipped to [-1000, 400] and mapped to [-1, 1]. RescaleType was absent in the inspected series, so unit interpretation relies on the CT context and slope/intercept metadata.

A threshold/morphology heuristic identifies low-density regions within a body outline, keeps up to two largest connected regions, and requires at least 80% mask coverage per patch. This is not validated lung segmentation. It can exclude dense disease or admit non-lung air. There is no nodule guarantee. Eight random 64 × 64 patches are sampled per selected slice with seed [42, patient number]. Exact repeated coordinates within a slice are excluded, but patches can overlap and adjacent slices remain correlated.

Candidate slice selection used a mask area threshold of 8,000 resampled pixels and removed 10% at each end of the candidate list. Training slice selection preceded the final smoothing revision. Validation selection used the revised preprocessing. Replay the original saved manifests to reproduce the recorded selections rather than silently regenerating them with the final code.

The split is fixed in `configs/patient_split_seed42.json`. All patients' middle-slice previews were inspected before splitting. Test patients were excluded from training, but their images were not completely unseen during data quality inspection. No test patches or test evaluation were completed in this run.
