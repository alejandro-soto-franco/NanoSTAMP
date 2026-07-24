# NanoSTAMP analysis notebooks

This GitHub repository contains the cleaned analysis notebooks associated with
the NanoSTAMP study. It supports analyses presented in Supplementary Figure 1c,
Figure 1d-g, Figure 2, and Supplementary Figures 6-11.

## Code and data availability

This GitHub repository contains the notebooks and requirement files only. It
does not contain research data.

The processed `.csv`, `.csv.gz`, and `.h5ad` inputs are deposited separately in
the Duke Research Data Repository:

> https://research.repository.duke.edu/record/554?ln=en

Raw microscope images and stitched or registered TIFF stacks are not deposited
on GitHub or in the Duke processed-data package.

The Duke record contains one top-level folder named `For Repository`, which
contains the processed data only. It does not contain the notebooks or other
code.

The processed-data deposit supports the downstream cell-level, functional, and
spatial-neighborhood analyses. Image-processing and spot-detection notebooks on
GitHub document the algorithms, parameters, barcode definitions, and
quality-control procedures used to generate the processed tables, but their
image-level steps cannot be rerun without the original images.

## Local directory structure

After cloning the GitHub repository, download the `For Repository` folder from
the Duke Research Data Repository. Rename `For Repository` to `Data` and place
it next to the GitHub `Code/` folder:

```text
NanoSTAMP/
├── Code/
│   ├── *.ipynb
│   └── requirements_*.txt
└── Data/
    ├── Supplementary_Figure_1c_2Oligo_RCA/
    │   └── Processed_Data/
    ├── Figure_1d_1e_Spleen_LNP/
    │   └── Precomputed_Analysis_Input/
    ├── Figure_1f_1g_Spleen_Neighborhoods/
    │   ├── Precomputed_Analysis_Input/
    │   └── Published_Output_Reference/
    └── Figure_2_and_Supplementary_Figures_6_11_Multiplex_LNP/
        └── Precomputed_Downstream_Input/
```

Keep `Code/` and the renamed `Data/` folder at the same level. The notebooks use
this relative layout to locate the downloaded inputs. `Data/` remains a local
download and is not part of the GitHub repository.

## Notebooks

| Notebook | Purpose | Runnable with deposited data |
|---|---|---|
| `Supplementary_Figure_1c_Image_Processing_and_Quantification.ipynb` | Documents two-oligo RCA-FISH image processing and quantification | Processed outputs can be inspected; image-level steps require the excluded TIFFs |
| `Figure_1d_Bit_1_Spot_Detection.ipynb` | Documents single-channel Bit_1 punctum detection | No; requires the excluded registered TIFFs |
| `Figure_1d_1e_Full_Barcode_Spot_Detection.ipynb` | Documents full 12-bit barcode detection and decoding | No; requires the excluded registered TIFFs |
| `Figure_1d_1e_Spleen_LNP_Analysis.ipynb` | Figure 1d/e spatial-tile and cell-type analyses | Yes |
| `Figure_1f_1g_Spleen_Neighborhood_Analysis.ipynb` | Figure 1f/g spleen neighborhood analyses | Yes |
| `Figure_2_and_Supplementary_6_11_Multiplex_LNP_Spot_Detection_Round_1.ipynb` | Documents multiplex spot detection for S1/S2 | No; requires the excluded registered TIFFs |
| `Figure_2_and_Supplementary_6_11_Multiplex_LNP_Spot_Detection_Round_2.ipynb` | Documents multiplex spot detection for S3/S4 | No; requires the excluded registered TIFFs |
| `Figure_2_and_Supplementary_6_8_Multiplex_LNP_Cell_and_Functional_Analysis.ipynb` | Cell-level uptake, composition, OVA-expression, and antigen-presentation analyses | Yes |
| `Figure_2_and_Supplementary_9_11_Multiplex_LNP_Spatial_Neighborhood_Analysis.ipynb` | Recurrent-neighborhood and local immune-response analyses | Yes, after the preceding Figure 2 notebook |

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/HickeyLab/NanoSTAMP.git
cd NanoSTAMP
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Code/requirements_publication_all.txt
python -m jupyter lab
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Workflow-specific requirement files are also provided in `Code/`.

## Recommended execution order

### Figure 1d and Figure 1e

Run `Figure_1d_1e_Spleen_LNP_Analysis.ipynb`. It reads the deposited compact
Bit_1 calls, full-barcode calls, and frozen cell annotations. The two
spot-detection notebooks are methodological records and do not need to be run.

### Figure 1f and Figure 1g

Run `Figure_1f_1g_Spleen_Neighborhood_Analysis.ipynb`.

### Figure 2 and Supplementary Figures 6-11

Run these notebooks in order:

1. `Figure_2_and_Supplementary_6_8_Multiplex_LNP_Cell_and_Functional_Analysis.ipynb`
2. `Figure_2_and_Supplementary_9_11_Multiplex_LNP_Spatial_Neighborhood_Analysis.ipynb`

The first notebook creates the merged annotated dataset consumed by the second.
The two multiplex spot-detection notebooks are methodological records and do
not need to be run.

## Upstream frozen analyses

Cell segmentation, clustering, and cell-type annotation were performed using
the previously published spatial-omics workflow and are treated as frozen
upstream inputs. The deposited files contain the cell identifiers, centroids,
measurements, annotations, and processed spot calls needed for the downstream
analyses.

## Outputs

The downstream notebooks display manuscript plots inline and write analysis
tables or merged datasets beneath the corresponding `Data/*/Generated_Output/`
directory when applicable.

## Reproducibility scope

The GitHub code and companion Duke dataset together support reproduction
beginning from the deposited processed CSV and H5AD files. They do not support
image-level reproduction beginning from raw microscopy data because those
images are outside the scope of both deposits.

All publication notebooks are distributed without stored outputs or execution
counts.

## Citation

If you use this code or the accompanying processed data, please cite:

> [AUTHORS]. [MANUSCRIPT TITLE]. [JOURNAL OR PREPRINT SERVER] ([YEAR]). [DOI]

Please also cite the previously published spatial-omics workflow where
appropriate.

## Contact

[NAME]  
[INSTITUTION]  
[EMAIL]

## License

[LICENSE NAME]
