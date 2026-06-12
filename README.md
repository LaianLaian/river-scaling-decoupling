# Climate and human organization decouple geometric and functional scaling in river networks



This repository contains the analysis scripts and source data required to reproduce all figures and results presented in the manuscript "Climate and human organization decouple geometric and functional scaling in river networks".



Using 41,677 rivers across China, this study investigates how climatic forcing, hydrological connectivity, and human spatial organization shape river-network scaling laws. The analyses reveal a fundamental distinction between:



\- Geometric scaling, represented by Hack's law linking river length and drainage area, which remains highly conserved across climatic gradients and river hierarchies.

\- Functional scaling, represented by runoff-efficiency scaling, which varies systematically with precipitation, hydrological connectivity, and administrative fragmentation.



The repository reproduces the four main components of the study:



1\. \*\*Climate-constrained geometric scaling\*\*

&#x20;  - Evaluation of Hack's law across climatic gradients and river hierarchies.



2\. \*\*Climate-dependent functional scaling\*\*

&#x20;  - Quantification of precipitation controls on runoff-efficiency scaling exponents.



3\. \*\*Hydrological connectivity regulates functional scaling\*\*

&#x20;  - Assessment of the influence of river-network structure and lake systems on runoff efficiency and scaling behavior.



4\. \*\*Human fragmentation reshapes geometric and functional scaling\*\*

&#x20;  - Examination of how administrative partitioning alters natural scaling relationships.




## Contents

### Analysis scripts

* `Step1_hacks_scaling_analysis.py`  
Hack's law analysis, river-grade stratification, and nested-model evaluation.
* `Step2_climate_hack_modulation.py`  
Per-river Hack exponent (`h`) versus long-term mean precipitation (`P`).
* `Step3_rp_scaling_climate.py`  
Sliding-window runoff-efficiency scaling (`beta`) along the precipitation gradient.
* `Step4_structure_efficiency_coupling.py`  
River-network structure proxies versus runoff efficiency, controlling for area and precipitation.
* `Step5_lake_modulation.py`  
Lake abundance/density effects on runoff efficiency and local scaling behavior.
* `Step6_admin_fragmentation.py`  
Administrative fragmentation as a perturbation to Hack's law and runoff-efficiency scaling.

### Figure source-data workbooks

* `SourceData_Fig1.xlsx`
* `SourceData_Fig2.xlsx`
* `SourceData_Fig3.xlsx`
* `SourceData_Fig4.xlsx`

Each workbook is organized by sheet, and each sheet corresponds to a panel in the manuscript figure set (for example, `Fig1a`, `Fig1b`, etc.). 
For panels generated directly from the restricted river-network database, summary statistics (e.g., scaling exponents, confidence intervals, model fit metrics, and sample sizes) are provided instead of the underlying river-level observations. For all other panels, the sheets contain the tabular data used to reproduce the corresponding figures.
Schematic-only panels are not associated with a spreadsheet sheet if no numeric data are required.

## How to reproduce the analysis

1. Place the merged river dataset and any auxiliary inputs in the paths expected by the scripts.
2. Edit the input/output paths near the top of each script if needed.
3. Run the scripts in sequence, starting with `Step1_hacks_scaling_analysis.py`.
4. The later scripts reuse the same cleaned river dataset and write their outputs to module-specific result folders.

## Data and variables

Common variables used across the scripts include:

* `L_km`: river length in km
* `A_km2`: drainage area in km²
* `P_mm`: long-term mean precipitation in mm
* `R_mm`: runoff in mm
* `RP = R/P`: runoff efficiency
* `h`: Hack exponent from `log10(L) ~ log10(A)`
* `beta`: runoff-efficiency scaling exponent from `log10(R/P) ~ log10(A)`

## Output structure

Typical outputs include:

* cleaned data tables,
* regression summaries,
* correlation tables,
* publication-quality PNG and PDF figures,
* markdown result summaries.

## Notes

* The scripts use the same cleaning logic for river data across modules.
* Several analyses are observational and should be interpreted as statistical associations rather than causal estimates.
* The administrative-fragmentation module uses an equal-split approximation for rivers crossing multiple county-level units.
* The lake module merges provincial lake inventory data to river records using inferred province identifiers.

## Data availability


Figure-level source data used to generate the figures are provided in this repository as `SourceData_Fig1.xlsx`–`SourceData_Fig4.xlsx`.

For panels derived directly from the restricted river-network database, summary statistics are provided rather than the underlying river-level observations.

The compiled river-network and lake datasets supporting the findings of this study are not publicly distributed. These datasets may be made available by W.W. Shao pending scientific review and completion of a material transfer agreement.

Requests for access should be directed to:

\[shaoww@iwhr.com]



## Suggested citation

If this repository is used in a manuscript, please cite the main paper and note that the analysis code and figure source data are provided here for reproducibility.

