rule cell_functional:
    input:
        fixture_sentinel(),
    output:
        tables=directory(f"{RESULTS_ROOT}/figure_2_cell_functional/tables"),
        annotated_h5ad=f"{RESULTS_ROOT}/figure_2_cell_functional/Figure_2_annotated_cells.h5ad",
        figure_source_data=directory(f"{RESULTS_ROOT}/figure_2_cell_functional/Figure_Source_Data"),
        figures=directory(f"{RESULTS_ROOT}/figure_2_cell_functional/figures"),
    script:
        "../scripts/run_cell_functional.py"
