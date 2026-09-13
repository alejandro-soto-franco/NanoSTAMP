rule cell_functional:
    input:
        fixture_sentinel(),
    output:
        tables=directory(f"{RESULTS_ROOT}/figure_2_cell_functional/tables"),
        annotated_h5ad=f"{RESULTS_ROOT}/figure_2_cell_functional/Figure_2_annotated_cells.h5ad",
    script:
        "../scripts/run_cell_functional.py"
