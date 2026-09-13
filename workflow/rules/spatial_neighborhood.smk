rule spatial_neighborhood:
    input:
        annotated_h5ad=rules.cell_functional.output.annotated_h5ad,
        tables=rules.cell_functional.output.tables,
    output:
        tables=directory(f"{RESULTS_ROOT}/figure_2_spatial_neighborhood/tables"),
    script:
        "../scripts/run_spatial_neighborhood.py"
