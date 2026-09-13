rule spatial_neighborhood_distance:
    input:
        spatial_tables=rules.spatial_neighborhood.output.tables,
        cell_functional_tables=rules.cell_functional.output.tables,
    output:
        tables=directory(f"{RESULTS_ROOT}/figure_2_spatial_neighborhood_distance/tables"),
    script:
        "../scripts/run_spatial_neighborhood_distance.py"
