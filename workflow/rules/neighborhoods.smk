rule spleen_neighborhoods:
    input:
        fixture_sentinel(),
    output:
        tables=directory(f"{RESULTS_ROOT}/figure_1f_1g/tables"),
    script:
        "../scripts/run_neighborhoods.py"
