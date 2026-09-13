rule spleen_lnp:
    input:
        fixture_sentinel(),
    output:
        tables=directory(f"{RESULTS_ROOT}/figure_1d_1e/tables"),
    script:
        "../scripts/run_spleen_lnp.py"
