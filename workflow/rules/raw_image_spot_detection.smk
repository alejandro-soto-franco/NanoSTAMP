rule raw_image_spot_detection:
    input:
        fixture_sentinel(),
    output:
        tables=directory(f"{RESULTS_ROOT}/raw_image_spot_detection/{{family}}/tables"),
    wildcard_constraints:
        family="full_barcode|bit_1|round_1|round_2",
    script:
        "../scripts/run_raw_image_spot_detection.py"


rule supplementary_figure_1c:
    input:
        fixture_sentinel(),
    output:
        tables=directory(f"{RESULTS_ROOT}/supplementary_figure_1c/tables"),
    script:
        "../scripts/run_supplementary_1c.py"
