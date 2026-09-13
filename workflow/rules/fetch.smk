rule make_smoke_fixture:
    output:
        f"{DATA_ROOT}/.smoke_fixture",
    script:
        "../scripts/make_smoke_fixture.py"


rule verify_manifest:
    input:
        manifest=config["duke_repository"]["manifest_file"],
    output:
        report=f"{RESULTS_ROOT}/manifest_report.txt",
    script:
        "../scripts/verify_manifest.py"
