JULIA = "julia --project=."
SRC = ["src/utils.jl", "src/common_core.jl", "src/common.jl"]

MONOD = "examples/monod"
RESULTS = f"{MONOD}/results"
MODEL = f"{MONOD}/model.jl"
DEPS = [MODEL, f"{MONOD}/plotting.jl"] + SRC

HALDANE = "examples/haldane"
HALDANE_RESULTS = f"{HALDANE}/results"
HALDANE_MODEL = f"{HALDANE}/model.jl"
HALDANE_DEPS = [HALDANE_MODEL] + SRC

DCMOTOR = "examples/dcmotor"
DCMOTOR_RESULTS = f"{DCMOTOR}/results"
DCMOTOR_MODEL = f"{DCMOTOR}/model.jl"
DCMOTOR_DEPS = [DCMOTOR_MODEL] + SRC

WEIBULL = "examples/weibull"
WEIBULL_RESULTS = f"{WEIBULL}/results"
WEIBULL_MODEL = f"{WEIBULL}/model.jl"
WEIBULL_DEPS = [WEIBULL_MODEL] + SRC

# ---- Default target ----
rule all:
    input:
        "paper/main_cdc.pdf",
        "paper/main_arxiv.pdf",

# ---- Training (VSC cluster) ----
rule train:
    input: f"{MONOD}/train.jl", *DEPS
    output: f"{RESULTS}/checkpoint.jls", f"{RESULTS}/plot_training_loss.png"
    shell: "WALLTIME=6:00:00 ./vsc/train_remote.sh"

# ---- BIM design optimization (CPU, local, sequential via resource lock) ----
rule optimize_bim:
    input: f"{MONOD}/optimize_bim.jl", *DEPS
    output: f"{RESULTS}/design_bim.jls"
    resources: bim_slot=1
    shell: "{JULIA} {input[0]}"

# ---- Static sPCE optimization (GPU, VSC cluster) ----
rule optimize_static:
    input: f"{MONOD}/optimize_static.jl", *DEPS
    output: f"{RESULTS}/design_spce.jls"
    shell: "TASK=optimize_static WALLTIME=6:00:00 ./vsc/train_remote.sh"

# ---- sPCE evaluation (GPU, local) ----
rule eval_spce:
    input:
        script=f"{MONOD}/eval_spce.jl",
        checkpoint=f"{RESULTS}/checkpoint.jls",
        bim=f"{RESULTS}/design_bim.jls",
        spce=f"{RESULTS}/design_spce.jls",
        deps=DEPS,
    output:
        scores=f"{RESULTS}/spce_scores.jls",
    shell: "{JULIA} {input.script}"

# ---- Posterior evaluation (GPU, local) ----
rule eval_posterior:
    input:
        script=f"{MONOD}/eval_posterior.jl",
        checkpoint=f"{RESULTS}/checkpoint.jls",
        bim=f"{RESULTS}/design_bim.jls",
        spce=f"{RESULTS}/design_spce.jls",
        deps=DEPS,
    output:
        results=f"{RESULTS}/posterior_results.jls",
    shell: "{JULIA} {input.script}"

# ---- Posterior plots (CPU, local) ----
rule plot_posterior:
    input:
        script=f"{MONOD}/plot_posterior.jl",
        results=f"{RESULTS}/posterior_results.jls",
        checkpoint=f"{RESULTS}/checkpoint.jls",
        deps=DEPS,
    output: f"{RESULTS}/plot_posterior.png"
    shell: "{JULIA} {input.script}"

# ---- Haldane training (VSC cluster) ----
rule haldane_train:
    input: f"{HALDANE}/train.jl", *HALDANE_DEPS
    output: f"{HALDANE_RESULTS}/checkpoint.jls"
    shell: "EXAMPLE=haldane WALLTIME=4:00:00 ./vsc/train_remote.sh"

# ---- Haldane comparison plot (CPU, local) ----
rule haldane_plot:
    input:
        script=f"{HALDANE}/plot_comparison.jl",
        checkpoint=f"{HALDANE_RESULTS}/checkpoint.jls",
        deps=HALDANE_DEPS,
    output: f"{HALDANE_RESULTS}/plot_comparison.png"
    shell: "{JULIA} {input.script}"

# ---- Monod dynamics plot (CPU, local) ----
rule monod_dynamics_plot:
    input:
        script=f"{MONOD}/plot_dynamics.jl",
        checkpoint=f"{RESULTS}/checkpoint.jls",
        bim=f"{RESULTS}/design_bim.jls",
        spce=f"{RESULTS}/design_spce.jls",
        deps=DEPS,
    output: f"{RESULTS}/plot_dynamics.png"
    shell: "{JULIA} {input.script}"

# ---- DC Motor training (GPU, local or VSC) ----
rule dcmotor_train:
    input: f"{DCMOTOR}/train.jl", *DCMOTOR_DEPS
    output: f"{DCMOTOR_RESULTS}/checkpoint.jls", f"{DCMOTOR_RESULTS}/plot_training_loss.png"
    shell: "EXAMPLE=dcmotor TASK=train ./vsc/train_remote.sh"

# ---- DC Motor: adaptive sPCE vs adaptive BIM comparison (CPU, local) ----
rule dcmotor_eval_comparison:
    input:
        script=f"{DCMOTOR}/eval_comparison.jl",
        checkpoint=f"{DCMOTOR_RESULTS}/checkpoint.jls",
        bim=f"{DCMOTOR}/adaptive_bim.jl",
        deps=DCMOTOR_DEPS,
    output: f"{DCMOTOR_RESULTS}/comparison_results.jls"
    shell: "{JULIA} {input.script}"

# ---- DC Motor timing plot (CPU, local) ----
rule dcmotor_plot_timing:
    input:
        script=f"{DCMOTOR}/plot_trajectories.jl",
        comparison=f"{DCMOTOR_RESULTS}/comparison_results.jls",
        deps=DCMOTOR_DEPS,
    output: f"{DCMOTOR_RESULTS}/plot_policy_timing.png"
    shell: "{JULIA} {input.script}"

# ---- Weibull PK training (VSC cluster) ----
rule weibull_train:
    input: f"{WEIBULL}/train.jl", *WEIBULL_DEPS
    output: f"{WEIBULL_RESULTS}/checkpoint.jls"
    shell: "EXAMPLE=weibull WALLTIME=10:00:00 ./vsc/train_remote.sh"

# ---- Weibull nuisance adaptation plot (CPU, local) ----
rule weibull_nuisance_plot:
    input:
        script=f"{WEIBULL}/plot_nuisance.jl",
        checkpoint=f"{WEIBULL_RESULTS}/checkpoint.jls",
        deps=WEIBULL_DEPS,
    output: f"{WEIBULL_RESULTS}/plot_nuisance.png"
    shell: "{JULIA} {input.script}"

# ---- Generate tables from results ----
rule generate_tables:
    input:
        script="scripts/generate_tables.jl",
        spce=f"{RESULTS}/spce_scores.jls",
        posterior=f"{RESULTS}/posterior_results.jls",
    output: "paper/tables/spce_table.tex"
    shell: "{JULIA} {input.script}"

rule generate_dcmotor_table:
    input:
        script="scripts/generate_tables.jl",
        comparison=f"{DCMOTOR_RESULTS}/comparison_results.jls",
    output: "paper/tables/dcmotor_table.tex"
    shell: "{JULIA} {input.script}"

# ---- Figures: copy to paper/figures/ ----
rule figures:
    input:
        f"{RESULTS}/plot_training_loss.png",
        f"{RESULTS}/plot_posterior.png",
        f"{RESULTS}/plot_dynamics.png",
        f"{HALDANE_RESULTS}/plot_comparison.png",
        f"{WEIBULL_RESULTS}/plot_nuisance.png",
        f"{DCMOTOR_RESULTS}/plot_policy_timing.png",
    output:
        "paper/figures/monod_training_loss.png",
        "paper/figures/monod_posterior.png",
        "paper/figures/monod_comparison.png",
        "paper/figures/haldane_comparison.png",
        "paper/figures/pharma_nuisance.png",
        "paper/figures/dcmotor_timing.png",
    run:
        import shutil, os
        os.makedirs("paper/figures", exist_ok=True)
        for src, dst in zip(input, output):
            shutil.copy2(src, dst)

# ---- Paper compilation (CDC: compact, arXiv: extended) ----
rule paper_cdc:
    input: rules.figures.output, rules.generate_tables.output, rules.generate_dcmotor_table.output
    output: "paper/main_cdc.pdf"
    shell:
        r"""
        cd paper \
        && pdflatex -interaction=nonstopmode -jobname=main_cdc "\def\buildmode{{cdc}}\input{{main}}" \
        && bibtex main_cdc \
        && pdflatex -interaction=nonstopmode -jobname=main_cdc "\def\buildmode{{cdc}}\input{{main}}" \
        && pdflatex -interaction=nonstopmode -jobname=main_cdc "\def\buildmode{{cdc}}\input{{main}}"
        """

rule paper_arxiv:
    input: rules.figures.output, rules.generate_tables.output, rules.generate_dcmotor_table.output
    output: "paper/main_arxiv.pdf"
    shell:
        r"""
        cd paper \
        && pdflatex -interaction=nonstopmode -jobname=main_arxiv "\def\buildmode{{arxiv}}\input{{main}}" \
        && bibtex main_arxiv \
        && pdflatex -interaction=nonstopmode -jobname=main_arxiv "\def\buildmode{{arxiv}}\input{{main}}" \
        && pdflatex -interaction=nonstopmode -jobname=main_arxiv "\def\buildmode{{arxiv}}\input{{main}}"
        """

rule paper:
    input: "paper/main_cdc.pdf", "paper/main_arxiv.pdf"
