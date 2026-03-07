JULIA = "julia --project=."
SRC = ["src/utils.jl", "src/common_core.jl", "src/common.jl", "src/plotting.jl"]

MONOD = "examples/monod"
RESULTS = f"{MONOD}/results"
MODEL = f"{MONOD}/model.jl"
DEPS = [MODEL] + SRC

HALDANE = "examples/haldane"
HALDANE_RESULTS = f"{HALDANE}/results"
HALDANE_MODEL = f"{HALDANE}/model.jl"
HALDANE_DEPS = [HALDANE_MODEL] + SRC

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
    output: f"{RESULTS}/checkpoint.jls"
    shell: "./vsc/train_remote.sh"

# ---- BIM design optimization (CPU, local, sequential via resource lock) ----
rule optimize_bim_std:
    input: f"{MONOD}/optimize_bim.jl", *DEPS
    output: f"{RESULTS}/bim_std_design.jls"
    resources: bim_slot=1
    shell: "{JULIA} {input[0]}"

rule optimize_bim_cheat:
    input: f"{MONOD}/optimize_bim.jl", *DEPS
    output: f"{RESULTS}/bim_cheat_design.jls"
    resources: bim_slot=1
    shell: "{JULIA} {input[0]} cheating"

# ---- Static sPCE optimization (GPU, VSC cluster) ----
rule optimize_static:
    input: f"{MONOD}/optimize_static.jl", *DEPS
    output: f"{RESULTS}/spce_static_design.jls"
    shell: "TASK=optimize_static WALLTIME=4:00:00 ./vsc/train_remote.sh"

# ---- sPCE evaluation (GPU, local) ----
rule eval_spce:
    input:
        script=f"{MONOD}/eval_spce.jl",
        checkpoint=f"{RESULTS}/checkpoint.jls",
        bim_std=f"{RESULTS}/bim_std_design.jls",
        bim_cheat=f"{RESULTS}/bim_cheat_design.jls",
        spce_static=f"{RESULTS}/spce_static_design.jls",
        deps=DEPS,
    output:
        scores=f"{RESULTS}/spce_scores.jls",
        hist=f"{RESULTS}/plot_spce_histograms.png",
        traj=f"{RESULTS}/plot_spce_trajectories.png",
    shell: "{JULIA} {input.script}"

# ---- Posterior evaluation (GPU, local) ----
rule eval_posterior:
    input:
        script=f"{MONOD}/eval_posterior.jl",
        checkpoint=f"{RESULTS}/checkpoint.jls",
        bim_std=f"{RESULTS}/bim_std_design.jls",
        bim_cheat=f"{RESULTS}/bim_cheat_design.jls",
        spce_static=f"{RESULTS}/spce_static_design.jls",
        deps=DEPS,
    output:
        results=f"{RESULTS}/posterior_results.jls",
        plot=f"{RESULTS}/plot_posterior.png",
    shell: "{JULIA} {input.script}"

# ---- Training loss plot (CPU, local) ----
rule plot_training:
    input:
        script=f"{MONOD}/plot_training.jl",
        checkpoint=f"{RESULTS}/checkpoint.jls",
        deps=DEPS,
    output: f"{RESULTS}/plot_training_loss.png"
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

# ---- Weibull PK training (VSC cluster) ----
rule weibull_train:
    input: f"{WEIBULL}/train.jl", *WEIBULL_DEPS
    output: f"{WEIBULL_RESULTS}/checkpoint.jls"
    shell: "EXAMPLE=weibull WALLTIME=8:00:00 ./vsc/train_remote.sh"

# ---- Figures: copy to paper/figures/ ----
rule figures:
    input:
        f"{RESULTS}/plot_training_loss.png",
        f"{RESULTS}/plot_spce_trajectories.png",
        f"{RESULTS}/plot_spce_histograms.png",
        f"{RESULTS}/plot_posterior.png",
        f"{HALDANE_RESULTS}/plot_comparison.png",
    output:
        "paper/figures/monod_training_loss.png",
        "paper/figures/monod_spce_trajectories.png",
        "paper/figures/monod_spce_histograms.png",
        "paper/figures/monod_posterior.png",
        "paper/figures/haldane_comparison.png",
    run:
        import shutil, os
        os.makedirs("paper/figures", exist_ok=True)
        for src, dst in zip(input, output):
            shutil.copy2(src, dst)

# ---- Paper compilation (CDC: compact, arXiv: extended) ----
rule paper_cdc:
    input: rules.figures.output
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
    input: rules.figures.output
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
