# MEGA Architecture and Theoretical Framework

This folder contains the formal specification of MEGA and the theoretical framework it belongs to. These documents were developed independently over approximately three years as the architecture was built and refined. They are provided here as reference material supporting the ARI benchmark submission.

If you have not yet read [What Kind of Thing is MEGA](https://www.fsadb.org/what-kind-of-thing-is-mega/), read that first. It is a conceptual walkthrough written for readers who want intuition about how MEGA works before engaging with formal material. The documents in this folder assume that intuition.

---

## The Documents

### `MEGA_PFOL.pdf` — Probabilistic First-Order Logic Specification

The formal specification of MEGA as implemented. Defines the signature (sorts, functions, predicates), the structural and probabilistic axioms, the metagene operations (capture, open), the population structure, and the system's evolution. Every operator in the code has a corresponding formal statement in this document, and the document's axioms are the reference against which the implementation is verified.

This is the most self-contained of the four documents and is appropriate for readers who want to confirm that the implementation matches a formal specification before engaging with claims about what the system is.

### `GADS_Theory.pdf` — Subsystem Derivation and Phase Dynamics

Derives MEGA's subsystem structure from a small set of axioms about locality, finiteness, and signal availability. Eight derivation chains each begin with an axiom, derive a constraint, and conclude with the subsystem that the constraint forces into existence. The closure argument at the end of Section 1 makes the case that the six subsystems together are necessary under the stated axioms, not a design choice.

Section 8 analyzes the system's dynamics incrementally, starting from boundary geometry alone (Layer 0) and adding subsystems one layer at a time. Each layer produces a specific failure mode that motivates the next layer's mechanism. Layer 0 demonstrates structural capacity limits of flat partitioning; Layer 1 demonstrates the Grammar-Density Singularity when compression is unconstrained; Layer 2 demonstrates that uniform compositional injection produces path-dependent failure modes no static probability can prevent.

This document is the architectural argument.

### `MEGA_Math_Search_Space_Model.pdf` — Hyperbolic Geometry of the Search Space

Establishes that metagenes induce Lorentz boosts on a hyperbolic embedding of the organism space. Develops the formal apparatus for describing how compositional identity warps distance between configurations: organisms sharing metagenes cluster more tightly in the transformed space, reflecting reduced effective edit distance. This is the geometric grounding for Section A10 of GADS_Theory and for the structural prior used in the VFE telemetry.

Read this document if you want to understand how the intuition "metagenes compress the search space" becomes a formal geometric statement.

### `MEGA_VFE.pdf` — Variational Free Energy Telemetry

Derives a passive telemetry signal from the architecture. The variational free energy decomposes into three terms: selection complexity (KL divergence from MEGA's intrinsic selection distribution), structural complexity (negative log-likelihood under a Boltzmann prior grounded in the hyperbolic geometry from the previous document), and accuracy (surprisal under a bigram generative model). The three-term structure mirrors the standard evidence lower bound from variational inference.

The VFE is explicitly passive — it observes the system's dynamics without influencing selection. It is not a fitness function. For readers familiar with the Free Energy Principle, this document is likely the most immediate entry point into the formal framework, because it connects MEGA's dynamics to a framework many researchers in this space already trust.

---

## `theory_sim/` — Prediction, Discovery, and Phase Analysis

The `theory_sim/` subfolder contains an incremental simulation program that plays two distinct roles in the development of the theoretical framework.

The first role is testing predictions made in advance. The conceptual walkthrough in "What Kind of Thing is MEGA" made predictions about what should happen when specific mechanisms were present or absent — the Grammar-Density Singularity under unchecked capture, the runaway compression when metagenes are treated identically to base genes, the necessity of the Open operator to counteract inward pressure. The subsequent theoretical work (PFOL, then GADS_Theory) extended these predictions by mapping the algorithm's flow formally. The simulators were built to test both sets of predictions by activating one subsystem at a time and observing whether the predicted failure modes manifested.

The second role is discovery. Running the layers individually surfaced dynamics that had not been predicted — behaviors of the boundary geometry, specific signatures of path-dependent failure, the character of the oscillatory equilibrium that appears when the full feedback loop closes. These findings are documented in Section 8 of GADS_Theory, which synthesizes both the predicted and the discovered dynamics into the formal phase-regime analysis.

Layer 0 is boundary geometry alone. Each subsequent layer adds exactly one mechanism: capture (Layer 1), compositional injection (Layer 2), the open operator (Layer 3), MCO temporal decay (Layer 4), the full metabolic pipeline (Layer 5), and fitness-driven differential survival (Layer 6). Isolating each mechanism's contribution is what made the phase analysis possible — in the full running system all mechanisms are active simultaneously and their effects are entangled.

The simulators produce two kinds of output at each layer: standard analytical plots (token fractions, DAG dynamics, contact statistics, metabolic activity) and per-generation delta-waveform plots of the primary dynamical signals. The deltas are also exported as CSVs and can be rendered as audio. Listening to the deltas makes the qualitative character of each layer's dynamics immediately legible in a way that static plots do not: a system that has collapsed into a singularity sounds silent; a system in sustained non-equilibrium sounds like continuous small-scale activity without dominant periodic structure.

See `theory_sim/README.md` for details on running the simulators and interpreting their outputs.

---

## How the Documents Relate

The documents were developed in roughly this order:

1. The search-space geometry (Math_Search_Space_Model) was the first attempt to formalize the intuition that metagenes warp the space organisms inhabit.
2. "What Kind of Thing is MEGA" was written as a constructive narrative that made predictions about failure modes and the mechanisms that resolve them.
3. The PFOL was derived next, formalizing exactly what the implementation does at the operator level.
4. The VFE telemetry was built on top of the PFOL, using the hyperbolic geometry from the first document as the structural prior.
5. GADS_Theory came last, asking why the subsystems had to be the subsystems they were and deriving the answer from locality and finiteness axioms.
6. The incremental simulation program in `theory_sim/` was built alongside GADS_Theory, both to test the predictions accumulated across the earlier work and to uncover dynamics the theoretical analysis alone could not reveal. Its findings informed Section 8 of GADS_Theory.

The documents have not yet been unified into a single coherent presentation. The PFOL is the ground truth for what MEGA is. GADS_Theory is the argument for why MEGA's architecture is forced rather than chosen. The Math document provides the geometric layer that GADS_Theory's Section A10 gestures at. The VFE document shows that the architecture supports a principled information-theoretic telemetry signal. The simulators make the dynamics visible layer by layer.

A reader engaging with all four documents will notice some drift between them. This reflects the staged development of the work rather than inconsistency in the underlying system.

---

## What is Complete and What is In Progress

Complete:

- The PFOL specification matches the implementation.
- The GADS subsystem derivation (Section 1 of GADS_Theory).
- The incremental phase-regime analysis (Layers 0 through 6 of Section 8).
- The hyperbolic embedding and Lorentz-boost structure.
- The three-term VFE decomposition.
- The incremental simulation program (`theory_sim/`) for Layers 0-6.

In active development:

- Formal tightening of the "necessary and complete" portion of the closure argument in GADS_Theory Section 1.
- A unified presentation that integrates all four documents into a single coherent work.
- Extended parameter-regime studies in the simulator to map the stability envelope of the full architecture.

The hackathon submission is not contingent on any of the in-progress work. The benchmark result stands on the environment and the ARI metric; MEGA's performance is reported as data; the formal framework is provided as reference material for readers who want to understand what kind of system the baseline is.

---

## Contact

**Matthew Andrews** — Independent researcher
Matthew.Andrews2024@gmail.com

The theoretical work has been developed independently over approximately three years without institutional affiliation. Questions, corrections, and collaboration inquiries are welcome.