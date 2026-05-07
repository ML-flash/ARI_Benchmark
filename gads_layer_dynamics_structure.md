# GADS Dynamics Layer Structure

This document is a reusable structure for writing each dynamics layer. Its purpose is to keep the argument straight while extracting theory from the functioning PFOL/codebase.

The governing order is:

```text
code behavior -> PFOL operational semantics -> local events -> state changes -> flux equations -> predictions -> verification -> interpretation
```

The dynamics argument should not begin with interpretation. It should begin with the local process and build upward.

---

## 1. Layer Header

```text
Layer N: [Layer Name]
Status: [verified / partially verified / placeholder / conjectural]
Active mechanism: [one-sentence description]
Primary question: What does this layer add to the dynamics?
```

Example:

```text
Layer 1: Capture / Coagulation
Status: verified, equation cleanup pending
Active mechanism: Context-1 capture converts delimited regions into reusable composition identifiers.
Primary question: How does capture transform Layer 0 boundary geometry into compositional identity dynamics?
```

---

## 2. Purpose of the Layer

State why this layer exists in the staged dynamics program.

This section should answer:

- What mechanism is being isolated?
- What previous layer is being extended?
- What new feedback path is introduced?
- What behavior should become visible only after this layer is active?

Template:

```text
Layer N extends Layer N-1 by enabling [mechanism].
The purpose of this layer is to isolate the dynamical consequences of [mechanism] while holding [other mechanisms] inactive.
```

Avoid broad interpretation here. Do not begin with claims like “this creates adaptive intelligence” or “this completes the system.” Begin with the concrete mechanism.

---

## 3. Active and Inactive Subsystems

List what is enabled and disabled. This prevents accidental mixing of mechanisms.

```markdown
### Active
- [Subsystem/event]
- [Subsystem/event]

### Inactive
- [Subsystem/event]
- [Subsystem/event]
```

Example for Layer 0:

```markdown
### Active
- Boundary insertion
- Boundary removal
- Boundary swap/mobility
- Point mutation
- Local insertion/deletion
- Depth-0 crossover

### Inactive
- Capture
- MG
- MCO
- PR
- DB
- EA reuse
- Decode/fitness selection
```

---

## 4. State Vector

Define the smallest set of variables needed to describe the layer.

Template:

```text
X_N(t) = (...)
```

Each variable should have an operational definition.

```markdown
| Variable | Meaning | How measured in code | Status |
|---|---|---|---|
| A0 | Number of depth-0 non-delimiter positions | population scan | exact observable |
| A1 | Number of interior non-delimiter positions | population scan | exact observable |
| A2 | Number of delimiter tokens | population scan | exact observable |
```

Rules:

- Do not introduce a state variable unless it is needed for a later equation or prediction.
- If a variable is not directly observable, mark it as inferred or closure-dependent.
- If a variable summarizes a distribution, say what information is lost by summarizing it.

---

## 5. Event Table

This is the core of each dynamics layer.

For each event, specify:

1. Eligibility: when the event can occur.
2. Rate/probability: how often it fires.
3. State change: what it does to the state vector.
4. Notes: special branches, duplicate behavior, boundary cases, or invariants.

Template:

```markdown
| Event | Eligibility | Rate / probability | State change | Notes |
|---|---|---|---|---|
| [event] | [where it can occur] | [operator probability] | [Delta state] | [conditions] |
```

Example for Layer 1 capture:

```markdown
| Event | Eligibility | Rate / probability | State change | Notes |
|---|---|---|---|---|
| Capture of segment length k | Interior token visit inside segment with k >= 2 | pi_C per interior token visit | Delta A0 = +1; Delta A1 = -k; Delta A2 = -2; Delta N = -(k+1) | If content is novel, Delta |M| = +1. If duplicate, Delta |M| = 0 but organism still compresses to existing ID. |
```

Rules:

- If the event table is not clear, the layer is not ready for dynamics.
- Every equation later in the layer should be traceable back to this table.
- Every interpretive claim should depend on one or more rows in this table.

---

## 6. Exact Local Consequences

Write the consequences that follow directly from the event table.

These are not approximations. They are local facts about individual events.

Template:

```text
A single [event] changes the state by:
Delta A0 = ...
Delta A1 = ...
Delta A2 = ...
Delta N  = ...
```

Example:

```text
A capture event of interior length k replaces [x1 ... xk] with one composition token.
Therefore:
Delta A0 = +1
Delta A1 = -k
Delta A2 = -2
Delta N  = -(k + 1)
```

This section should be mathematically simple. If it becomes complicated, the event table is probably not decomposed enough.

---

## 7. Conservation Laws and Invariants

List what must remain true if the dynamics are written correctly.

Examples:

```markdown
| Invariant | Meaning | Check |
|---|---|---|
| A0 + A1 + A2 = N | State partition exhausts population mass | algebra and population scan |
| Boundary markers are paired | A2 changes in units of 2 under insertion/removal/capture | event table |
| No nesting | Delimited regions do not overlap recursively | invariant check in code |
| Decode preservation under deletion | Inlining must preserve phenotypic expansion | Layer 5+ only |
```

Rules:

- If an equation violates an invariant, the equation is wrong or missing a term.
- Invariants are often the fastest way to catch mistakes like incorrect length-change terms.

---

## 8. Aggregate Flux Equations

Build aggregate dynamics by summing event contributions:

```text
expected change = sum over events of (event rate) x (state change caused by event)
```

Separate equations by certainty level.

### 8.1 Exact Operator Fluxes

These come directly from PFOL/code-level operator probabilities.

Example:

```text
Boundary insertion flux ∝ pi_9 A0
Boundary removal flux ∝ pi_6 A2
```

### 8.2 Mean-Field Equations

These aggregate over the population and usually assume that local structure is summarized by the state variables.

Example structure:

```text
dA2/dt = [creation terms] - [destruction terms]
dA0/dt = [exposure creation terms] - [exposure consumption terms]
dA1/dt = [interior creation terms] - [interior consumption terms]
dN/dt  = [length creation terms] - [length destruction terms]
```

### 8.3 Distribution Equations

Use when a scalar summary is insufficient.

Examples:

```text
n_k(t): number of boundary segments with interior length k
P(k,t): segment-length distribution
G(g,t): gap-length distribution
```

A layer may need distribution dynamics if the mechanism depends on segment length, gap length, rank, age, or graph structure.

---

## 9. Closure Assumptions and Residual Terms

Every coarse-grained dynamics needs closures. Name them explicitly.

Template:

```markdown
| Closure / residual | Why needed | How handled | Status |
|---|---|---|---|
| eta_seq | Start-of-generation n_k changes during mutation scan | measured empirically | open closure |
| kbar closure | Aggregate equations need mean segment length | direct measurement or dense-regime approximation | partially closed |
```

Rules:

- Do not hide residuals.
- A named residual is progress, not failure.
- If a quantity cannot yet be derived from the state vector, mark it as an open closure.

---

## 10. Predictions

State predictions that could be wrong.

Good predictions are measurable and specific.

Template:

```markdown
| Prediction | Observable | Expected result | Failure would mean |
|---|---|---|---|
| [claim] | [measured quantity] | [sign/rate/correlation/distribution] | [what breaks] |
```

Examples:

```markdown
| Prediction | Observable | Expected result | Failure would mean |
|---|---|---|---|
| Capture is flat per interior-token visit | capture rate by k | approximately pi_C for k >= 2 | capture implementation or measurement is wrong |
| Segment-level capture is length-biased | per-segment capture probability | increases with k as 1 - (1 - pi_C)^k | Context-1 kernel interpretation is wrong |
| Capture reduces physical length | Delta N after capture | Delta N = -(k+1) | state-change accounting is wrong |
```

Avoid predictions like “the system becomes more adaptive” unless adaptivity has a defined observable.

---

## 11. Verification Protocol

Say exactly how the prediction is tested.

Template:

```markdown
| Test | Setup | Measurement window | Compared quantities | Pass/fail rule |
|---|---|---|---|---|
| [test name] | [parameters / active layer] | [generations] | [theory vs actual] | [threshold] |
```

Rules:

- Use targeted tests, not just full-system runs.
- State whether measurement is mutation-only, full generation, pre-singularity, post-singularity, etc.
- Record parameter values.
- Record seeds if relevant.

---

## 12. Results

Report results plainly.

Template:

```markdown
| Quantity | Theory | Observed | Error / ratio | Status |
|---|---:|---:|---:|---|
| [quantity] | [value] | [value] | [ratio] | PASS / WEAK / FAIL |
```

Then include a short interpretation of what the results do and do not establish.

Rules:

- Do not upgrade a result beyond what the test measured.
- If a test only verifies a pre-singularity regime, say so.
- If a result depends on a parameter range, say so.

---

## 13. Interpretation

Only after local rules, equations, and verification should the layer receive conceptual interpretation.

Template:

```text
The verified dynamics show that [mechanism] produces [specific dynamical effect].
This matters because [relation to the larger GADS process].
However, this layer does not yet establish [limits].
```

Example:

```text
Layer 1 shows that capture converts boundary-protected physical regions into reusable composition identities. This converts flat partition structure into grammatical depth, but it also consumes boundary surface and creates a contraction pressure. Therefore Layer 1 explains both the escape from Layer 0 flat capacity and the onset of singularity pressure.
```

Interpretation should be constrained by the verified claims.

---

## 14. Layer Status

End every layer with a status block.

Template:

```markdown
### Status

- Verified:
  - [claim]
  - [claim]

- Partially verified:
  - [claim]

- Open closure terms:
  - [term]

- Not claimed here:
  - [claim beyond current layer]

- Next required work:
  - [next verification or derivation]
```

Example:

```markdown
### Status

- Verified:
  - Capture fires flat per interior-token visit for k >= 2.
  - Capture of length k changes total length by -(k+1).
  - Realized capture flux strongly predicts dN, dA2, dA0, and dA1 in the pre-singularity window.

- Partially verified:
  - Start-of-generation R_hat predicts capture flux but overcounts without eta_seq.

- Open closure terms:
  - eta_seq, the sequential scan coupling between Layer 0 restructuring and Layer 1 capture.

- Not claimed here:
  - Regulation by open, MCO, PR, DB, EA, or fitness selection.

- Next required work:
  - Derive or empirically characterize eta_seq under different parameter regimes.
```

---

# Claim Types

Every important statement in the dynamics should be one of these types.

```markdown
| Type | Meaning | Example |
|---|---|---|
| Definition | A variable or object is defined | A0 is the number of depth-0 positions. |
| Operator rule | PFOL/code says an event can occur | Capture fires per interior-token visit with probability pi_C. |
| Local consequence | Direct state change from one event | Capture of length k gives Delta N = -(k+1). |
| Aggregate equation | Sum of expected event effects | dA2/dt = creation - destruction. |
| Closure | Approximation needed to close equations | kbar approx N/A2 - 1 in dense regime. |
| Empirical result | Measured in simulation | observed/theory ratio = 0.998. |
| Conjecture | Plausible but not verified | PR/DB may bound MG size. |
| Interpretation | Meaning of verified dynamics | Capture converts physical length into grammatical depth. |
```

If a paragraph mixes several claim types, split it.

---

# Layer Readiness Checklist

A layer is ready to be written as dynamics only if most of these are available:

```markdown
- [ ] Active and inactive subsystems are explicit.
- [ ] State vector is defined.
- [ ] Event table is complete.
- [ ] Exact local state changes are known.
- [ ] Invariants are checked.
- [ ] Aggregate equations are derived from event rows.
- [ ] Closure assumptions are named.
- [ ] Predictions are measurable.
- [ ] Verification protocol exists.
- [ ] Results are reported.
- [ ] Interpretation is limited to what was shown.
- [ ] Status block separates verified, partial, conjectural, and not-claimed material.
```

If the event table is missing, the layer is a placeholder.

---

# Minimal Placeholder Format

For layers not yet developed, use this instead of a full dynamics section.

```markdown
## Layer N: [Name]

Status: placeholder.

PFOL mechanism reserved for this layer:
[one paragraph]

Expected dynamical role:
[one paragraph]

State variables likely needed:
- ...

Events likely needed:
- ...

Not claimed yet:
- No equations are claimed.
- No phase behavior is claimed.
- No verification is claimed.
```

This preserves the scaffold without making the placeholder look like a completed theory.

---

# Working Standard

The standard for the dynamics is not perfection. The standard is disciplined traceability.

Every dynamics claim should trace through this chain:

```text
PFOL/code event
-> local state change
-> aggregate flux or distribution equation
-> measurable prediction
-> verification or named open residual
```

If a claim cannot trace through that chain, it can remain in notes, but it should not be presented as developed dynamics.

