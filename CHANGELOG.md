# Changelog

## Unreleased

* `no_pii`: added `context_keywords` and `context_mode` (`"extend"` or `"replace"`) to
  `find_pii()`, `mask_pii()` and `NoPII` for custom SSN, routing number and Aadhaar/VID context
  keywords, matched case-insensitively with Unicode letter boundaries. Defaults are unchanged.
* Added tested LangGraph example (`examples/langgraph_agent/graph.py`), optional `[langgraph]`
  extra, integration tests (`tests/test_langgraph.py`), and a dedicated `langgraph` CI job.
* `no_pii`: added Aadhaar (12-digit, first digit 2-9) and Virtual ID (`aadhaar_vid`, 16-digit)
  detection using the Verhoeff checksum (`verhoeff_valid`), with context gating
  (`require_aadhaar_context=True` by default for `aadhaar`, `aadhar`, `uidai`, `virtual id`,
  `vid`) and opt-in PDF/OCR line-wrap support (`allow_line_breaks=False`).
* `pci_compliant`: catch card numbers immediately following a non-numeric period (such as
  `Paid.4111...`) while continuing to ignore decimal numbers (`0.4111...`).
* Pinned all GitHub Actions in `ci.yml` and `release.yml` to commit SHAs.
* Scoped `release.yml` write permissions (`contents: write`, `id-token: write`) to the
  `build-and-publish` job only, with top-level `permissions: {}`.

## 0.3.2

* Added real Jaeger OpenTelemetry trace screenshot (`docs/assets/jaeger-trace.png`) linked with
  an absolute URL so it renders on both GitHub and PyPI.

## 0.3.1

* Documentation & metadata polish for PyPI: updated README with `pip install agent-invariant-eval`,
  PyPI badge, cleaned tagline formatting, and added ORCID to `CITATION.cff`.

## 0.3.0

### Runtime and telemetry

* **Pre-execution span input gate.** `EvaluationSession._open()` now evaluates `Stage.SPAN`
  invariants on `span.input` before yielding control to the `with` block, blocking tool side
  effects on non-compliant inputs. `_close()` skips re-scanning unchanged pre-checked inputs so
  scanning stays linear.
* **Cycle guards and callback isolation.** `flatten_text()` and `collect_numbers()` guard against
  self-referential containers. `on_report` callback errors are logged without breaking verdicts.
* **Context & OTel fixes.** Cross-context `EvaluationSession.__exit__` no longer clobbers an outer
  session. `TelemetryEmitter` supports `with TelemetryEmitter(...) as emitter:` and exports
  `gen_ai.usage.output_tokens` on tool spans even when `input_tokens == 0`.

### Compliance

* **Known secret prefixes.** `find_pii()` detects AWS (`AKIA`/`ASIA`), GitHub (`gh[pousr]_`,
  `github_pat_`), Stripe (`sk_live_`/`rk_live_`/`sk_test_`/`rk_test_`), Anthropic (`sk-ant-`),
  OpenAI (`sk-proj-`), Google (`AIza`), and Slack (`xox[baprs]-`) credentials regardless of
  length, without duplicating `high_entropy_secret` findings.
* **Allowlists and BIN filtering.** Added `allowed_domains` (matching exact domains and
  subdomains) to `NoPII` and `mask_pii()`, `ignore_keys` to `NoPII` and `PCICompliant`,
  `require_known_bin` to `PCICompliant`, `find_pci()`, and `mask_pci()`, plus CLI flags
  `--allowed-domain` and `--require-known-bin` on `agent-eval scan`.
* **Fixed PAN false positives on decimals.** `_PAN_CANDIDATE` now ignores fractional parts of
  decimal numbers (`0.8475...`), epoch timestamps (`1700000123.456789`), large amounts, and OIDs,
  accepting dot separators only in strict `4.4.4.4` card grouping (`4111.1111.1111.1111`).

### Packaging and CI

* **Python 3.11+ support.** Replaced PEP 695 generics in `find_cycle` with `TypeVar`.
* **Optional extras.** Moved `typer` to `[cli]` and `opentelemetry-exporter-otlp-proto-http` to
  `[otlp]` (with `[all]` and `[dev]`).
* **CI, security & release workflows.** Added GitHub Actions CI (`permissions: contents: read`,
  `ruff`, `mypy --strict`, `pytest --cov` with 90% floor, `run_eval.py`, `python -m build` across
  Python 3.11–3.13), `SECURITY.md`, `CONTRIBUTING.md`, `CITATION.cff`, issue templates, and PyPI
  Trusted Publishing release workflow with tag-vs-version verification.

## 0.2.0

Addresses an external review of 0.1.0. Behaviour changes are marked **changed**.

### Runtime and concurrency

* Open span stack is now context local (`contextvars`). Parallel branches under
  `asyncio.gather`, `TaskGroup` and `to_thread` attach to the correct parent. Recording and
  checks are serialised by a lock, so threads can record safely with an explicit `parent_id`.
* PII and PCI scanning is incremental. A span boundary scans only the span that closed
  (`BaseInvariant.check_span`). The output stage still scans everything once.
* `@evaluate_agent` supports generator and async generator agents with `stream="buffer"`
  (default, fail closed) and `stream="passthrough"`. A plain function that returns a generator
  or coroutine object now fails closed with a `TypeError`.
* `wrapped.with_report(...)` returns `(result, report)` and is race free. `last_report` stays
  as a single threaded convenience.

### Invariants

* **changed** `numeric_grounding`: numbers inside ids, dates and times in tool strings no
  longer ground a figure. New `ground_from_strings` option. More currencies (symbols, ISO
  codes in upper case, Indian digit grouping).
* **changed** `no_pii`: a bare nine digit number is a routing number only with a routing
  keyword and a valid ABA checksum. `require_routing_context=False` restores checksum only.
  Context keywords now match keys such as `routing_number` and `ssn_number`.
* **changed** `no_context_bloat`: baseline floored at `min_baseline_tokens` (500).
* **changed** `no_swallowed_tool_failures`: the default acknowledgement pattern accepts
  inability phrases only. Apologies and bare words such as "error" no longer count. New
  `allow_acknowledgement` option.
* **changed** `no_routing_loops`: counts repeated directed handoffs (`max_edge_repeats`, default
  1). `max_visits_per_agent` is now optional and off by default. Hub and spoke supervisors pass.
* **changed** Token totals deduplicate rolled up parent usage
  (`Trajectory.token_attribution()`). SLA budgets, context bloat and costs use it.

### Telemetry, benchmark, CLI

* OTel trace and span ids now equal the trajectory ids. Every child span carries
  `agent_eval.span_id`, and `agent_eval.violations` when an invariant failed on it.
  **changed** root attribute `gen_ai.conversation.id` replaced by `agent_eval.trace_id`
  (a trace id is not a conversation id).
* `FrontierRunner` accepts async agents, `max_concurrency`, and `await runner.arun(...)`.
* `ModelConfig.pricing` prices each span by its model for mixed model runs.
* `agent-eval benchmark --models` accepts a JSON file or `module:attribute`. Added
  `--max-concurrency`. `examples/payment_dispute_agent/models.json` is now shipped.
* CLI tests added.

### Follow up fixes

* Abandoning a `passthrough` stream early evaluates the partial output instead of recording an
  agent crash.
* Token roll up is detected per side (input and output), which can only over count.
* Stricter digit grouping. Money in an unsupported number format (`€12,50`, `1.234,56 EUR`)
  now fails `numeric_grounding` as an unparseable figure instead of being skipped.

## 0.1.0

Initial release.
