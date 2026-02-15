• Master Execution Plan Complete ✅

  I have meticulously reviewed all 7 design documents and created a comprehensive execution plan. Here's my analysis summary:

  Documents Analyzed

   Document                   Content                            Key Insights
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   new_design_1.md            Executive summary & architecture   Research Terminal aesthetic, Clean Architecture layers
   new_design_2.md            Sprint 1 - Foundation              Configuration, database, circuit breaker
   new_design_3.md            Sprint 2 - Core Services           Event bus, cost tracker, orchestrator
   new_design_4.md            Sprint 3 - Avant-Garde UI          CSS tokens, bento-grid, components
   new_design_5.md            Sprint 4 - Production              Docker, observability, E2E tests
   new_design_critique.md     Frontend CSS strategy              Phosphor Grid implementation
   new_design_critique_2.md   Architectural audit                17 concrete defects identified

  Critical Defects Requiring Phase 0 Fixes

   ID    Defect                               Impact                   Fix Location
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   C-1   TokenUsage.cost property with args   Runtime crash            domain/models.py
   C-2   SQLite PRAGMA never enabled          Data corruption          infrastructure/database.py
   C-3   Repository session leak              Connection exhaustion    infrastructure/repositories.py
   C-4   Circuit breaker + AsyncGenerator     Streaming crashes        infrastructure/openrouter.py + orchestrator.py
   C-5   Missing import json                  Runtime crash            infrastructure/openrouter.py
   C-6   Orphan EventBus instances            Silent budget failures   application/cost_tracker.py
   C-7   Broken unit test                     Test suite fails         tests/unit/test_circuit_breaker.py

  Deliverables Created

  1. MASTER_EXECUTION_PLAN.md (36,911 bytes)
    • 6 phases with detailed breakdown
    • Each file: description, interface definition, TDD checklist
    • Validation criteria per phase
    • Success metrics and risk register
  2. TODO.md (17,000 bytes)
    • Granular task checklist for every file
    • RED/GREEN/REFACTOR TDD workflow
    • Progress tracking with legend

  Phase Breakdown

   Phase   Duration   Focus            Key Outputs
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   0       2-3 hrs    Critical fixes   7 defect fixes, all tests pass
   1       4-6 hrs    Foundation       pyproject.toml, domain models, config
   2       6-8 hrs    Infrastructure   Database, repository, OpenRouter client
   3       6-8 hrs    Application      Events, cost tracking, orchestrator
   4       8-10 hrs   UI Components    Terminal aesthetic, XSS protection
   5       6-8 hrs    Integration      Streamlit app, streaming bridge
   6       6-8 hrs    Production       Docker, E2E tests, observability

  Total Estimated Effort: 38-51 hours

  TDD Methodology Applied

  Every file includes:

  1. RED: Write failing test first
  2. GREEN: Implement minimal code to pass
  3. REFACTOR: Clean up while keeping tests green

  Validation Strategy

  • Unit Tests: Individual component behavior
  • Integration Tests: Database, API clients, streaming
  • E2E Tests: Full conversation flows
  • Security Tests: XSS penetration testing
  • Performance Tests: Time-to-first-token, render speed

  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Review Request

  Please review the following before I proceed to execution:

  1. Phase breakdown - Are the 6 phases logical? Should any be combined or split?
  2. Priority of Phase 0 - Are you aligned with fixing all critical bugs first?
  3. Technology choices - Any objections to:
    • Poetry for dependency management
    • SQLAlchemy 2.0 with SQLite
    • Bleach for XSS protection
    • IBM Plex Mono + Instrument Sans fonts (per critique recommendation)
  4. Success criteria - Are the targets appropriate?
    • 99.9% message delivery
    • <800ms time-to-first-token
    • Cost accuracy ±$0.0001

