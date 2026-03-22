# Prompt Candidate b.v7.proposal.1

## Goal
- Improve mechanism understanding in PromptLadderTest (H2)
- Preserve output schema and cross-version comparability

## User Prompt Additions
1. First restate mechanism assumptions in one sentence.
2. Then compute utility gap: compensation - marginal privacy cost.
3. Finally return JSON only with keys `share` and `reason`.
