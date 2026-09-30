- source_spec: `_bmad-output/specs/spec-ai-finance-assistant/stories/1-project-skeleton-and-alpha-vantage-client.md`
  summary: A live empty `{"Global Quote": {}}` overwrites any stale real quote for that symbol and raises SymbolNotFoundError for a full TTL, even for bundled symbols.
  evidence: Unverified (maybe-false); would be medium. It is settled by observing whether Alpha Vantage ever returns an empty Global Quote for a valid ticker. If it does, keep the stale entry separate from the not-found marker.
- source_spec: `_bmad-output/specs/spec-ai-finance-assistant/stories/1-project-skeleton-and-alpha-vantage-client.md`
  summary: An Alpha Vantage `Information` reply is always treated as a rate limit, so an invalid API key could look like a quota hit and silently serve mock data.
  evidence: Unverified (maybe-false); would be medium. It is settled by calling GLOBAL_QUOTE with a deliberately wrong key and checking whether the reply uses `Information` or `Error Message`. If it uses `Information`, distinguish on message text (changes the frozen matrix row, so it needs the human).
