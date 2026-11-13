# Instructions for AI assistants

Follow https://github.com/romkey/python-ai-rules/blob/main/RULES.md.
Follows python-ai-rules v0.1.0. When re-scanning, apply the entries in
its UPGRADING.md newer than that, then update this line.
With Claude Code, the `python-rules` plugin loads them for you.

Read docs/overview.md first. `main` is protected: work on a branch and
open a pull request; CI and Lint must pass before it can merge.

<!-- Project-specific notes go below. -->

- SQLite, not PostgreSQL, by the owner's choice: one process, small data.
- The distribution and command are `chat-hacksman`, not the repo name
  (`pdxhackerspace-chat-hacksman`), by the owner's choice; the image is
  named after the repo.
- Embedding query prefixes live in `KNOWN_QUERY_PREFIXES` in
  `src/chat_hacksman/config.py`; keep it, `.env.example` and
  `docs/configuration.md` in step.
