---
context:
  keywords: [project, overview, framework]
  task_types: [any]
  priority: high
  max_tokens: 1500
  children:
    - path: .claude/CLAUDE.md
      when: [claude, runtime, workflow]
    - path: council/CLAUDE.md
      when: [module, feature]
  dependencies: []
---
# ai-council

## Purpose

See [README.md](README.md) for full documentation.

