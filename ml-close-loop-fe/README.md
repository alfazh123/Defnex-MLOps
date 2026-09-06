# DEFNEX MLOps — Frontend

React Router 8 + TypeScript dashboard for the DEFNEX closed-loop MLOps platform.

The app talks to the FastAPI backend in [`ml-close-loop-be/`](../ml-close-loop-be/)
(see its [README](../ml-close-loop-be/README.md) for API docs, auth, and setup).

## Current routes

- `/` — home/dashboard
- `/chat` — chat page

## Stack

React 19 · React Router 8 · TypeScript · Tailwind CSS · shadcn/ui + Base UI

## Requirements

- Node.js ≥ 20 (the `Dockerfile` uses Node 24)
- [pnpm](https://pnpm.io/)

## Commands

```bash
pnpm install              # install dependencies (use --frozen-lockfile in CI)
pnpm run dev              # dev server with HMR, http://localhost:3000
pnpm run typecheck        # react-router typegen + tsc
pnpm run build            # production build (react-router build)
pnpm run start            # serve the production build
```

## Development

Start the backend first (`ml-close-loop-be/`, see its README quickstart), then:

```bash
pnpm install
pnpm run dev
```
