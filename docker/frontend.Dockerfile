# Phase 27 -- Frontend container (multi-stage: deps -> build -> run).
# Build context is the project root (see docker-compose.yml).
#
# IMPORTANT: Next.js inlines NEXT_PUBLIC_* env vars into the client bundle
# at BUILD time, not at container start time -- they must be passed as
# build ARGs (see docker-compose.yml's `build.args`), not just runtime
# `environment:` entries, or the frontend will silently keep pointing at
# whatever default lib/api-client.ts falls back to.

FROM node:20-alpine AS deps
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ .

ARG NEXT_PUBLIC_API_BASE_URL
ARG NEXT_PUBLIC_WS_BASE_URL
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}
ENV NEXT_PUBLIC_WS_BASE_URL=${NEXT_PUBLIC_WS_BASE_URL}

RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/next.config.mjs ./next.config.mjs

EXPOSE 3000
CMD ["npm", "start"]
