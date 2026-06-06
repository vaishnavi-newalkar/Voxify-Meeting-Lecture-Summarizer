# ── Stage 1: Build React app ──────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Accept backend URL as a build argument (empty = same-origin proxy via nginx)
ARG VITE_API_URL=""

# Install dependencies first (layer cache)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install

# Copy source and build
COPY frontend/ .

# Bake the backend URL into the production bundle
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

# ── Stage 2: Serve with nginx ────────────────────────────────────────────────
FROM nginx:alpine

# Install envsubst (part of gettext — not included in nginx:alpine by default)
RUN apk add --no-cache gettext

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built React app from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD wget -qO- http://localhost:80/ || exit 1

# Default backend URL — overridden by Render environment variable
ENV BACKEND_URL=https://voxify-backend.onrender.com

# Substitute ${BACKEND_URL} in nginx config at container startup, then launch nginx
CMD ["/bin/sh", "-c", "envsubst '${BACKEND_URL}' < /etc/nginx/conf.d/default.conf > /tmp/default.conf && cp /tmp/default.conf /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]