# ==============================================================================
# Multi-Stage Production Dockerfile for React Dashboard
# Stage 1: Build React/Vite application into static bundle
# Stage 2: Serve via lightweight Nginx Alpine container with reverse proxy
# ==============================================================================

# --- Stage 1: Build Environment ---
FROM node:20-alpine AS build

WORKDIR /app

# Install project dependencies using lockfile
COPY frontend/package*.json ./
RUN npm ci --prefer-offline --no-audit

# Copy frontend source code and configuration
COPY frontend/ ./

# Compile production-optimized bundle
RUN npm run build

# --- Stage 2: Runtime Web Server ---
FROM nginx:1.27-alpine AS runtime

# Remove default Nginx site configuration
RUN rm -rf /etc/nginx/conf.d/default.conf /usr/share/nginx/html/*

# Copy custom reverse proxy and routing configuration
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

# Copy compiled static assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Expose standard HTTP port
EXPOSE 80

# Production healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:80/healthz || exit 1

# Run Nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
