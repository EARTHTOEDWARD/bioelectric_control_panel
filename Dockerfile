## UI build stage
FROM node:20-alpine AS ui-builder
WORKDIR /ui
COPY bcp_frontend_v1/ ./
# Ensure asset base matches FastAPI mount path
ENV VITE_BASE=/app/
RUN npm install && npm run build

## API runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy project files before install to allow editable/local install
COPY pyproject.toml ./
COPY bcp ./bcp
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .[ws]

# Copy built UI into the image and point FastAPI to it
COPY --from=ui-builder /ui/dist /app/ui
ENV BCP_UI_STATIC_DIR=/app/ui

ENV PORT=8000
EXPOSE 8000

CMD ["python", "-m", "bcp.cli"]
