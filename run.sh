#!/bin/bash

# RAG Pipeline Startup Script

echo "==================================="
echo "RAG Pipeline Startup"
echo "==================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please copy .env.example to .env and configure your settings:"
    echo "  cp .env.example .env"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running!"
    echo "Please start Docker and try again."
    exit 1
fi

# Start PostgreSQL with PgVector
echo "📦 Starting PostgreSQL with PgVector..."
docker-compose up -d

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 5

# Check if PostgreSQL is healthy
until docker exec ragpipeline_postgres pg_isready -U postgres > /dev/null 2>&1; do
    echo "   Still waiting for PostgreSQL..."
    sleep 2
done

echo "✅ PostgreSQL is ready!"

# Start FastAPI application
echo "🚀 Starting FastAPI application..."
echo "   API will be available at: http://localhost:8000"
echo "   API docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "==================================="

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
