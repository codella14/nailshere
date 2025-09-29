
### 1. Clone and Setup

```bash
# Clone your repository
git clone https://github.com/codella14/nailshere.git
cd nailshere

# Ensure your model file is in place
ls model/best_10k.pt  # Should exist
```

### 2. Build and Run

```bash
# Build the Docker image
docker-compose build

# Start the services
docker-compose up -d

# Check if services are running
docker-compose ps
```

### 3. Verify Deployment

```bash
# Check application logs
docker-compose logs web

# Test the application
curl http://localhost:5001
```

### uv commands

1. create virtual environment
```bash
uv venv
```

2. sync environment (optional)
```bash
uv sync
```

3. run project
```bash
uv run python app.py
```