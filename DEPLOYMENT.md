# Nail Detection Flask App - Production Deployment Guide

This guide provides instructions for deploying the nail detection Flask application to a live server using Docker.

## Prerequisites

- Docker and Docker Compose installed on your server
- At least 4GB RAM available for the application
- 10GB+ free disk space for models and uploaded files
- Domain name (optional, for SSL setup)

## Quick Start

### 1. Clone and Setup

```bash
# Clone your repository
git clone <your-repo-url>
cd nail_detection_flask

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
curl http://localhost/
```

## Production Configuration

### Environment Variables

You can customize the deployment by setting environment variables in `docker-compose.yml`:

```yaml
environment:
  - MODEL_PATH=model/best_10k.pt  # Path to your YOLO model
  - PYTHONUNBUFFERED=1
  - PYTHONDONTWRITEBYTECODE=1
```

### Resource Limits

The current configuration includes:
- Memory limit: 4GB
- Memory reservation: 2GB
- 4 Gunicorn workers
- Nginx reverse proxy with rate limiting

### File Storage

The following directories are mounted as volumes for persistence:
- `./static/uploads` - User uploaded images
- `./static/segmentations` - Processed segmentation results
- `./static/results` - Generated 3D models and PBR maps
- `./logs` - Application logs
- `./user_models` - Custom user models

## SSL/HTTPS Setup (Optional)

### Using Let's Encrypt with Certbot

1. Install Certbot:
```bash
sudo apt update
sudo apt install certbot python3-certbot-nginx
```

2. Obtain SSL certificate:
```bash
sudo certbot --nginx -d yourdomain.com
```

3. Update nginx.conf to redirect HTTP to HTTPS:
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    # ... rest of your nginx configuration
}
```

## Monitoring and Maintenance

### Health Checks

The application includes health checks:
- Container health check: `curl -f http://localhost:8000/`
- Nginx health endpoint: `http://yourdomain.com/health`

### Log Management

```bash
# View application logs
docker-compose logs -f web

# View nginx logs
docker-compose logs -f nginx

# Rotate logs (add to crontab)
0 0 * * * docker-compose exec nginx nginx -s reload
```

### Backup Strategy

```bash
# Backup uploaded files and results
tar -czf backup-$(date +%Y%m%d).tar.gz static/ logs/

# Backup with Docker volumes
docker run --rm -v nail_detection_flask_static:/data -v $(pwd):/backup alpine tar czf /backup/static-backup.tar.gz -C /data .
```

## Scaling

### Horizontal Scaling

To scale the application:

```yaml
# In docker-compose.yml
services:
  web:
    # ... existing config
    deploy:
      replicas: 3
```

### Load Balancer Configuration

For multiple instances, update nginx.conf:

```nginx
upstream flask_app {
    server web_1:8000;
    server web_2:8000;
    server web_3:8000;
}
```

## Performance Optimization

### Gunicorn Configuration

Adjust worker count based on CPU cores:
```bash
# For 8 CPU cores
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "8", ...]
```

### Nginx Optimization

Enable caching for static files:
```nginx
location /static/ {
    alias /var/www/static/;
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```bash
   # Increase memory limits in docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 8G
   ```

2. **Model Loading Issues**
   ```bash
   # Check model file exists
   docker-compose exec web ls -la model/
   
   # Check model path environment variable
   docker-compose exec web env | grep MODEL_PATH
   ```

3. **Permission Issues**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER static/ logs/
   ```

### Debug Mode

For debugging, run in development mode:
```bash
# Override the command in docker-compose.yml
services:
  web:
    # ... other config
    command: ["python", "app.py"]
```

## Security Considerations

1. **Firewall Configuration**
   ```bash
   # Allow only necessary ports
   sudo ufw allow 22    # SSH
   sudo ufw allow 80    # HTTP
   sudo ufw allow 443   # HTTPS
   sudo ufw enable
   ```

2. **Regular Updates**
   ```bash
   # Update base images
   docker-compose pull
   docker-compose up -d
   ```

3. **Monitor Resource Usage**
   ```bash
   # Check resource usage
   docker stats
   ```

## API Usage

### Endpoints

- `GET /` - Main application interface
- `POST /detect` - Nail detection and segmentation
- `POST /3d_converter` - Complete 3D conversion workflow
- `GET /static/results/<filename>` - Download generated files

### Rate Limits

- API endpoints: 10 requests/second
- Upload endpoints: 2 requests/second
- Burst limits configured for handling traffic spikes

## Support

For issues or questions:
1. Check application logs: `docker-compose logs web`
2. Verify system resources: `docker stats`
3. Test individual components: `curl http://localhost/health`

## Production Checklist

- [ ] Model file (`model/best_10k.pt`) is present
- [ ] Sufficient disk space (10GB+)
- [ ] Memory allocation (4GB+)
- [ ] SSL certificate configured (if using HTTPS)
- [ ] Firewall rules configured
- [ ] Backup strategy implemented
- [ ] Monitoring setup
- [ ] Log rotation configured
- [ ] Health checks working
- [ ] Rate limiting tested
