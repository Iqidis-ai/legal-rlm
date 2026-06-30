#!/bin/bash
# init-ssl.sh - Initialize Let's Encrypt SSL certificates for rlm.iryslegal.com

set -e

DOMAIN="rlm.iryslegal.com"
EMAIL="admin@irys.ai"  # Change this to your email
STAGING=0  # Set to 1 to test with staging (no rate limits)

echo "============================================"
echo "SSL Certificate Setup for $DOMAIN"
echo "============================================"

# Create required directories
mkdir -p certbot/conf certbot/www

# Create temporary nginx config for initial certificate
echo "Creating temporary nginx config for certificate challenge..."
cat > nginx-init.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    server {
        listen 80;
        server_name rlm.iryslegal.com;

        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }

        location / {
            return 200 'SSL setup in progress...';
            add_header Content-Type text/plain;
        }
    }
}
EOF

# Stop any running containers
echo "Stopping existing containers..."
docker-compose down 2>/dev/null || true

# Start nginx with temporary config
echo "Starting nginx for certificate challenge..."
docker run -d --name nginx-init \
    -p 80:80 \
    -v "$(pwd)/nginx-init.conf:/etc/nginx/nginx.conf:ro" \
    -v "$(pwd)/certbot/www:/var/www/certbot" \
    nginx:alpine

# Wait for nginx to start
sleep 3

# Request certificate
echo "Requesting certificate from Let's Encrypt..."
if [ $STAGING -eq 1 ]; then
    STAGING_ARG="--staging"
else
    STAGING_ARG=""
fi

docker run --rm \
    -v "$(pwd)/certbot/conf:/etc/letsencrypt" \
    -v "$(pwd)/certbot/www:/var/www/certbot" \
    certbot/certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    $STAGING_ARG \
    -d $DOMAIN

# Stop temporary nginx
echo "Stopping temporary nginx..."
docker stop nginx-init
docker rm nginx-init
rm nginx-init.conf

# Verify certificate exists
if [ -f "certbot/conf/live/$DOMAIN/fullchain.pem" ]; then
    echo ""
    echo "============================================"
    echo "SUCCESS! Certificate obtained for $DOMAIN"
    echo "============================================"
    echo ""
    echo "Now start the full stack with:"
    echo "  docker-compose up -d --build"
    echo ""
    echo "Your API will be available at:"
    echo "  https://$DOMAIN/health"
    echo "  https://$DOMAIN/docs"
    echo ""
else
    echo ""
    echo "ERROR: Certificate not found. Check the logs above."
    exit 1
fi
