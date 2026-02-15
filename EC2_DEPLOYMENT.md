# EC2 Deployment Guide — Legal Knowledge Graph Backend

## 1. Recommended Instance

| Spec | Minimum | Recommended |
|------|---------|-------------|
| Instance type | `t3.medium` (2 vCPU, 4 GB) | `t3.large` (2 vCPU, 8 GB) |
| Storage | 20 GB gp3 | 40 GB gp3 |
| OS | Ubuntu 22.04 LTS (ami) | Ubuntu 22.04 LTS |

> FAISS index + parallel extraction threads benefit from ≥ 4 GB RAM.
> If you expect many concurrent requests or very large documents, consider `t3.xlarge` (4 vCPU, 16 GB).

---

## 2. Security Group (Firewall)

| Rule | Port | Source |
|------|------|--------|
| SSH | 22 | Your IP / bastion |
| App (HTTP) | 8000 | Your frontend / ALB |
| HTTPS (optional) | 443 | 0.0.0.0/0 (if Nginx + SSL) |

> **Never** expose port 8000 to 0.0.0.0/0 without a reverse proxy + auth.

---

## 3. System Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+ & build dependencies
sudo apt install -y python3.11 python3.11-venv python3.11-dev \
    build-essential libpq-dev git

# (Optional) Install Nginx for reverse proxy
sudo apt install -y nginx
```

---

## 4. Application Setup

```bash
# Clone your repo
cd /opt
sudo mkdir -p kg-backend && sudo chown $USER:$USER kg-backend
git clone <your-repo-url> kg-backend
cd kg-backend

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install Gunicorn for production WSGI serving
pip install gunicorn
```

---

## 5. Environment Configuration

```bash
cp .env.example .env   # or create from scratch
nano .env
```

```env
GEMINI_API_KEY="your-gemini-key"
APP_ENV=production

development_POSTGRES_URL="postgresql://user:pass@dev-host:5432/db?sslmode=require"
staging_POSTGRES_URL="postgresql://user:pass@staging-host:5432/db?sslmode=require"
production_POSTGRES_URL="postgresql://user:pass@prod-host:5432/db?sslmode=require"

AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...

GEMINI_REQUESTS_PER_MINUTE=60
GEMINI_MAX_PARALLEL_WORKERS=6
MAX_DOWNLOAD_PARALLEL=6
```

> **Tip**: Use AWS Secrets Manager or SSM Parameter Store instead of `.env` for production secrets.

---

## 6. Running with Gunicorn (Production)

```bash
# Start with 2 sync workers (Flask + threads inside extraction)
gunicorn \
  --bind 0.0.0.0:8000 \
  --workers 2 \
  --timeout 600 \
  --access-logfile /var/log/kg-backend/access.log \
  --error-logfile /var/log/kg-backend/error.log \
  "visualization_server:create_visualization_app()"
```

> `--timeout 600` is important because document extraction can take minutes per request.

---

## 7. Systemd Service (Auto-start on Boot)

Create `/etc/systemd/system/kg-backend.service`:

```ini
[Unit]
Description=Knowledge Graph Backend
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/kg-backend
Environment="PATH=/opt/kg-backend/venv/bin:/usr/bin"
EnvironmentFile=/opt/kg-backend/.env
ExecStart=/opt/kg-backend/venv/bin/gunicorn \
    --bind 0.0.0.0:8000 \
    --workers 2 \
    --timeout 600 \
    --access-logfile /var/log/kg-backend/access.log \
    --error-logfile /var/log/kg-backend/error.log \
    "visualization_server:create_visualization_app()"
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Create log directory
sudo mkdir -p /var/log/kg-backend && sudo chown ubuntu:ubuntu /var/log/kg-backend

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable kg-backend
sudo systemctl start kg-backend

# Check status
sudo systemctl status kg-backend
journalctl -u kg-backend -f   # tail logs
```

---

## 8. Nginx Reverse Proxy (Optional — SSL + Domain)

```nginx
# /etc/nginx/sites-available/kg-backend
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 600s;   # match gunicorn timeout
        proxy_send_timeout 600s;
        client_max_body_size 50M;  # allow large document uploads
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/kg-backend /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# SSL with Let's Encrypt
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## 9. Elastic IP

Allocate an Elastic IP in the AWS Console and associate it with your EC2 instance so the public IP persists across stop/start cycles. Point your DNS (A record) at this IP.

---

## 10. Monitoring & Health Check

### Health check endpoint

```bash
curl http://localhost:8000/api/stats?matter_id=test
# Should return JSON (even if empty graph) — confirms the server is alive
```

### CloudWatch Agent (optional)

```bash
sudo apt install -y amazon-cloudwatch-agent
# Configure to ship /var/log/kg-backend/*.log to CloudWatch Logs
```

### Log rotation

```bash
# /etc/logrotate.d/kg-backend
/var/log/kg-backend/*.log {
    daily
    rotate 14
    compress
    missingok
    notifempty
    postrotate
        systemctl reload kg-backend > /dev/null 2>&1 || true
    endscript
}
```

---

## 11. Security Hardening

```bash
# Firewall (UFW)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Keep .env file protected
chmod 600 /opt/kg-backend/.env

# Disable root login
sudo sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart sshd
```

---

## 12. Updating the Application

```bash
cd /opt/kg-backend
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart kg-backend
```

---

## 13. Quick Reference — Key Ports & Paths

| Item | Value |
|------|-------|
| App port | `8000` |
| App directory | `/opt/kg-backend` |
| Virtual env | `/opt/kg-backend/venv` |
| Env config | `/opt/kg-backend/.env` |
| Systemd service | `/etc/systemd/system/kg-backend.service` |
| Application logs | `/var/log/kg-backend/` |
| Nginx config | `/etc/nginx/sites-available/kg-backend` |

