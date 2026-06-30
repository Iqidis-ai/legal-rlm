# Irys RLM - Production Deployment Guide

Deploy Irys RLM as a lightweight API service on EC2 with S3 document storage.

---

## Pre-Deployment Checklist

Before deploying, ensure you have:

- [ ] **Google Gemini API Key** - Get from https://aistudio.google.com/app/apikey
- [ ] **AWS Account** with permissions to create EC2, S3, IAM
- [ ] **S3 Bucket** created for document storage
- [ ] **EC2 Key Pair** for SSH access
- [ ] **Security Group** configured (ports 22, 8000)

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Your Chat      │────▶│  Irys RLM API   │────▶│  Google Gemini  │
│  Service        │     │  (EC2)          │     │  API            │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  S3 Bucket      │
                        │  (Documents)    │
                        └─────────────────┘
```

## Quick Start (Local Testing)

```bash
# 1. Install dependencies
pip install -e .

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Run server
python run_server.py

# 4. Test API
curl http://localhost:8000/health
```

## AWS Setup (One-Time)

### Step 1: Create S3 Bucket

```bash
# Create bucket
aws s3 mb s3://your-company-irys-documents --region us-east-1

# Create folder structure
aws s3api put-object --bucket your-company-irys-documents --key matters/
```

### Step 2: Create IAM Role for EC2

Create a role named `IrysEC2Role` with this policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket",
        "s3:HeadObject"
      ],
      "Resource": [
        "arn:aws:s3:::your-company-irys-documents",
        "arn:aws:s3:::your-company-irys-documents/*"
      ]
    }
  ]
}
```

**To create via AWS Console:**
1. Go to IAM → Roles → Create Role
2. Select "AWS Service" → "EC2"
3. Create policy with above JSON
4. Name it `IrysS3ReadPolicy`
5. Attach to role `IrysEC2Role`

### Step 3: Create Security Group

Create security group `irys-api-sg`:

| Type | Port | Source | Description |
|------|------|--------|-------------|
| SSH | 22 | Your IP | SSH access |
| HTTP | 80 | 0.0.0.0/0 | Redirect to HTTPS |
| HTTPS | 443 | 0.0.0.0/0 | API access (SSL) |

---

## EC2 Deployment

### Recommended Instance

| Instance | vCPU | RAM | Use Case |
|----------|------|-----|----------|
| t3.micro | 1 | 1 GB | Testing, low traffic |
| t3.small | 2 | 2 GB | **Recommended** for production |
| t3.medium | 2 | 4 GB | High traffic, many concurrent jobs |

**Configuration:**
- AMI: Amazon Linux 2023 or Ubuntu 22.04
- Storage: 20 GB gp3
- IAM Role: `IrysEC2Role` (created above)
- Security Group: `irys-api-sg`

### Step 1: Launch and Connect

```bash
# Launch instance via AWS Console with above settings

# S SH into your instance
ssh -i your-key.pem ec2-user@your-instance-ip
```

### Step 2: Install Docker

```bash
# Amazon Linux 2023
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Log out and back in for group changes
```

### Step 3: Deploy with Docker

```bash
# Clone repository
git clone https://github.com/your-repo/legal-rlm.git
cd legal-rlm

# Create .env file
cat > .env << 'EOF'
GEMINI_API_KEY=your_api_key_here
S3_BUCKET=your-bucket-name
S3_REGION=us-east-1
IRYS_LOG_LEVEL=INFO
EOF

# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f

# Check health
curl http://localhost:8000/health
```

### Step 4: Configure Security Group

Allow inbound traffic:
- **Port 80** from your chat service IP (nginx reverse proxy)
- **Port 22** for SSH (your IP only)

## API Usage

### From Your Chat Service

```python
import httpx

IRYS_API = "http://your-ec2-elastic-ip"  # No port needed with nginx

# Start investigation
async def investigate_documents(query: str, s3_prefix: str):
    async with httpx.AsyncClient() as client:
        # Start job
        response = await client.post(
            f"{IRYS_API}/investigate",
            json={
                "query": query,
                "s3_prefix": s3_prefix,
                "callback_url": "https://your-chat-service/webhook"
            }
        )
        job = response.json()
        return job["job_id"]

# Poll for results
async def get_results(job_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{IRYS_API}/investigate/{job_id}")
        return response.json()
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/investigate` | Start investigation |
| `GET` | `/investigate/{job_id}` | Get results |
| `GET` | `/jobs` | List all jobs |
| `POST` | `/search` | Quick keyword search |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | OpenAPI docs |

### Example: Start Investigation

```bash
curl -X POST http://your-elastic-ip/investigate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the payment terms?",
    "s3_prefix": "documents/contracts/acme"
  }'
```

Response:
```json
{
  "job_id": "inv_abc123def456",
  "status": "pending",
  "message": "Investigation started",
  "estimated_seconds": 60
}
```

### Example: Get Results

```bash
curl http://your-elastic-ip/investigate/inv_abc123def456
```

Response (when complete):
```json
{
  "job_id": "inv_abc123def456",
  "status": "completed",
  "query": "What are the payment terms?",
  "analysis": "Based on the contract documents...",
  "citations": [...],
  "entities": {"companies": [...], "people": [...]},
  "documents_processed": 5,
  "duration_seconds": 45.2
}
```

## S3 Document Structure

Organize your documents in S3 by matter/case:

```
your-bucket/
├── documents/
│   ├── contracts/
│   │   ├── acme-deal/
│   │   │   ├── main_contract.pdf
│   │   │   ├── amendment_1.pdf
│   │   │   └── side_letter.docx
│   │   └── beta-acquisition/
│   │       └── ...
│   └── litigation/
│       └── ...
```

When calling the API, use the prefix:
```json
{"s3_prefix": "documents/contracts/acme-deal"}
```

## Resource Tuning

### For t3.micro (1 GB RAM)

```env
IRYS_MAX_CONCURRENT_JOBS=1
IRYS_MAX_DOCS_PER_JOB=20
IRYS_MAX_TEMP_SIZE_MB=300
```

### For t3.small (2 GB RAM)

```env
IRYS_MAX_CONCURRENT_JOBS=3
IRYS_MAX_DOCS_PER_JOB=50
IRYS_MAX_TEMP_SIZE_MB=500
```

### For t3.medium (4 GB RAM)

```env
IRYS_MAX_CONCURRENT_JOBS=5
IRYS_MAX_DOCS_PER_JOB=100
IRYS_MAX_TEMP_SIZE_MB=1000
```

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Returns:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "gemini_connected": true,
  "s3_connected": true,
  "active_jobs": 2,
  "temp_storage_mb": 125.5,
  "uptime_seconds": 86400
}
```

### CloudWatch Integration

Add to docker-compose.yml for CloudWatch logging:

```yaml
logging:
  driver: awslogs
  options:
    awslogs-group: /ecs/irys-rlm
    awslogs-region: us-east-1
    awslogs-stream-prefix: irys
```

## Webhooks

Instead of polling, use webhooks for async results:

```python
# When starting investigation
response = await client.post(
    f"{IRYS_API}/investigate",
    json={
        "query": "What are the payment terms?",
        "s3_prefix": "documents/contracts/acme",
        "callback_url": "https://your-service.com/webhook/investigation"
    }
)
```

Irys will POST results to your callback URL when complete.

## Troubleshooting

### Check Logs

```bash
docker-compose logs -f irys-api
```

### Common Issues

| Issue | Solution |
|-------|----------|
| S3 access denied | Check IAM role permissions |
| Gemini 401/403 | Verify API key is valid |
| Out of memory | Reduce `IRYS_MAX_CONCURRENT_JOBS` |
| Temp storage full | Reduce `IRYS_CLEANUP_SECONDS` |

### Test S3 Access

```bash
aws s3 ls s3://your-bucket/documents/
```

## Cost Optimization

- Use **Spot Instances** for non-critical workloads
- Use **IAM roles** instead of access keys (more secure, no key rotation)
- Set aggressive **cleanup** to minimize EBS usage
- Use **S3 Intelligent-Tiering** for documents

---

## Complete Deployment Script

Copy and run this script on a fresh EC2 instance:

```bash
#!/bin/bash
# deploy-irys.sh - Complete Irys RLM deployment script

set -e

echo "=========================================="
echo "Irys RLM Deployment Script"
echo "=========================================="

# Variables - EDIT THESE
GEMINI_API_KEY="your-gemini-api-key-here"
S3_BUCKET="your-bucket-name"
S3_REGION="us-east-1"
REPO_URL="https://github.com/your-org/legal-rlm.git"

# Install Docker
echo "Installing Docker..."
sudo yum update -y
sudo yum install -y docker git
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Clone repository
echo "Cloning repository..."
cd /home/ec2-user
git clone $REPO_URL irys-rlm
cd irys-rlm

# Create environment file
echo "Creating .env file..."
cat > .env << EOF
GEMINI_API_KEY=${GEMINI_API_KEY}
S3_BUCKET=${S3_BUCKET}
S3_REGION=${S3_REGION}
IRYS_LOG_LEVEL=INFO
IRYS_MAX_CONCURRENT_JOBS=3
IRYS_MAX_DOCS_PER_JOB=50
EOF

# Build and start
echo "Building and starting service..."
sudo docker-compose up -d --build

# Wait for startup
echo "Waiting for service to start..."
sleep 10

# Health check
echo "Checking health..."
curl -s http://localhost:8000/health | python3 -m json.tool

echo "=========================================="
echo "Deployment complete!"
echo "API available at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
echo "=========================================="
```

Save as `deploy-irys.sh`, edit the variables, then run:
```bash
chmod +x deploy-irys.sh
./deploy-irys.sh
```

---

## Updating the Service

```bash
cd /home/ec2-user/irys-rlm

# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up -d --build

# Check logs
docker-compose logs -f
```

---

## Summary

| Step | Action |
|------|--------|
| 1 | Create S3 bucket |
| 2 | Create IAM role with S3 read access |
| 3 | Launch EC2 with IAM role attached |
| 4 | SSH in and run deployment script |
| 5 | Test health endpoint |
| 6 | Configure chat service to call API |

**Your API will be at:** `https://rlm.iryslegal.com`

**Interactive docs at:** `https://rlm.iryslegal.com/docs`

---

## HTTPS Setup (Let's Encrypt)

### Prerequisites
- Domain pointing to your Elastic IP (A record)
- Ports 80 and 443 open in security group

### Step 1: Get SSL Certificate

```bash
cd /home/ubuntu/legal-rlm

# Edit init-ssl.sh to set your email
nano init-ssl.sh  # Change EMAIL="admin@irys.ai" to your email

# Run the SSL initialization script
./init-ssl.sh
```

### Step 2: Start Services

```bash
# Create .env file if not already done
cat > .env << 'EOF'
GEMINI_API_KEY=your_gemini_key
S3_BUCKET=your-bucket-name
S3_REGION=us-east-1
EOF

# Start all services
docker-compose up -d --build

# Verify
curl https://rlm.iryslegal.com/health
```

### Certificate Auto-Renewal

Certificates auto-renew via the certbot container. It checks every 12 hours and renews if expiring within 30 days.

To manually renew:
```bash
docker-compose run --rm certbot renew
docker-compose exec nginx nginx -s reload
```
