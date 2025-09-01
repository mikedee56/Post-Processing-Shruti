# Production Operations Guide
## Sanskrit ASR Post-Processing System v0.1.2

### 📋 Overview

This guide provides comprehensive instructions for deploying, monitoring, and maintaining the Sanskrit ASR Post-Processing System in a production environment.

### 🚀 Quick Start Deployment

#### Prerequisites
- Docker Desktop with WSL 2 integration enabled
- Docker Compose v2.0+
- Minimum 8GB RAM, 4 CPU cores
- 50GB available disk space

#### Deployment Steps

1. **Clone and prepare the repository:**
   ```bash
   git clone <repository-url>
   cd Post-Processing-Shruti
   ```

2. **Run the automated deployment:**
   ```bash
   ./deploy_production.sh
   ```
   
   This script will:
   - Verify prerequisites
   - Load environment configuration  
   - Create required directories
   - Deploy all services via Docker Compose
   - Run health checks

3. **Verify deployment:**
   ```bash
   python3 scripts/health_check.py
   ```

### 🌐 Service Access

| Service | URL | Default Credentials |
|---------|-----|-------------------|
| Main Application | http://localhost:8000 | N/A |
| Expert Dashboard | http://localhost:3000 | OAuth required |
| Prometheus | http://localhost:9090 | N/A |
| Grafana | http://localhost:3001 | admin / admin123 |
| Jaeger | http://localhost:16686 | N/A |
| Airflow | http://localhost:8081 | admin / admin123 |

### 🏗️ Architecture Overview

The system consists of several interconnected services:

#### Core Services
- **sanskrit-processor**: Main ASR post-processing application
- **postgres**: Database with pgvector extension
- **redis**: Semantic embedding cache
- **expert-dashboard**: Human-in-the-loop validation interface

#### Processing Services
- **celery-worker**: Async task processing
- **celery-beat**: Periodic task scheduler  
- **airflow**: Batch processing orchestration

#### Monitoring Stack
- **prometheus**: Metrics collection
- **grafana**: Dashboards and visualization
- **jaeger**: Distributed tracing
- **nginx**: Load balancer and reverse proxy

### 🔧 Configuration Management

#### Environment Configuration
- **Development**: `.env`
- **Production**: `.env.production`
- **YAML Config**: `config/production.yml`

Key configuration parameters:
```yaml
processing:
  batch_size: 100
  max_workers: 4
  timeout: 300

academic:
  quality_threshold: 0.85
  transliteration_standard: "IAST"

features:
  enhanced_verse_identification: true
  wisdom_library_enabled: true
```

### 📊 Monitoring and Alerting

#### Health Monitoring
- **Endpoint**: http://localhost:8000/health
- **Script**: `python3 scripts/health_check.py`
- **Frequency**: Every 30 seconds via Prometheus

#### Key Metrics to Monitor
- **Application Performance**: Request latency, throughput
- **Processing Quality**: Academic scores, verse identification accuracy
- **Resource Usage**: CPU, memory, disk space
- **Database Health**: Connection pool, query performance
- **Cache Performance**: Redis hit rates, memory usage

#### Grafana Dashboards
Access Grafana at http://localhost:3001 with admin/admin123:
- **System Overview**: General health and performance
- **Processing Metrics**: Academic quality scores
- **Infrastructure**: Resource utilization
- **Alerting**: Critical system alerts

### 🔍 Troubleshooting

#### Common Issues

1. **Docker Desktop WSL Integration**
   ```bash
   # Enable WSL integration in Docker Desktop settings
   # Or use Windows Docker path directly
   "/mnt/c/Program Files/Docker/Docker/resources/bin/docker" --version
   ```

2. **Memory Issues**
   ```bash
   # Check container resource usage
   docker stats
   
   # Increase memory limits in docker-compose.production.yml
   deploy:
     resources:
       limits:
         memory: 4G
   ```

3. **Database Connection Issues**
   ```bash
   # Check PostgreSQL logs
   docker logs sanskrit-postgres
   
   # Test connection
   docker exec sanskrit-postgres pg_isready -U sanskrit_user -d sanskrit_db
   ```

4. **Optional Dependencies**
   The system gracefully handles missing optional dependencies:
   - `sanskrit_parser`: Falls back to basic tokenization
   - `beautifulsoup4`: Uses regex parsing for Wisdom Library
   - `inltk`: Uses fallback similarity implementation

#### Log Locations
- **Application logs**: `logs/sanskrit-processor.log`
- **Container logs**: `docker logs <container-name>`
- **Nginx logs**: Mounted volume `nginx_logs`
- **Airflow logs**: Mounted volume `airflow_logs`

### 🔒 Security Considerations

#### Network Security
- All services run in isolated Docker network
- Nginx reverse proxy handles external traffic
- Rate limiting configured for API endpoints

#### Data Protection
- Database passwords in environment variables
- No sensitive data in container images
- Secure defaults for all services

#### Access Control
- OAuth integration for expert dashboard
- Admin credentials for monitoring tools
- Role-based access for Airflow workflows

### 🚀 Performance Tuning

#### Application Level
```yaml
performance:
  max_workers: 4          # Adjust based on CPU cores
  batch_size: 100         # Balance memory vs throughput
  cache_ttl: 86400       # Cache expiration in seconds
```

#### Database Optimization
```sql
-- Index optimization for scripture lookups
CREATE INDEX idx_verses_text ON verses USING gin(to_tsvector('english', text));
CREATE INDEX idx_verses_embedding ON verses USING ivfflat (embedding vector_cosine_ops);
```

#### Cache Configuration
```yaml
redis:
  maxmemory: 2gb
  maxmemory-policy: allkeys-lru
  timeout: 10
```

### 📈 Scaling Guidelines

#### Horizontal Scaling
- Increase Celery worker replicas for async processing
- Scale application containers behind Nginx load balancer
- Use PostgreSQL read replicas for query scaling

#### Vertical Scaling
- Monitor resource usage via Grafana
- Adjust container resource limits
- Optimize JVM/Python memory settings

### 🔄 Maintenance Procedures

#### Regular Maintenance
1. **Daily**: Check health monitors, review error logs
2. **Weekly**: Database maintenance, cache cleanup
3. **Monthly**: Security updates, dependency updates
4. **Quarterly**: Performance review, capacity planning

#### Backup Strategy
```bash
# Database backup
docker exec sanskrit-postgres pg_dump -U sanskrit_user sanskrit_db > backup.sql

# Application data backup
docker cp sanskrit-app:/app/data ./backup-data/

# Configuration backup
cp -r config/ .env.production ./backup-config/
```

#### Update Procedure
1. Test updates in staging environment
2. Create system backup
3. Deploy with rolling updates via Docker Compose
4. Verify health checks pass
5. Monitor system performance

### 🆘 Emergency Procedures

#### System Recovery
```bash
# Stop all services
docker-compose -f docker-compose.production.yml down

# Start core services only
docker-compose -f docker-compose.production.yml up -d postgres redis sanskrit-processor

# Verify core functionality
python3 scripts/health_check.py
```

#### Data Recovery
```bash
# Restore database
docker exec -i sanskrit-postgres psql -U sanskrit_user sanskrit_db < backup.sql

# Restore application data
docker cp ./backup-data/ sanskrit-app:/app/data
```

### 📞 Support and Contacts

For technical support:
- Check logs first: `docker logs <service-name>`
- Run diagnostics: `python3 scripts/health_check.py`
- Monitor dashboards: http://localhost:3001
- Review traces: http://localhost:16686

### 📝 Change Log

- **v0.1.2**: Production deployment with Epic 4 features
- **v0.1.1**: Enhanced verse identification with Wisdom Library
- **v0.1.0**: Initial production release

---

**Note**: This system emphasizes Wisdom Library (wisdomlib.org) as the single most effective resource for Sanskrit scripture processing, as specifically requested in the production requirements.