#!/bin/bash

# Production Deployment Script for ASR Post-Processing System
# Usage: ./deploy.sh [environment] [version]

set -euo pipefail

# Configuration
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
NAMESPACE="production"
IMAGE_REPOSITORY="asr-post-processor"
KUBECTL_TIMEOUT="300s"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check required tools
    command -v docker >/dev/null 2>&1 || log_error "Docker is required but not installed"
    command -v kubectl >/dev/null 2>&1 || log_error "kubectl is required but not installed"
    
    # Check kubectl connection
    kubectl cluster-info >/dev/null 2>&1 || log_error "Cannot connect to Kubernetes cluster"
    
    # Check namespace exists
    kubectl get namespace $NAMESPACE >/dev/null 2>&1 || {
        log_warn "Namespace $NAMESPACE does not exist, creating..."
        kubectl create namespace $NAMESPACE
    }
    
    log_info "Prerequisites validated"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    cd "$(dirname "$0")/../.."
    
    # Build image with version tag
    docker build \
        -f deploy/docker/Dockerfile \
        -t "${IMAGE_REPOSITORY}:${VERSION}" \
        -t "${IMAGE_REPOSITORY}:latest" \
        .
    
    log_info "Docker image built successfully"
}

# Push image to registry (if configured)
push_image() {
    if [[ -n "${DOCKER_REGISTRY:-}" ]]; then
        log_info "Pushing image to registry..."
        
        docker tag "${IMAGE_REPOSITORY}:${VERSION}" "${DOCKER_REGISTRY}/${IMAGE_REPOSITORY}:${VERSION}"
        docker push "${DOCKER_REGISTRY}/${IMAGE_REPOSITORY}:${VERSION}"
        
        log_info "Image pushed to registry"
    else
        log_warn "DOCKER_REGISTRY not set, skipping image push"
    fi
}

# Create secrets if they don't exist
create_secrets() {
    log_info "Creating Kubernetes secrets..."
    
    # Check if secrets exist
    if kubectl get secret app-secrets -n $NAMESPACE >/dev/null 2>&1; then
        log_warn "Secrets already exist, skipping creation"
        return
    fi
    
    # Create secrets from environment variables
    kubectl create secret generic app-secrets -n $NAMESPACE \
        --from-literal=DB_HOST="${DB_HOST:-postgres}" \
        --from-literal=DB_NAME="${DB_NAME:-asr_db}" \
        --from-literal=DB_USER="${DB_USER:-postgres}" \
        --from-literal=DB_PASSWORD="${DB_PASSWORD:-changeme}" \
        --from-literal=REDIS_HOST="${REDIS_HOST:-redis}" \
        --from-literal=REDIS_PASSWORD="${REDIS_PASSWORD:-changeme}" \
        --from-literal=JWT_SECRET="${JWT_SECRET:-changeme-jwt-secret}" \
        --from-literal=SLACK_WEBHOOK="${SLACK_WEBHOOK:-}" \
        --from-literal=BACKUP_BUCKET="${BACKUP_BUCKET:-asr-backups}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_info "Secrets created"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Update image tag in deployment
    sed "s|image: asr-post-processor:latest|image: ${IMAGE_REPOSITORY}:${VERSION}|g" \
        deploy/kubernetes/app-deployment.yaml | \
        kubectl apply -f -
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=$KUBECTL_TIMEOUT \
        deployment/asr-post-processor -n $NAMESPACE
    
    log_info "Deployment completed successfully"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Check pod status
    kubectl get pods -n $NAMESPACE -l app=asr-post-processor
    
    # Check service endpoints
    kubectl get endpoints -n $NAMESPACE asr-post-processor-service
    
    # Test health endpoint
    local pod_name=$(kubectl get pods -n $NAMESPACE -l app=asr-post-processor -o jsonpath='{.items[0].metadata.name}')
    if [[ -n "$pod_name" ]]; then
        kubectl exec $pod_name -n $NAMESPACE -- curl -f http://localhost:8080/health || {
            log_error "Health check failed"
        }
    fi
    
    log_info "Health checks passed"
}

# Rollback deployment
rollback_deployment() {
    log_warn "Rolling back deployment..."
    kubectl rollout undo deployment/asr-post-processor -n $NAMESPACE
    kubectl rollout status deployment/asr-post-processor -n $NAMESPACE
    log_info "Rollback completed"
}

# Main deployment function
main() {
    log_info "Starting deployment of ASR Post-Processing System"
    log_info "Environment: $ENVIRONMENT, Version: $VERSION"
    
    # Set trap for cleanup on error
    trap 'log_error "Deployment failed, consider running rollback"' ERR
    
    validate_prerequisites
    build_image
    push_image
    create_secrets
    deploy_to_kubernetes
    run_health_checks
    
    log_info "Deployment completed successfully!"
    log_info "Application is available at: http://$(kubectl get service asr-post-processor-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback_deployment
        ;;
    "health")
        run_health_checks
        ;;
    *)
        echo "Usage: $0 [deploy|rollback|health] [version]"
        exit 1
        ;;
esac