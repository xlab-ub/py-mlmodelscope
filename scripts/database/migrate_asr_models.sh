#!/bin/bash

# ==========================================
# MLModelScope ASR Models Database Migration
# ==========================================
# Comprehensive setup script for ASR models across all supported frameworks
# Follows industry best practices for database migrations
# Version: 2.0.0
# Author: MLModelScope Team

set -euo pipefail  # Strict error handling

# ==========================================
# CONFIGURATION
# ==========================================

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m' # No Color

# Paths
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
readonly MIGRATION_DIR="${SCRIPT_DIR}/migrations"
readonly SQL_FILE="${MIGRATION_DIR}/001_add_asr_models.sql"
readonly LOG_DIR="${SCRIPT_DIR}/logs"
readonly LOG_FILE="${LOG_DIR}/migration_$(date +%Y%m%d_%H%M%S).log"

# Docker configuration
readonly COMPOSE_DIR="${SCRIPT_DIR}/../../../mlmodelscope-api"
readonly CONTAINER_SQL_PATH="/tmp/001_add_asr_models.sql"

# Migration metadata
readonly MIGRATION_VERSION="001"
readonly MIGRATION_NAME="add_asr_models"
readonly MIGRATION_DESCRIPTION="Add comprehensive ASR models for PyTorch, TensorFlow, and JAX frameworks"

# ==========================================
# LOGGING FUNCTIONS
# ==========================================

setup_logging() {
    mkdir -p "$LOG_DIR"
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
}

log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo ""
    echo -e "${CYAN}${BOLD}========================================${NC}"
    echo -e "${CYAN}${BOLD}$1${NC}"
    echo -e "${CYAN}${BOLD}========================================${NC}"
    echo ""
}

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

cleanup_on_exit() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Script failed with exit code $exit_code"
        log_info "Check log file: $LOG_FILE"
    fi
    
    # Clean up temporary files
    if [[ -n "${DOCKER_COMPOSE_CMD:-}" ]]; then
        $DOCKER_COMPOSE_CMD exec -T db rm -f "$CONTAINER_SQL_PATH" 2>/dev/null || true
    fi
    
    exit $exit_code
}

trap cleanup_on_exit EXIT

check_prerequisites() {
    log_header "Checking Prerequisites"
    
    # Check if SQL migration file exists
    if [[ ! -f "$SQL_FILE" ]]; then
        log_error "Migration file not found: $SQL_FILE"
        exit 1
    fi
    log_success "Migration file found: $SQL_FILE"
    
    # Check if docker-compose.yml exists
    if [[ ! -f "$COMPOSE_DIR/docker-compose.yml" ]]; then
        log_error "docker-compose.yml not found: $COMPOSE_DIR"
        exit 1
    fi
    log_success "Docker Compose configuration found"
    
    # Detect Docker Compose command
    if docker compose version &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker compose"
    elif command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker-compose"
    else
        log_error "Neither 'docker compose' nor 'docker-compose' command found"
        exit 1
    fi
    log_success "Docker Compose command detected: $DOCKER_COMPOSE_CMD"
}

check_database_status() {
    log_header "Checking Database Status"
    
    cd "$COMPOSE_DIR"
    
    # Check if database container is running
    if ! $DOCKER_COMPOSE_CMD ps db | grep -q "Up"; then
        log_error "Database container is not running"
        log_info "Try starting services with: $DOCKER_COMPOSE_CMD up -d"
        exit 1
    fi
    log_success "Database container is running"
    
    # Test database connectivity
    if ! $DOCKER_COMPOSE_CMD exec -T db pg_isready -U c3sr -d c3sr &> /dev/null; then
        log_error "Database is not ready to accept connections"
        exit 1
    fi
    log_success "Database is ready"
}

create_migration_table() {
    log_header "Setting Up Migration Tracking"
    
    # Create migrations table if it doesn't exist
    local migration_table_sql="
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version VARCHAR(255) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        applied_by VARCHAR(255) DEFAULT 'migration_script'
    );
    "
    
    if $DOCKER_COMPOSE_CMD exec -T db psql -U c3sr -d c3sr -c "$migration_table_sql" &> /dev/null; then
        log_success "Migration tracking table ready"
    else
        log_error "Failed to create migration tracking table"
        exit 1
    fi
}

check_migration_status() {
    log_header "Checking Migration Status"
    
    # Check if this migration has already been applied
    local check_sql="SELECT version FROM schema_migrations WHERE version = '$MIGRATION_VERSION';"
    local migration_exists
    migration_exists=$($DOCKER_COMPOSE_CMD exec -T db psql -U c3sr -d c3sr -tAc "$check_sql" 2>/dev/null | tr -d '[:space:]')
    
    if [[ "$migration_exists" == "$MIGRATION_VERSION" ]]; then
        log_warning "Migration $MIGRATION_VERSION has already been applied"
        read -p "Do you want to reapply this migration? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Migration skipped by user choice"
            exit 0
        fi
        
        # Remove existing migration record
        local remove_sql="DELETE FROM schema_migrations WHERE version = '$MIGRATION_VERSION';"
        $DOCKER_COMPOSE_CMD exec -T db psql -U c3sr -d c3sr -c "$remove_sql" &> /dev/null
        log_info "Existing migration record removed"
    fi
}

get_current_model_counts() {
    log_header "Current Model Status"
    
    # Get current model counts by framework
    local count_sql="
    SELECT 
        COALESCE(f.name, 'Unknown') as framework,
        COUNT(*) as count
    FROM models m
    LEFT JOIN frameworks f ON m.framework_id = f.id
    WHERE m.output_type IN ('automatic_speech_recognition', 'audio_to_text')
    GROUP BY f.name
    ORDER BY f.name;
    "
    
    log_info "Current ASR model counts by framework:"
    $DOCKER_COMPOSE_CMD exec -T db psql -U c3sr -d c3sr -c "$count_sql" 2>/dev/null || log_warning "Could not retrieve current model counts"
}

apply_migration() {
    log_header "Applying Migration $MIGRATION_VERSION"
    
    # Copy SQL file to container
    log_info "Copying migration file to database container..."
    if ! $DOCKER_COMPOSE_CMD cp "$SQL_FILE" db:"$CONTAINER_SQL_PATH" 2>/tmp/copy_error.log; then
        log_warning "Copy with ownership failed, trying alternative method..."
        if $DOCKER_COMPOSE_CMD exec -T db sh -c "cat > $CONTAINER_SQL_PATH" < "$SQL_FILE"; then
            log_success "Migration file copied using alternative method"
        else
            log_error "Failed to copy migration file to container"
            exit 1
        fi
    else
        log_success "Migration file copied successfully"
    fi
    
    # Apply migration
    log_info "Executing migration SQL..."
    if $DOCKER_COMPOSE_CMD exec -T db psql -U c3sr -d c3sr -f "$CONTAINER_SQL_PATH"; then
        log_success "Migration SQL executed successfully"
    else
        log_error "Migration execution failed"
        exit 1
    fi
    
    # Record migration in tracking table
    local record_sql="
    INSERT INTO schema_migrations (version, name, description) 
    VALUES ('$MIGRATION_VERSION', '$MIGRATION_NAME', '$MIGRATION_DESCRIPTION');
    "
    
    if $DOCKER_COMPOSE_CMD exec -T db psql -U c3sr -d c3sr -c "$record_sql" &> /dev/null; then
        log_success "Migration recorded in tracking table"
    else
        log_warning "Failed to record migration in tracking table"
    fi
}

verify_migration() {
    log_header "Verifying Migration Results"
    
    # Get new model counts by framework
    local verification_sql="
    SELECT 
        f.name as framework,
        COUNT(*) as model_count
    FROM models m
    JOIN frameworks f ON m.framework_id = f.id
    WHERE m.output_type = 'audio_to_text'
    GROUP BY f.name, f.id
    ORDER BY f.id;
    "
    
    log_info "ASR models by framework after migration:"
    if ! $DOCKER_COMPOSE_CMD exec -T db psql -U c3sr -d c3sr -c "$verification_sql" 2>/dev/null; then
        log_warning "Could not retrieve verification counts"
    fi
    
    # Get total count
    local total_count
    total_count=$($DOCKER_COMPOSE_CMD exec -T db psql -U c3sr -d c3sr -tAc "SELECT COUNT(*) FROM models WHERE output_type = 'audio_to_text';" 2>/dev/null || echo "0")
    log_success "Total ASR models in database: $total_count"
    
    # Test a few sample models
    log_info "Testing sample model queries:"
    local sample_models=("whisper_base_en" "whisper_large_v3" "wav2vec2_base_960h")
    for model in "${sample_models[@]}"; do
        local model_exists
        model_exists=$($DOCKER_COMPOSE_CMD exec -T db psql -U c3sr -d c3sr -tAc "SELECT COUNT(*) FROM models WHERE name = '$model' AND output_type = 'audio_to_text';" 2>/dev/null || echo "0")
        if [[ "$model_exists" -gt 0 ]]; then
            log_success "  ✓ $model found"
        else
            log_warning "  ✗ $model not found"
        fi
    done
}

restart_services() {
    log_header "Restarting Services"
    
    # Restart API service to refresh model cache
    log_info "Restarting API service..."
    if $DOCKER_COMPOSE_CMD restart api; then
        log_success "API service restarted"
    else
        log_warning "Failed to restart API service"
    fi
    
    # Wait for API to be ready
    log_info "Waiting for API service to be ready..."
    sleep 5
}

test_api_endpoints() {
    log_header "Testing API Endpoints"
    
    if command -v curl &> /dev/null; then
        # Test if Python API is accessible
        if curl -s "http://localhost:8005/" &> /dev/null; then
            local api_model_count
            api_model_count=$(curl -s "http://localhost:8005/models?task=audio_to_text" | grep -c '"name"' 2>/dev/null || echo "0")
            log_success "API responding: $api_model_count ASR models available via API"
            
            # Test a specific model
            if curl -s "http://localhost:8005/models?task=audio_to_text" | grep -q "whisper_base_en" 2>/dev/null; then
                log_success "Sample model (whisper_base_en) available via API"
            else
                log_warning "Sample model not found in API response"
            fi
        else
            log_warning "API endpoint not accessible on localhost:8005"
        fi
    else
        log_warning "curl not available, skipping API endpoint tests"
    fi
}

print_summary() {
    log_header "Migration Summary"
    
    log_success "Migration $MIGRATION_VERSION completed successfully!"
    echo ""
    log_info "Migration Details:"
    log_info "  • Version: $MIGRATION_VERSION"
    log_info "  • Name: $MIGRATION_NAME"
    log_info "  • Description: $MIGRATION_DESCRIPTION"
    log_info "  • Log file: $LOG_FILE"
    echo ""
    log_info "Models Added:"
    log_info "  • PyTorch Framework: Whisper variants, Wav2Vec2, HuBERT, SeamlessM4T, WhisperX, CrisperWhisper"
    log_info "  • TensorFlow Framework: Whisper variants, Wav2Vec2"
    log_info "  • JAX Framework: Whisper variants, Wav2Vec2 (if JAX framework exists)"
    echo ""
    log_success "ASR models should now be visible in the MLModelScope UI"
    log_success "under the 'Audio to Text' task section."
    echo ""
    log_info "Next Steps:"
    log_info "  1. Clear browser cache if models are not immediately visible"
    log_info "  2. Check browser console for any JavaScript errors"
    log_info "  3. Verify model files exist in py-mlmodelscope agent directories"
    log_info "  4. Test end-to-end functionality with sample audio files"
}

# ==========================================
# MAIN EXECUTION
# ==========================================

main() {
    log_header "MLModelScope ASR Models Migration v2.0"
    
    setup_logging
    log_info "Starting migration at $(date)"
    log_info "Migration version: $MIGRATION_VERSION"
    log_info "Migration name: $MIGRATION_NAME"
    
    check_prerequisites
    check_database_status
    create_migration_table
    check_migration_status
    get_current_model_counts
    apply_migration
    verify_migration
    restart_services
    test_api_endpoints
    print_summary
    
    log_success "Migration completed successfully at $(date)"
}

# ==========================================
# SCRIPT ENTRY POINT
# ==========================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
