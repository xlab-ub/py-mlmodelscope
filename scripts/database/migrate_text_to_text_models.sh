#!/bin/bash

# ==========================================
# MLModelScope Text-to-Text Models Database Migration
# ==========================================
# Comprehensive setup script for text-to-text models across all supported frameworks
# Follows industry best practices for database migrations
# Version: 1.0.0
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
readonly SQL_FILE="${MIGRATION_DIR}/002_add_text_to_text_models.sql"
readonly LOG_DIR="${SCRIPT_DIR}/logs"
readonly LOG_FILE="${LOG_DIR}/text_to_text_migration_$(date +%Y%m%d_%H%M%S).log"

# Docker configuration
readonly COMPOSE_DIR="${SCRIPT_DIR}/../../../mlmodelscope-api"
readonly CONTAINER_SQL_PATH="/tmp/002_add_text_to_text_models.sql"

# Migration metadata
readonly MIGRATION_VERSION="002"
readonly MIGRATION_NAME="add_text_to_text_models"
readonly MIGRATION_DESCRIPTION="Add comprehensive text-to-text models for PyTorch, TensorFlow, and ONNX Runtime frameworks"

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
    WHERE m.output_type = 'text_to_text'
    GROUP BY f.name
    ORDER BY f.name;
    "
    
    log_info "Current text-to-text model counts by framework:"
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
    WHERE m.output_type = 'text_to_text'
    GROUP BY f.name, f.id
    ORDER BY f.id;
    "
    
    log_info "Text-to-text models by framework after migration:"
    if ! $DOCKER_COMPOSE_CMD exec -T db psql -U c3sr -d c3sr -c "$verification_sql" 2>/dev/null; then
        log_warning "Could not retrieve verification counts"
    fi
    
    # Get total count
    local total_count
    total_count=$($DOCKER_COMPOSE_CMD exec -T db psql -U c3sr -d c3sr -tAc "SELECT COUNT(*) FROM models WHERE output_type = 'text_to_text';" 2>/dev/null || echo "0")
    log_success "Total text-to-text models in database: $total_count"
    
    # Test a few sample models
    log_info "Testing sample model queries:"
    local sample_models=("gpt_2" "meta_llama_3_8b_instruct" "gemma_7b_it" "flan_t5_large")
    for model in "${sample_models[@]}"; do
        local model_exists
        model_exists=$($DOCKER_COMPOSE_CMD exec -T db psql -U c3sr -d c3sr -tAc "SELECT COUNT(*) FROM models WHERE name = '$model' AND output_type = 'text_to_text';" 2>/dev/null || echo "0")
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
            api_model_count=$(curl -s "http://localhost:8005/models?task=text_to_text" | grep -c '"name"' 2>/dev/null || echo "0")
            log_success "API responding: $api_model_count text-to-text models available via API"
            
            # Test a specific model
            if curl -s "http://localhost:8005/models?task=text_to_text" | grep -q "gpt_2" 2>/dev/null; then
                log_success "Sample model (gpt_2) available via API"
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
    log_info "  • PyTorch Framework: GPT-2, OPT, BLOOM, Llama 2/3, Gemma, Mistral, Phi-3, Qwen2.5, Falcon, Flan-T5, UnifiedQA, DeepSeek"
    log_info "  • TensorFlow Framework: GPT-2"
    log_info "  • ONNX Runtime Framework: BLOOM, GPT-2, Llama 3.2, Qwen2.5"
    echo ""
    log_success "Text-to-text models should now be visible in the MLModelScope UI"
    log_success "under the 'Text to Text' task section."
    echo ""
    log_info "Next Steps:"
    log_info "  1. Clear browser cache if models are not immediately visible"
    log_info "  2. Check browser console for any JavaScript errors"
    log_info "  3. Verify model files exist in py-mlmodelscope agent directories"
    log_info "  4. Test end-to-end functionality with sample text inputs"
    log_info "  5. Check model-specific requirements (e.g., authentication for gated models)"
    echo ""
    log_info "Note: JAX agent text-to-text models (gpt_2, distilgpt2) are available"
    log_info "but JAX framework is not registered in database. Register JAX framework"
    log_info "first if you want to include these models in a future migration."
}

print_model_categories() {
    log_header "Model Categories Summary"
    
    log_info "Text-to-Text Models by Category:"
    echo ""
    log_info "Foundation Models:"
    log_info "  • GPT Family: GPT-2, DistilGPT-2"
    log_info "  • OPT Family: OPT-125M, OPT-2.7B"
    log_info "  • BLOOM Family: BLOOM-560M, BLOOMZ-560M"
    echo ""
    log_info "Instruction-Tuned Models:"
    log_info "  • Llama Family: Llama 2 Chat, Llama 3 Instruct variants, specialized versions"
    log_info "  • Gemma Family: Gemma IT variants"
    log_info "  • Mistral Family: Mistral Instruct variants, BioMistral"
    log_info "  • Phi Family: Phi-3 Mini Instruct variants"
    log_info "  • Qwen Family: Qwen2.5 Instruct variants"
    log_info "  • Falcon Family: Falcon Instruct variants"
    echo ""
    log_info "Task-Specific Models:"
    log_info "  • T5 Family: Flan-T5 (Small, Base, Large)"
    log_info "  • QA Models: UnifiedQA variants"
    log_info "  • Reasoning Models: DeepSeek R1, ChatQA variants"
    echo ""
    log_info "Domain-Specific Models:"
    log_info "  • Medical: BioMistral 7B"
    log_info "  • Multilingual: BLOOM, Qwen2.5, Gemma variants"
    echo ""
    log_info "Size Categories:"
    log_info "  • Ultra-compact (≤1B): Qwen2.5 0.5B, Llama 3.2 1B variants"
    log_info "  • Compact (1-3B): Gemma 2B, OPT-2.7B"
    log_info "  • Medium (3-8B): Most 7B variants, Llama 3 8B"
    log_info "  • Large (8-15B): Llama 2 13B variants"
}

# ==========================================
# MAIN EXECUTION
# ==========================================

main() {
    log_header "MLModelScope Text-to-Text Models Migration v1.0"
    
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
    print_model_categories
    
    log_success "Migration completed successfully at $(date)"
}

# ==========================================
# SCRIPT ENTRY POINT
# ==========================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
