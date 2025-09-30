#!/bin/bash
# ENBEL Docker Entrypoint Script
# ===============================
# Configurable entrypoint for ENBEL climate-health analysis container

set -e

# Default values
CONFIG_FILE="${CONFIG_FILE:-configs/default.yaml}"
ANALYSIS_MODE="${ANALYSIS_MODE:-improved}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Function to print usage
print_usage() {
    echo "ENBEL Climate-Health Analysis Container"
    echo "======================================="
    echo ""
    echo "Usage: docker run [OPTIONS] enbel/climate-health-analysis [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  pipeline         Run the complete analysis pipeline (default)"
    echo "  simple           Run simple analysis mode"
    echo "  improved         Run improved analysis mode"
    echo "  state-of-the-art Run state-of-the-art analysis mode"
    echo "  test             Run test suite"
    echo "  config           Validate configuration"
    echo "  shell            Start interactive shell"
    echo "  jupyter          Start Jupyter notebook server"
    echo "  help             Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  CONFIG_FILE      Configuration file path (default: configs/default.yaml)"
    echo "  ANALYSIS_MODE    Analysis mode (simple/improved/state-of-the-art)"
    echo "  LOG_LEVEL        Logging level (DEBUG/INFO/WARNING/ERROR)"
    echo "  DATA_PATH        Path to data directory"
    echo "  RESULTS_PATH     Path to results directory"
    echo ""
    echo "Examples:"
    echo "  # Run improved analysis"
    echo "  docker run -v /data:/app/data enbel/climate-health-analysis improved"
    echo ""
    echo "  # Run with custom config"
    echo "  docker run -e CONFIG_FILE=configs/production.yaml enbel/climate-health-analysis"
    echo ""
    echo "  # Start Jupyter notebook"
    echo "  docker run -p 8888:8888 enbel/climate-health-analysis jupyter"
}

# Function to setup environment
setup_environment() {
    echo "Setting up ENBEL environment..."
    
    # Set Python path
    export PYTHONPATH="/app/src:$PYTHONPATH"
    
    # Create output directories if they don't exist
    mkdir -p results models figures logs cache
    
    # Set file permissions
    chmod -R 755 results models figures logs cache 2>/dev/null || true
    
    echo "Environment setup complete."
}

# Function to validate configuration
validate_config() {
    echo "Validating configuration..."
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Configuration file not found: $CONFIG_FILE"
        echo "Available configs:"
        ls -la configs/ 2>/dev/null || echo "No configs directory found"
        exit 1
    fi
    
    python -c "
import sys
sys.path.insert(0, '/app/src')
from enbel_pp.config import ENBELConfig
try:
    config = ENBELConfig('$CONFIG_FILE')
    print('Configuration valid ✓')
except Exception as e:
    print(f'Configuration error: {e}')
    sys.exit(1)
"
}

# Function to run tests
run_tests() {
    echo "Running ENBEL test suite..."
    cd /app
    python -m pytest tests/ -v --tb=short
}

# Function to run pipeline
run_pipeline() {
    local mode="$1"
    echo "Running ENBEL analysis pipeline in $mode mode..."
    
    python -c "
import sys
sys.path.insert(0, '/app/src')
from enbel_pp.pipeline import ClimateHealthPipeline

try:
    pipeline = ClimateHealthPipeline(
        analysis_mode='$mode',
        config_file='$CONFIG_FILE'
    )
    pipeline.load_data()
    results = pipeline.run_comprehensive_analysis()
    print(pipeline.generate_summary_report())
    print('Analysis completed successfully ✓')
except Exception as e:
    print(f'Analysis failed: {e}')
    sys.exit(1)
"
}

# Function to start Jupyter notebook
start_jupyter() {
    echo "Starting Jupyter notebook server..."
    
    # Install jupyter if not present
    pip list | grep jupyter >/dev/null || pip install jupyter
    
    # Start notebook server
    jupyter notebook \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --notebook-dir=/app \
        --NotebookApp.token='' \
        --NotebookApp.password=''
}

# Function to start interactive shell
start_shell() {
    echo "Starting interactive shell..."
    echo "ENBEL environment is ready. Python path: $PYTHONPATH"
    echo "Try: python -c 'from enbel_pp import ClimateHealthPipeline; print(\"ENBEL loaded successfully\")'"
    /bin/bash
}

# Main execution
main() {
    echo "ENBEL Climate-Health Analysis Container"
    echo "======================================="
    
    # Setup environment
    setup_environment
    
    # Parse command
    case "${1:-pipeline}" in
        "pipeline")
            validate_config
            run_pipeline "$ANALYSIS_MODE"
            ;;
        "simple")
            validate_config
            run_pipeline "simple"
            ;;
        "improved")
            validate_config
            run_pipeline "improved"
            ;;
        "state-of-the-art")
            validate_config
            run_pipeline "state_of_the_art"
            ;;
        "test")
            run_tests
            ;;
        "config")
            validate_config
            echo "Configuration validation completed ✓"
            ;;
        "jupyter")
            start_jupyter
            ;;
        "shell")
            start_shell
            ;;
        "help"|"--help"|"-h")
            print_usage
            ;;
        *)
            echo "Unknown command: $1"
            echo "Use 'help' to see available commands."
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"