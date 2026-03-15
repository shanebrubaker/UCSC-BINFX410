"""
Unreliable API Simulator for Bioinformatics Workflows
======================================================

This Lambda function simulates an unreliable external API call, similar to what
you might encounter when calling:
- NCBI Entrez API (gene databases)
- UniProt REST API (protein sequences)
- UCSC Genome Browser API
- Third-party sequencing facility APIs

Educational Purpose:
- Demonstrates why retry logic is critical in bioinformatics pipelines
- Shows how Step Functions handles transient failures automatically
- Teaches exponential backoff patterns for API rate limiting

Author: AWS Educational Project
License: MIT
"""

import json
import random
import time
from datetime import datetime


def lambda_handler(event, context):
    """
    Main Lambda handler function that simulates an unreliable API call.

    Args:
        event (dict): Input from Step Functions containing:
            - attempt (int): Current attempt number (tracked by caller)
            - sample_id (str): Sample identifier (simulated bioinformatics data)
        context (object): AWS Lambda context object with runtime information

    Returns:
        dict: Success response with processing results

    Raises:
        Exception: Random failures to trigger Step Functions retry logic
    """

    # ========================================================================
    # 1. EXTRACT INPUT PARAMETERS
    # ========================================================================
    # Step Functions will pass data in the 'event' parameter
    attempt = event.get('attempt', 1)
    sample_id = event.get('sample_id', 'SAMPLE_UNKNOWN')

    # Generate unique execution ID for tracking across retries
    execution_id = context.request_id if context else 'local-test'

    # ========================================================================
    # 2. LOG ATTEMPT INFORMATION
    # ========================================================================
    # CloudWatch automatically captures all print statements as logs
    # These logs are critical for understanding retry behavior
    timestamp = datetime.utcnow().isoformat()

    print(f"{'='*70}")
    print(f"EXECUTION ATTEMPT #{attempt}")
    print(f"{'='*70}")
    print(f"Timestamp:     {timestamp}")
    print(f"Sample ID:     {sample_id}")
    print(f"Execution ID:  {execution_id}")
    print(f"Request ID:    {context.request_id if context else 'N/A'}")
    print(f"{'='*70}")

    # ========================================================================
    # 3. SIMULATE API PROCESSING TIME
    # ========================================================================
    # Real bioinformatics APIs take time to process requests
    # This simulates network latency and processing time
    processing_time = random.uniform(0.1, 0.5)  # 100-500ms
    print(f"Simulating API processing: {processing_time:.3f} seconds...")
    time.sleep(processing_time)

    # ========================================================================
    # 4. SIMULATE RANDOM FAILURES (60% failure rate)
    # ========================================================================
    # This failure rate demonstrates retry logic effectively
    # In production, you might see 5-20% failure rates for external APIs
    failure_probability = 0.60
    random_value = random.random()

    print(f"Failure check: {random_value:.3f} vs threshold {failure_probability:.3f}")

    if random_value < failure_probability:
        # ====================================================================
        # FAILURE CASE: Simulate common API errors
        # ====================================================================
        error_types = [
            "HTTP 503: Service Temporarily Unavailable",
            "HTTP 429: Rate Limit Exceeded",
            "HTTP 500: Internal Server Error",
            "ConnectionTimeout: Request timed out after 30s",
            "NetworkError: Unable to reach remote server"
        ]

        error_message = random.choice(error_types)

        print(f"❌ FAILURE SIMULATED")
        print(f"Error Type: {error_message}")
        print(f"Attempt {attempt} failed - Step Functions will retry with exponential backoff")
        print(f"{'='*70}\n")

        # Raise exception to trigger Step Functions retry logic
        raise Exception(f"{error_message} (Attempt {attempt})")

    # ========================================================================
    # 5. SUCCESS CASE: Return processed results
    # ========================================================================
    print(f"✓ SUCCESS")
    print(f"API call completed successfully on attempt #{attempt}")

    # Simulate bioinformatics API response
    # In real scenarios, this might be sequence data, alignment scores, etc.
    result = {
        'status': 'success',
        'sample_id': sample_id,
        'attempt_number': attempt,
        'execution_id': execution_id,
        'timestamp': timestamp,
        'processing_time_seconds': round(processing_time, 3),

        # Simulated bioinformatics results
        'results': {
            'sequence_length': random.randint(1000, 50000),
            'quality_score': round(random.uniform(30, 40), 2),
            'gc_content': round(random.uniform(0.40, 0.60), 3),
            'alignment_coverage': round(random.uniform(0.85, 0.99), 3)
        },

        # Metadata for learning
        'metadata': {
            'total_attempts': attempt,
            'api_endpoint': 'simulated-genomics-api.example.com',
            'region': 'us-east-1'
        }
    }

    print(f"Results: {json.dumps(result['results'], indent=2)}")
    print(f"{'='*70}\n")

    return result


# ============================================================================
# LOCAL TESTING
# ============================================================================
# This code allows you to test the Lambda function locally before deployment
if __name__ == "__main__":
    print("\n🧬 LOCAL TESTING: Unreliable API Simulator\n")

    # Mock Lambda context for local testing
    class MockContext:
        request_id = "local-test-" + str(random.randint(1000, 9999))
        function_name = "unreliable-api-simulator"
        memory_limit_in_mb = 128

    # Test with multiple attempts to see retry behavior
    for attempt_num in range(1, 6):
        test_event = {
            'attempt': attempt_num,
            'sample_id': 'SAMPLE_12345_RNA_SEQ'
        }

        try:
            result = lambda_handler(test_event, MockContext())
            print(f"✓ Attempt {attempt_num} succeeded!\n")
            break
        except Exception as e:
            print(f"✗ Attempt {attempt_num} failed: {e}\n")
            if attempt_num < 5:
                # Simulate exponential backoff (2^attempt seconds)
                wait_time = 2 ** attempt_num
                print(f"Waiting {wait_time} seconds before retry...\n")
                time.sleep(wait_time)
