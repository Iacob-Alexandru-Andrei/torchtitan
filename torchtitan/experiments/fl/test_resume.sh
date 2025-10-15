#!/bin/bash
# Test script to verify resume functionality

echo "=== Resume Functionality Test ==="
echo ""

echo "1. Checking if source checkpoint exists on S3..."
export S3_ENDPOINT_URL='http://taranaki.cl.cam.ac.uk:9000'
aws s3 ls s3://checkpoints/torchtitan/16M-baseline-20251011-122516/step-10/ --endpoint-url=$S3_ENDPOINT_URL || echo "ERROR: Source checkpoint not found on S3!"
echo ""

echo "2. Cleaning local checkpoints directory..."
rm -rf checkpoints/
mkdir -p checkpoints/
echo "✓ Local checkpoints directory cleaned"
echo ""

echo "3. Checking run configuration..."
echo "RESUME_FROM_RUN_STEP=${RESUME_FROM_RUN_STEP}"
echo "RUN_UUID=${RUN_UUID}"
echo ""

echo "4. Running training with resume..."
echo "Look for these debug markers in the logs:"
echo "  - '✓ Checkpoint download complete for step 10'"
echo "  - '[RESUME DEBUG] Before checkpoint load: self.step = 0'"
echo "  - '[RESUME DEBUG] States to load keys: [...]'"
echo "  - '[RESUME DEBUG] Trainer.load_state_dict called with: {...}'"
echo "  - '[RESUME DEBUG] After load: self.step = 10'"
echo "  - 'Training starts at step 11'"
echo ""
echo "Press Ctrl+C to stop when you see the training start message..."
echo ""
