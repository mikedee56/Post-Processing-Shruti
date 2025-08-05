#!/bin/bash
# Manual trigger for QA evaluation auto-push workflow
# Usage: ./scripts/auto-push-qa.sh

echo "🚀 Triggering QA evaluation auto-push workflow..."

# Run the post-QA hook
./.git/hooks/post-qa-eval

echo "✨ QA evaluation auto-push workflow completed!"