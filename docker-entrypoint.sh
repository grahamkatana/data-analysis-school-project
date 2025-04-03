#!/bin/bash
set -e

# Initialize the database if not already done
flask init-db || true

# Run the command specified in CMD
exec "$@"