#!/bin/bash

# Configuration - Update these values for your EC2 instance
EC2_USER="ubuntu"  # or "ec2-user" depending on your AMI
EC2_HOST="your-ec2-public-ip"
KEY_PATH="path/to/your/key.pem"
APP_DIR="/home/ubuntu/energy-dashboard"

echo "🚀 Starting deployment to EC2..."

# Step 1: Copy updated files to EC2
echo "📁 Copying files to EC2..."
scp -i $KEY_PATH app.py requirements.txt $EC2_USER@$EC2_HOST:$APP_DIR/

# Step 2: Connect and restart services
echo "🔄 Connecting to EC2 and restarting services..."
ssh -i $KEY_PATH $EC2_USER@$EC2_HOST << 'EOF'
cd /home/ubuntu/energy-dashboard

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install/update dependencies
pip install -r requirements.txt

# Kill existing streamlit processes
pkill -f "streamlit run"

# Start streamlit in background
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &

echo "✅ Application restarted successfully!"
echo "📊 Dashboard available at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8501"
EOF

echo "🎉 Deployment completed!"
