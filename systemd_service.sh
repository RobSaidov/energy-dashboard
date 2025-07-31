#!/bin/bash

# Create systemd service file for the energy dashboard
sudo tee /etc/systemd/system/energy-dashboard.service > /dev/null << EOF
[Unit]
Description=Energy Consumption Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/energy-dashboard
Environment=PATH=/home/ubuntu/energy-dashboard/venv/bin
ExecStart=/home/ubuntu/energy-dashboard/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable energy-dashboard
sudo systemctl start energy-dashboard

echo "✅ Systemd service created and started!"
echo "Use 'sudo systemctl status energy-dashboard' to check status"
echo "Use 'sudo systemctl restart energy-dashboard' to restart"
