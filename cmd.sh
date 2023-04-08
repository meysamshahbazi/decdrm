export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
export DISPLAY=:0
sudo jetson_clocks

cmake -DCMAKE_BUILD_TYPE=Release ..

sudo usermod -a -G dialout $USER
# reboot 
# disable secend display 
# sleep time to never
# add to bashrc

sudo cp dvr.service /etc/systemd/system/
sudo chmod  744 run.sh 
sudo chmod 664 /etc/systemd/system/dvr.service
sudo systemctl daemon-reload 
sudo systemctl enable dvr.service

# -------------------------------------------------------
sudo systemctl set-default multi-user.target
sudo systemctl reboot

sudo systemctl set-default graphical.target
sudo reboot

auth  [success=ignore default=1] pam_succeed_if.so user = nvidia
auth  sufficient                 pam_succeed_if.so use_uid user = root


mkdir ~/.config/autostart
nano .desktop

[Desktop Entry]
Type=Application
Name=<Name of application as displayed>
Exec=/home/user/v4l2nv-j/run.sh
Icon=<full path to icon>
Comment=<optinal comments>
X-GNOME-Autostart-enabled=true


# for out of range clock 
find ./ -type f -exec touch {} +



