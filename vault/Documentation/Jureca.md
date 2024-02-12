
- Use wsl/ubuntu to generate a key using:
	-  ssh-keygen -t ed25519 -a 100 -f ~/.ssh/id_ed25519
	- get public key by: cat ~/.ssh/id_ed25519.pub
- Upload public key with IP address on:  https://judoor.fz-juelich.de/account/a/JSC_LDAP/pierschke1/
- Connect with jureca using 
	- ssh -4 -i ~/.ssh/id_ed25519 pierschke1@jureca.fz-juelich.de
- For copying data from jureca to local machine use the following example (NOTE: Run this locally, not when already connected to jureca):
	- robin@DESKTOP-40VDPSE:/mnt/c/Users/robin/Desktop$ scp pierschke1@jureca.fz-juelich.de:/p/project/cjinm17/kropp1/robin_deformation_data/reg_stack/B20_0524_Slice15_transformed.tif .