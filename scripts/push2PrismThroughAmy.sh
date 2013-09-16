export SSHPASS=AAG44aaF
sshpass -e sftp -oBatchMode=no -b - amonszpa@amy.cs.ucl.ac.uk << !
   put -r $1 /cs/research/vecg/prism/data3/amonszpart/$2
   bye
!

