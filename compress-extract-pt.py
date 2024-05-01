import gzip
import os

def compress(sd_path):
  # Compress the file
  with open(sd_path+".pt", 'rb') as f_in:
      with gzip.open(f"{sd_path}-compressed.pt.gz", 'wb') as f_out:
          f_out.writelines(f_in)
  # remove the original file after compression
  os.remove(sd_path+".pt")  

def extract(cmp_path):
   # Load from the compressed file
  with gzip.open(cmp_path+".pt.gz", 'rb') as f_in:
      with open(cmp_path[:11]+".pt", 'wb') as f_out:
          f_out.writelines(f_in)


file = "save-50x32-actual"
#compress(f"./saved_models/{file}")
