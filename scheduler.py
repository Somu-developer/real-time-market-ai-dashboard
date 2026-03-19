import time
import os
while True:
    print("Retraining model...")
    os.system("python train.py")
    print("Models Updated. Waiting for the next update...\n")
    time.sleep(7200)  #2 hour 