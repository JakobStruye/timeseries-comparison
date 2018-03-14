import os
folder = os.path.basename(os.getcwd())

if folder == "paper_settings":
    print "Run me from outside paper_settings please!"
    exit(1)

