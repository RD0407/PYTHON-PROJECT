from pytube import YouTube
from sys import argv

link = argv[1]
yT= YouTube(link)

print("Title: ", yT.title)
print("Views: ", yT.views)

dwnld = yT.streams.get_highest_resolution()
#dwnld.download('') 
